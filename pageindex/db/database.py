from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
from pageindex.env_settings import settings

engine = create_engine(
    settings.DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def _first_column(conn, table: str) -> str | None:
    row = conn.execute(
        text(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = :table
            ORDER BY ordinal_position
            LIMIT 1
            """
        ),
        {"table": table},
    ).fetchone()
    return row[0] if row else None


def _column_exists(conn, table: str, column: str) -> bool:
    row = conn.execute(
        text(
            """
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = :table AND column_name = :col
            """
        ),
        {"table": table, "col": column},
    ).fetchone()
    return row is not None


def _dependent_foreign_keys(conn, referenced_table: str) -> list[dict]:
    """
    Return FK constraints from other tables that reference referenced_table.
    Each item includes schema, table, constraint name, and constraint definition.
    """
    rows = conn.execute(
        text(
            """
            SELECT
              n.nspname AS schema_name,
              c.relname AS table_name,
              con.conname AS constraint_name,
              pg_get_constraintdef(con.oid) AS constraint_def
            FROM pg_constraint con
            JOIN pg_class c ON c.oid = con.conrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE con.contype = 'f'
              AND con.confrelid = ('public.' || :ref_table)::regclass
            ORDER BY n.nspname, c.relname, con.conname
            """
        ),
        {"ref_table": referenced_table},
    ).fetchall()
    out: list[dict] = []
    for r in rows:
        out.append(
            {
                "schema": r[0],
                "table": r[1],
                "name": r[2],
                "def": r[3],
            }
        )
    return out


def _drop_fk(conn, fk: dict) -> None:
    conn.execute(
        text(
            f'ALTER TABLE "{fk["schema"]}"."{fk["table"]}" DROP CONSTRAINT "{fk["name"]}"'
        )
    )


def _add_fk(conn, fk: dict) -> None:
    conn.execute(
        text(
            f'ALTER TABLE "{fk["schema"]}"."{fk["table"]}" ADD CONSTRAINT "{fk["name"]}" {fk["def"]}'
        )
    )


def _compact_jobs_seq_id(conn) -> None:
    """Assign seq_id 1..N on document_jobs by created_at (all rows, not only NULL)."""
    conn.execute(
        text(
            """
            WITH ranked AS (
                SELECT doc_id,
                       ROW_NUMBER() OVER (ORDER BY created_at ASC NULLS LAST, doc_id) AS rn
                FROM document_jobs
            )
            UPDATE document_jobs dj
            SET seq_id = ranked.rn
            FROM ranked
            WHERE dj.doc_id = ranked.doc_id
            """
        )
    )


def _sync_nodes_seq_from_jobs(conn) -> None:
    conn.execute(
        text(
            """
            UPDATE document_nodes dn
            SET seq_id = dj.seq_id,
                file_name = dj.file_name
            FROM document_jobs dj
            WHERE dj.doc_id = dn.doc_id
            """
        )
    )


def _apply_sortable_node_ids(conn) -> None:
    """Replace random UUID primary keys with ids that sort by (seq_id, node_id)."""
    from pageindex.db.sort_ids import sortable_node_row_id

    rows = conn.execute(
        text("SELECT id, seq_id, node_id FROM document_nodes")
    ).fetchall()
    id_map: dict[str, str] = {}
    for old_id, seq_id, node_id in rows:
        new_id = sortable_node_row_id(seq_id, node_id)
        if str(old_id) != new_id:
            id_map[str(old_id)] = new_id
    if not id_map:
        return

    deps = _dependent_foreign_keys(conn, "document_nodes")
    for fk in deps:
        _drop_fk(conn, fk)

    has_retrieved = conn.execute(
        text(
            """
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = 'retrieved_nodes'
            """
        )
    ).fetchone()

    for old_id, new_id in id_map.items():
        if has_retrieved:
            conn.execute(
                text(
                    "UPDATE retrieved_nodes SET document_node_id = :new WHERE document_node_id = :old"
                ),
                {"new": new_id, "old": old_id},
            )
        conn.execute(
            text("UPDATE document_nodes SET id = :new WHERE id = :old"),
            {"new": new_id, "old": old_id},
        )

    for fk in deps:
        _add_fk(conn, fk)


def _drop_table_primary_key(conn, table: str) -> None:
    row = conn.execute(
        text(
            """
            SELECT con.conname
            FROM pg_constraint con
            JOIN pg_class rel ON rel.oid = con.conrelid
            JOIN pg_namespace n ON n.oid = rel.relnamespace
            WHERE n.nspname = 'public' AND rel.relname = :table AND con.contype = 'p'
            """
        ),
        {"table": table},
    ).fetchone()
    if row:
        conn.execute(text(f'ALTER TABLE "{table}" DROP CONSTRAINT "{row[0]}"'))


def _set_jobs_primary_key_on_seq_id(conn) -> None:
    """Use seq_id as PK so pgAdmin default sort is ascending document order."""
    row = conn.execute(
        text(
            """
            SELECT a.attname
            FROM pg_index i
            JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
            WHERE i.indrelid = 'public.document_jobs'::regclass
              AND i.indisprimary
            LIMIT 1
            """
        )
    ).fetchone()
    if row and row[0] == "seq_id":
        return

    conn.execute(
        text(
            "ALTER TABLE document_nodes DROP CONSTRAINT IF EXISTS document_nodes_doc_id_fkey"
        )
    )
    conn.execute(text("DROP INDEX IF EXISTS ux_document_jobs_seq_id"))
    _drop_table_primary_key(conn, "document_jobs")
    conn.execute(text("ALTER TABLE document_jobs ADD PRIMARY KEY (seq_id)"))
    conn.execute(
        text(
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_document_jobs_doc_id ON document_jobs(doc_id)"
        )
    )


def _rebuild_document_nodes_sorted(conn) -> None:
    """Recreate document_nodes with rows inserted in seq_id, node_id order."""
    deps = _dependent_foreign_keys(conn, "document_nodes")
    for fk in deps:
        _drop_fk(conn, fk)
    conn.execute(
        text(
            "ALTER TABLE document_nodes DROP CONSTRAINT IF EXISTS document_nodes_doc_id_fkey"
        )
    )

    conn.execute(
        text(
            """
            CREATE TABLE document_nodes_sorted AS
            SELECT * FROM document_nodes
            ORDER BY seq_id ASC NULLS LAST, node_id ASC
            """
        )
    )
    conn.execute(text("DROP TABLE document_nodes"))
    conn.execute(text("ALTER TABLE document_nodes_sorted RENAME TO document_nodes"))
    conn.execute(text("ALTER TABLE document_nodes ADD PRIMARY KEY (id)"))
    conn.execute(
        text(
            "ALTER TABLE document_nodes ADD CONSTRAINT document_nodes_doc_id_fkey "
            "FOREIGN KEY (doc_id) REFERENCES document_jobs(doc_id) ON DELETE CASCADE"
        )
    )
    conn.execute(
        text("CREATE INDEX IF NOT EXISTS ix_document_nodes_doc_id ON document_nodes(doc_id)")
    )
    conn.execute(
        text("CREATE INDEX IF NOT EXISTS ix_document_nodes_seq_id ON document_nodes(seq_id)")
    )
    conn.execute(
        text(
            "CREATE INDEX IF NOT EXISTS ix_document_nodes_doc_node ON document_nodes(doc_id, node_id)"
        )
    )
    conn.execute(
        text(
            "CREATE INDEX IF NOT EXISTS ix_document_nodes_seq_node ON document_nodes(seq_id, node_id)"
        )
    )

    for fk in deps:
        _add_fk(conn, fk)


def _cluster_sorted_tables(conn) -> None:
    conn.execute(
        text(
            "CREATE INDEX IF NOT EXISTS ix_document_nodes_seq_node ON document_nodes(seq_id, node_id)"
        )
    )
    pk = conn.execute(
        text(
            """
            SELECT indexrelid::regclass::text
            FROM pg_index
            WHERE indrelid = 'public.document_jobs'::regclass AND indisprimary
            """
        )
    ).scalar()
    if pk:
        conn.execute(text(f"CLUSTER document_jobs USING {pk}"))
    conn.execute(text("CLUSTER document_nodes USING ix_document_nodes_seq_node"))


def _migrate_schema_sort_v3(conn) -> None:
    _compact_jobs_seq_id(conn)
    _sync_nodes_seq_from_jobs(conn)
    _renumber_all_document_nodes(conn)
    _apply_sortable_node_ids(conn)
    _set_jobs_primary_key_on_seq_id(conn)
    _rebuild_document_nodes_sorted(conn)
    _cluster_sorted_tables(conn)


def _renumber_all_document_nodes(conn) -> None:
    """Set seq_id from document_jobs and node_id to 0001..N per doc in ascending tree order."""
    from pageindex.db.node_order import ordered_flat_nodes

    docs = conn.execute(
        text("SELECT doc_id, seq_id FROM document_jobs ORDER BY seq_id ASC NULLS LAST")
    ).fetchall()
    for doc_id, job_seq in docs:
        rows = conn.execute(
            text(
                """
                SELECT id, node_json
                FROM document_nodes
                WHERE doc_id = :doc_id
                ORDER BY node_id ASC
                """
            ),
            {"doc_id": doc_id},
        ).fetchall()
        if not rows:
            continue

        items: list[tuple[str, dict]] = []
        for rid, njson in rows:
            if isinstance(njson, dict):
                items.append((rid, dict(njson)))

        ordered = ordered_flat_nodes([n for _, n in items])
        id_map: dict[str, str] = {}
        for i, n in enumerate(ordered, start=1):
            old = str(n.get("_original_node_id") or n.get("node_id") or "")
            id_map[old] = f"{i:04d}"

        rid_by_old: dict[str, str] = {}
        for rid, njson in items:
            old = str(njson.get("_original_node_id") or njson.get("node_id") or "")
            if old:
                rid_by_old[old] = rid

        for n in ordered:
            old = str(n.get("_original_node_id") or n.get("node_id") or "")
            if not old:
                continue
            pid = n.get("_original_parent_id")
            if pid is None:
                pid = n.get("parent_id")
            pid_str = str(pid) if pid is not None else ""
            new_parent = id_map.get(pid_str) if pid is not None else None
            row_id = rid_by_old.get(old)
            if not row_id:
                continue
            conn.execute(
                text(
                    """
                    UPDATE document_nodes
                    SET seq_id = :seq_id,
                        node_id = :node_id,
                        parent_id = :parent_id
                    WHERE id = :id
                    """
                ),
                {
                    "seq_id": int(job_seq) if job_seq is not None else None,
                    "node_id": id_map.get(old),
                    "parent_id": new_parent,
                    "id": row_id,
                },
            )


def _reorder_tables(conn) -> None:
    """Recreate both tables with seq_id, doc_id, file_name as the first three columns."""
    # Drop any external FKs referencing these tables (e.g. retrieved_nodes -> document_nodes)
    deps_nodes = _dependent_foreign_keys(conn, "document_nodes")
    deps_jobs = _dependent_foreign_keys(conn, "document_jobs")
    for fk in deps_nodes + deps_jobs:
        _drop_fk(conn, fk)

    conn.execute(
        text(
            """
            CREATE TABLE document_jobs_new (
                seq_id INTEGER,
                doc_id VARCHAR PRIMARY KEY,
                file_name VARCHAR,
                status VARCHAR NOT NULL DEFAULT 'pending',
                error_message TEXT,
                results JSONB,
                created_at TIMESTAMPTZ DEFAULT now(),
                updated_at TIMESTAMPTZ
            )
            """
        )
    )
    conn.execute(
        text(
            """
            INSERT INTO document_jobs_new (
                seq_id, doc_id, file_name, status, error_message, results, created_at, updated_at
            )
            SELECT seq_id, doc_id, file_name, status, error_message, results, created_at, updated_at
            FROM document_jobs
            """
        )
    )

    # SELECT uses COALESCE for legacy job_seq_id column name during copy
    nodes_seq_col = "seq_id" if _column_exists(conn, "document_nodes", "seq_id") else "job_seq_id"
    conn.execute(
        text(
            f"""
            CREATE TABLE document_nodes_new (
                seq_id INTEGER,
                doc_id VARCHAR NOT NULL,
                file_name VARCHAR,
                id VARCHAR PRIMARY KEY,
                node_id VARCHAR NOT NULL,
                parent_id VARCHAR,
                type VARCHAR NOT NULL,
                title VARCHAR NOT NULL,
                path TEXT,
                level INTEGER NOT NULL,
                raw_content TEXT,
                compressed_content TEXT,
                micro_summary TEXT,
                content_hash VARCHAR,
                retrieval_ready BOOLEAN DEFAULT FALSE,
                is_front_matter BOOLEAN DEFAULT FALSE,
                metadata_json JSONB,
                node_json JSONB,
                created_at TIMESTAMPTZ DEFAULT now()
            )
            """
        )
    )
    conn.execute(
        text(
            f"""
            INSERT INTO document_nodes_new (
                seq_id, doc_id, file_name, id, node_id, parent_id, type, title, path, level,
                raw_content, compressed_content, micro_summary, content_hash,
                retrieval_ready, is_front_matter, metadata_json, node_json, created_at
            )
            SELECT
                {nodes_seq_col}, doc_id, file_name, id, node_id, parent_id, type, title, path, level,
                raw_content, compressed_content, micro_summary, content_hash,
                retrieval_ready, is_front_matter, metadata_json, node_json, created_at
            FROM document_nodes
            """
        )
    )

    conn.execute(text("DROP TABLE document_nodes"))
    conn.execute(text("DROP TABLE document_jobs"))
    conn.execute(text("ALTER TABLE document_jobs_new RENAME TO document_jobs"))
    conn.execute(text("ALTER TABLE document_nodes_new RENAME TO document_nodes"))
    conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ux_document_jobs_seq_id ON document_jobs(seq_id)"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS ix_document_nodes_doc_id ON document_nodes(doc_id)"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS ix_document_nodes_seq_id ON document_nodes(seq_id)"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS ix_document_nodes_doc_node ON document_nodes(doc_id, node_id)"))

    # Re-add external FKs
    for fk in deps_jobs + deps_nodes:
        _add_fk(conn, fk)


def ensure_schema() -> None:
    """
    Lightweight startup migration for this repo.
    Adds new columns/indexes if missing, backfills seq_id, and reorders columns
    so seq_id is 1st, doc_id 2nd, file_name 3rd in pgAdmin.
    """
    dialect = engine.dialect.name
    if dialect != "postgresql":
        # SQLite/dev: column order in pgAdmin is not applicable; keep additive migrations only.
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE document_jobs ADD COLUMN IF NOT EXISTS seq_id INTEGER"))
            conn.execute(text("ALTER TABLE document_jobs ADD COLUMN IF NOT EXISTS file_name VARCHAR"))
            conn.execute(text("ALTER TABLE document_nodes ADD COLUMN IF NOT EXISTS seq_id INTEGER"))
            conn.execute(text("ALTER TABLE document_nodes ADD COLUMN IF NOT EXISTS file_name VARCHAR"))
        return

    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE document_jobs ADD COLUMN IF NOT EXISTS seq_id INTEGER"))
        conn.execute(text("ALTER TABLE document_jobs ADD COLUMN IF NOT EXISTS file_name VARCHAR"))
        if _column_exists(conn, "document_nodes", "job_seq_id") and not _column_exists(conn, "document_nodes", "seq_id"):
            conn.execute(text("ALTER TABLE document_nodes RENAME COLUMN job_seq_id TO seq_id"))
        conn.execute(text("ALTER TABLE document_nodes ADD COLUMN IF NOT EXISTS seq_id INTEGER"))
        conn.execute(text("ALTER TABLE document_nodes ADD COLUMN IF NOT EXISTS file_name VARCHAR"))

        conn.execute(
            text(
                """
                WITH ranked AS (
                    SELECT doc_id,
                           ROW_NUMBER() OVER (ORDER BY created_at, doc_id) AS rn
                    FROM document_jobs
                )
                UPDATE document_jobs dj
                SET seq_id = ranked.rn
                FROM ranked
                WHERE dj.doc_id = ranked.doc_id
                  AND dj.seq_id IS NULL
                """
            )
        )

        conn.execute(
            text(
                """
                UPDATE document_nodes dn
                SET seq_id = (SELECT seq_id FROM document_jobs dj WHERE dj.doc_id = dn.doc_id),
                    file_name  = (SELECT file_name FROM document_jobs dj WHERE dj.doc_id = dn.doc_id)
                WHERE dn.file_name IS NULL OR dn.seq_id IS NULL
                """
            )
        )

        needs_reorder = (
            _first_column(conn, "document_jobs") != "seq_id"
            or _first_column(conn, "document_nodes") != "seq_id"
        )
        if needs_reorder:
            _reorder_tables(conn)

        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS app_migrations (
                    key VARCHAR PRIMARY KEY,
                    value VARCHAR,
                    created_at TIMESTAMPTZ DEFAULT now()
                )
                """
            )
        )
        already_v2 = conn.execute(
            text("SELECT value FROM app_migrations WHERE key='nodes_resort_v2'")
        ).fetchone()
        if not already_v2:
            _renumber_all_document_nodes(conn)
            conn.execute(
                text(
                    "INSERT INTO app_migrations(key, value) VALUES('nodes_resort_v2', 'done')"
                )
            )

        already_v3 = conn.execute(
            text("SELECT value FROM app_migrations WHERE key='schema_sort_v3'")
        ).fetchone()
        if not already_v3:
            _migrate_schema_sort_v3(conn)
            conn.execute(
                text(
                    "INSERT INTO app_migrations(key, value) VALUES('schema_sort_v3', 'done')"
                )
            )


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
