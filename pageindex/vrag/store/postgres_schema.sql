-- Optional PostgreSQL schema for vectorless RAG (FTS + hierarchy)
-- Requires: CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS documents (
    doc_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_pdf      TEXT NOT NULL,
    schema_version  TEXT NOT NULL DEFAULT '2.1',
    page_count      INT NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS index_nodes (
    node_id                 TEXT PRIMARY KEY,
    doc_id                  UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    parent_id               TEXT REFERENCES index_nodes(node_id) ON DELETE CASCADE,
    type                    TEXT NOT NULL,
    level                   SMALLINT NOT NULL DEFAULT 0,
    title                   TEXT NOT NULL,
    path                    TEXT NOT NULL,
    raw_content             TEXT NOT NULL DEFAULT '',
    compressed_content      TEXT NOT NULL DEFAULT '',
    micro_summary           TEXT NOT NULL DEFAULT '',
    content_hash            TEXT NOT NULL DEFAULT '',
    page_start              INT NOT NULL DEFAULT 0,
    page_end                INT NOT NULL DEFAULT 0,
    char_start              INT NOT NULL DEFAULT 0,
    char_end                INT NOT NULL DEFAULT 0,
    token_count_raw         INT NOT NULL DEFAULT 0,
    token_count_compressed  INT NOT NULL DEFAULT 0,
    is_retrieval_chunk      BOOLEAN NOT NULL DEFAULT FALSE,
    is_front_matter         BOOLEAN NOT NULL DEFAULT FALSE,
    aliases                 JSONB NOT NULL DEFAULT '[]',
    keywords                JSONB NOT NULL DEFAULT '[]',
    synonyms                JSONB NOT NULL DEFAULT '[]',
    children_ids            JSONB NOT NULL DEFAULT '[]',
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_nodes_doc ON index_nodes(doc_id);
CREATE INDEX IF NOT EXISTS idx_nodes_parent ON index_nodes(parent_id);
CREATE INDEX IF NOT EXISTS idx_nodes_hash ON index_nodes(content_hash);
CREATE INDEX IF NOT EXISTS idx_nodes_retrieval ON index_nodes(doc_id) WHERE is_retrieval_chunk;

-- Full-text search on compressed (primary) and raw (secondary)
ALTER TABLE index_nodes ADD COLUMN IF NOT EXISTS fts_compressed tsvector
    GENERATED ALWAYS AS (to_tsvector('english', coalesce(compressed_content, ''))) STORED;
ALTER TABLE index_nodes ADD COLUMN IF NOT EXISTS fts_raw tsvector
    GENERATED ALWAYS AS (to_tsvector('english', coalesce(raw_content, ''))) STORED;

CREATE INDEX IF NOT EXISTS idx_nodes_fts_compressed ON index_nodes USING GIN (fts_compressed);
CREATE INDEX IF NOT EXISTS idx_nodes_fts_raw ON index_nodes USING GIN (fts_raw);
CREATE INDEX IF NOT EXISTS idx_nodes_title_trgm ON index_nodes USING GIN (title gin_trgm_ops);
