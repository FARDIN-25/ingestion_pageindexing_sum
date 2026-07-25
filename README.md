# Document Ingestion — PageIndex + Vector Embeddings

Local FastAPI app for PDF ingestion with **two independent pipelines**:

| Mode | What it does | Where data is stored |
|------|----------------|----------------------|
| **PageIndex** | Cloud tree indexing (vectorless RAG) | `document_jobs` + `document_nodes` |
| **Vector** | Local chunking + embeddings | `vectors` only |
| **Hybrid** | Runs both paths separately | Both sets of tables (same `doc_id`) |

The pipelines do **not** share tables. Vector never writes to PageIndex tables, and PageIndex never writes to `vectors`.

---

## Features

- Drag-and-drop PDF upload UI (`static/index.html`)
- Processing modes: **PageIndex**, **Vector**, **Hybrid**
- Vector document types:
  - **Book** — sliding-window chunking + overlap (or no chunking)
  - **Notice / Reply** — GST notice+reply split; embeds notice only
  - **FAQ** — Q/A pair extraction; embeds questions only
- Live job queue with separate PageIndex / Vector progress
- Local embedding model via `sentence-transformers` (default: Qwen3-Embedding-0.6B)
- PostgreSQL + `pgvector` for vector storage
- Usage / credit dashboard at `/usage`
- PostgreSQL + `pgvector` for vector storage

---

## Software & prerequisites

| Software | Version / notes | Purpose |
|----------|-----------------|---------|
| **Python** | 3.11+ recommended | Runtime |
| **PostgreSQL** | 14+ | Job / node / vector storage |
| **pgvector** | Extension on Postgres | Embedding column + HNSW index |
| **Git** | Any recent | Clone / version control |
| **PageIndex API key** | From [dash.pageindex.ai](https://dash.pageindex.ai/api-keys) | PageIndex cloud indexing |
| **CUDA (optional)** | If `EMBEDDING_DEVICE=cuda` | Faster local embeddings |

### Python packages (see `requirements.txt`)

| Package | Role |
|---------|------|
| `fastapi`, `uvicorn`, `python-multipart` | Web API + server |
| `pymupdf` | PDF text extraction |
| `SQLAlchemy`, `psycopg2-binary`, `alembic` | ORM, Postgres driver, migrations |
| `pgvector` | Vector type for SQLAlchemy |
| `sentence-transformers`, `torch` | Local embedding model |
| `pydantic-settings`, `python-dotenv` | Config / `.env` |
| `requests`, `pyyaml` | HTTP + config helpers |
| `pytest` | Tests |

---

## Architecture

```
PDF upload (UI)
      │
      ├─ mode=pageindex ──► pageindex_upload_service
      │                         └─► document_jobs / document_nodes
      │
      ├─ mode=vector ──────► ingest_document(doc_type)
      │                         ├─ book        → vector_service  → vectors
      │                         ├─ notice_reply → notice_faq_ingestion → vectors
      │                         └─ faq          → notice_faq_ingestion → vectors
      │
      └─ mode=hybrid ─────► both paths in parallel (separate tables)
```

### Key modules

| Path | Responsibility |
|------|----------------|
| `app.py` | FastAPI routes, upload fan-out by mode |
| `run_pageindex_server.py` | Start uvicorn (preferred entrypoint) |
| `pageindex/pageindex_upload_service.py` | PageIndex cloud → jobs/nodes |
| `pageindex/vector_service.py` | Book chunking + embed → `vectors` only |
| `pageindex/notice_faq_ingestion.py` | Notice/reply + FAQ vector ingest |
| `pageindex/embedding_service.py` | Singleton local embedding model |
| `pageindex/vector_chunking.py` | Chunk size / overlap / content hash |
| `pageindex/db/models.py` | `DocumentJob`, `DocumentNode`, `Vector` |
| `pageindex/db/database.py` | Engine + schema ensure (incl. pgvector) |
| `static/index.html` | Upload UI |

---

## Database tables

### PageIndex path

- **`document_jobs`** — one row per PageIndex (or hybrid) job  
- **`document_nodes`** — hierarchical tree nodes from PageIndex

### Vector path (standalone)

**`vectors`** — no foreign keys to PageIndex tables.

| Column | Notes |
|--------|--------|
| `id` | UUID primary key |
| `doc_id` | Correlates uploads across paths |
| `job_id` | Free integer for the vector run (not `document_jobs.seq_id`) |
| `node_id` | Chunk / role key (`v0000`, `nr_notice`, `fq0001_q`, …) |
| `chunk_index` | Order within the doc |
| `chunk_text` | Stored text |
| `content_hash` | SHA-256 for dedup |
| `embedding` | `vector(N)` (nullable for reply/answer rows) |
| `embedding_dim`, `model_name` | Nullable when `status=not_embedded` |
| `status` | e.g. `completed`, `embedded`, `not_embedded`, `pending` |
| `chunk_meta` | JSONB metadata (`doc_type`, `role`, pair ids, chunk settings, …) |

Example `chunk_meta` for notice/reply:

```json
{
  "doc_type": "notice_reply",
  "role": "notice",
  "pair_id": "ZD330723034368E",
  "notice_section": "73",
  "tax_period": "JUL 2017 - MAR 2018",
  "issue_category": "itc_mismatch"
}
```

---

## Setup

### 1. Clone and create a virtualenv

```powershell
cd d:\LLM\ingestion_pageindexing_sum
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. PostgreSQL + pgvector

1. Install PostgreSQL and enable the extension:

```sql
CREATE DATABASE your_db;
\c your_db
CREATE EXTENSION IF NOT EXISTS vector;
```

2. Ensure the DB user can create extensions / tables.

### 3. Configure `.env`

Copy `.env.example` → `.env` and fill in values:

```env
PAGEINDEX_API_KEY=your_pageindex_api_key_here
DATABASE_URL=postgresql://user:password@localhost:5432/your_db

# Embedding (local)
EMBEDDING_MODEL_NAME=Qwen/Qwen3-Embedding-0.6B
EMBEDDING_DIM=1024
EMBEDDING_BATCH_SIZE=32
EMBEDDING_DEVICE=cpu

# Book vector chunking defaults
VECTOR_CHUNK_SIZE=1000
VECTOR_CHUNK_OVERLAP=200
```

Optional: set `PAGEINDEX_API_BASE` if you use a non-default API host.

### 4. Migrations (optional)

Schema is also ensured on server startup. To run Alembic explicitly:

```powershell
alembic upgrade head
```

Migrations live under `alembic/versions/` (including standalone `vectors` without PageIndex FKs).

---

## Run the server

**Do not** use `python app.py` as the entrypoint for serving (it imports and exits). Use:

```powershell
.\venv\Scripts\Activate.ps1
python run_pageindex_server.py
```

Then open:

- Upload UI: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Usage dashboard: [http://127.0.0.1:8000/usage](http://127.0.0.1:8000/usage)
- Health: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

Override port if needed:

```powershell
$env:PAGEINDEX_PORT="8090"
python run_pageindex_server.py
```

Logs go to the terminal and `logs/app.log`.

---

## Using the upload UI

1. Choose a **processing mode**:
   - **PageIndex** — tree indexing only  
   - **Vector** — embeddings only  
   - **Hybrid** — both in parallel  
2. If Vector or Hybrid is selected, choose a **document type**:
   - **Book** — editable chunk size / overlap, or “No chunking”  
   - **Notice / Reply** — expects pages marked `GST / NOTICE` then `GST / REPLY`  
   - **FAQ** — expects `Q:`/`Question:` + `A:`/`Answer:` blocks  
3. Drag & drop a PDF (or click to browse).
4. Watch the live queue for separate PageIndex / Vector progress.

### Notice / Reply behavior

- Keyword detector only (no LLM): state persists across pages after `GST / NOTICE` / `GST / REPLY`
- Notice row: embedded (`status=embedded`)
- Reply row: stored without embedding (`status=not_embedded`)
- Metadata (reference no, section, tax period, GSTIN, issue category) lives in `chunk_meta`

### FAQ behavior

- Embeds each question; stores answers without embeddings
- Q/A regex may need tuning for your FAQ layout (see TODO in `notice_faq_ingestion.py`)

---

## Main API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/process` | Upload PDF (`multipart`: `file`, `mode`, `doc_type`, chunk options) |
| `GET` | `/api/ingestion/jobs` | Live queue (PageIndex jobs + in-memory vector jobs) |
| `GET` | `/api/ingestion/jobs/{doc_id}` | PageIndex job detail + tree |
| `GET` | `/api/ingestion/jobs/{doc_id}/tree` | Tree view for completed PageIndex jobs |
| `POST` | `/api/search` | Local VRAG search over a structure |
| `GET` | `/api/health` | Health check |
| `GET` | `/api/usage/*` | Credit / usage APIs |
| `POST` | `/api/ingestion/ingest/run` | Disabled (410) — reserved; upload UI only |

### `POST /api/process` form fields

| Field | Values / default |
|-------|------------------|
| `file` | PDF (required) |
| `mode` | `pageindex` \| `vector` \| `hybrid` (default `hybrid`) |
| `doc_type` | `book` \| `notice_reply` \| `faq` (default `book`; Vector path) |
| `chunk_size` | int (book only; default from env) |
| `chunk_overlap` | int (book only) |
| `no_chunking` | `true` / `false` (book only) |

---

## Project layout (high level)

```
ingestion_pageindexing_sum/
├── app.py                      # FastAPI application
├── run_pageindex_server.py     # Server entrypoint
├── requirements.txt
├── .env.example
├── alembic.ini
├── alembic/versions/           # Schema migrations
├── static/
│   ├── index.html              # Upload + queue UI
│   └── usage.html              # Credits dashboard
├── pageindex/
│   ├── embedding_service.py
│   ├── vector_service.py
│   ├── vector_chunking.py
│   ├── notice_faq_ingestion.py
│   ├── pageindex_upload_service.py
│   ├── env_settings.py
│   ├── db/                     # models, repository, schema ensure
│   ├── vrag/                   # Vectorless RAG helpers
│   └── usage/                  # Metering / credits
├── tests/
└── logs/                       # Runtime logs (created at run)
```

---

## Development notes

- **Separation rule:** Vector path → `vectors` only. PageIndex path → `document_jobs` / `document_nodes` only. Hybrid runs both independently.
- First embedding run may download the Hugging Face model (`EMBEDDING_MODEL_NAME`); allow time and disk space.
- Prefer `cpu` until CUDA + matching `torch` are verified.
- Tree viewer in the UI applies to completed **PageIndex** trees, not vector-only jobs.
- Tests: `pytest` (see `tests/`).

---

## Troubleshooting

| Issue | What to check |
|-------|----------------|
| Port in use | Set `PAGEINDEX_PORT`, or free 8000 |
| `DATABASE_URL` errors | Postgres running; credentials; DB exists |
| `vector` type missing | `CREATE EXTENSION vector;` |
| Embedding OOM / slow | Lower `EMBEDDING_BATCH_SIZE`; use `cpu` |
| Notice/reply not found | PDF must contain exact `GST / NOTICE` and `GST / REPLY` |
| Book chunks but expected notice meta | Upload with **Vector** + **Notice / Reply**, not Book |
| No reply row / empty `chunk_meta` roles | Wrong doc type (Book path) or old rows before notice/faq code |

---

## License / upstream

This project builds on [PageIndex](https://vectify.ai/pageindex) concepts (vectorless, reasoning-based RAG) and adds a **local vector embedding pipeline** for book / notice-reply / FAQ document types. See Vectify AI / PageIndex docs for the upstream cloud product and API.
