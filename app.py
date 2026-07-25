import asyncio
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# Unbuffered output in terminal
os.environ.setdefault("PYTHONUNBUFFERED", "1")

from dotenv import load_dotenv
from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pageindex.log_util import log_error, log_exception, log_info, setup_logging
from pageindex.config import ConfigLoader, setup_pageindex_env

load_dotenv()
setup_pageindex_env()

setup_logging()
log_info("=" * 60)
log_info("PageIndex app loaded — PageIndex + Vector pipelines")
log_info("OpenRouter is DISABLED")
log_info("=" * 60)

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
STATIC_DIR = BASE_DIR / "static"
LOG_FILE = BASE_DIR / "logs" / "app.log"

UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

executor = ThreadPoolExecutor(max_workers=1)


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _set_service_progress(store_name: str, doc_id: str, **fields: Any) -> None:
    store = getattr(app.state, store_name, None)
    if store is None:
        return
    current = dict(store.get(doc_id) or {})
    current.update(fields)
    current["updated_at"] = _utc_now_iso()
    store[doc_id] = current


def _set_pageindex_progress(doc_id: str, **fields: Any) -> None:
    _set_service_progress("pageindex_progress", doc_id, **fields)


def _set_vector_progress(doc_id: str, **fields: Any) -> None:
    _set_service_progress("vector_progress", doc_id, **fields)


def _run_pageindex_branch(pdf_path: Path, safe_name: str, doc_id: str) -> None:
    from pageindex.pageindex_upload_service import pageindex_upload_service

    def on_progress(event: dict[str, Any]) -> None:
        _set_pageindex_progress(doc_id, **{k: v for k, v in event.items() if k != "doc_id"})

    try:
        pageindex_upload_service.process_pdf(
            str(pdf_path),
            doc_id=doc_id,
            file_name=safe_name,
            results_dir=RESULTS_DIR,
            progress_callback=on_progress,
        )
    except Exception as exc:
        _set_pageindex_progress(doc_id, status="failed", message=str(exc)[:300])
        log_exception("[pageindex] branch failed doc_id=%s: %s", doc_id, exc)


async def _run_vector_branch(
    pdf_path: Path,
    safe_name: str,
    doc_id: str,
    job_id: int,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    no_chunking: bool = False,
    doc_type: str = "book",
) -> None:
    from pageindex.notice_faq_ingestion import ingest_document

    def on_progress(event: dict[str, Any]) -> None:
        _set_vector_progress(doc_id, **{k: v for k, v in event.items() if k != "doc_id"})

    doc_type_norm = (doc_type or "book").strip().lower()
    try:
        _set_vector_progress(
            doc_id,
            status="processing",
            message=f"Vector ingest ({doc_type_norm})...",
        )
        result = await ingest_document(
            str(pdf_path),
            doc_id,
            job_id,
            doc_type_norm,
            # book-only kwargs (ignored by notice_reply / faq)
            file_name=safe_name,
            progress_callback=on_progress,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            no_chunking=no_chunking,
        )
        if doc_type_norm != "book":
            # notice_reply / faq don't use vector_service progress callbacks
            msg = (
                f"Ingested {doc_type_norm}"
                if not isinstance(result, list)
                else f"Ingested {len(result)} FAQ pair(s)"
            )
            _set_vector_progress(
                doc_id,
                status="completed",
                message=msg,
                total=1 if not isinstance(result, list) else len(result),
                done=1 if not isinstance(result, list) else len(result),
            )
    except Exception as exc:
        _set_vector_progress(doc_id, status="failed", message=str(exc)[:300])
        log_exception("[vector] branch failed doc_id=%s doc_type=%s: %s", doc_id, doc_type_norm, exc)


def _finalize_job_status(doc_id: str, mode: str) -> None:
    """Finalize status per path.

    - pageindex / hybrid: update document_jobs (PageIndex table)
    - vector / hybrid: update in-memory vector_jobs registry only (no document_jobs write for vector-only)
    """
    from pageindex.db.database import SessionLocal
    from pageindex.db.repository import IngestionRepository

    pi = getattr(app.state, "pageindex_progress", {}).get(doc_id) or {}
    vec = getattr(app.state, "vector_progress", {}).get(doc_id) or {}
    pi_status = pi.get("status")
    vec_status = vec.get("status")

    vector_jobs = getattr(app.state, "vector_jobs", {})
    if doc_id in vector_jobs:
        if vec_status == "completed":
            vector_jobs[doc_id]["status"] = "completed"
            vector_jobs[doc_id]["error_message"] = None
        elif vec_status == "skipped":
            pass
        else:
            vector_jobs[doc_id]["status"] = "failed"
            vector_jobs[doc_id]["error_message"] = vec.get("message") or "Vector failed"

    # PageIndex path owns document_jobs — skip for vector-only.
    if mode == "vector":
        return

    if mode == "pageindex":
        final = "completed" if pi_status == "completed" else "failed"
        err = None if final == "completed" else (pi.get("message") or "PageIndex failed")
    else:  # hybrid
        if pi_status == "completed":
            final = "completed"
            err = None
        else:
            final = "failed"
            err = pi.get("message") or f"pageindex={pi_status}"

    db = SessionLocal()
    try:
        IngestionRepository(db).upsert_job(doc_id, status=final, error_message=err)
    finally:
        db.close()


def _process_upload_job(
    pdf_path: Path,
    safe_name: str,
    doc_id: str,
    job_id: int,
    mode: str = "hybrid",
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    no_chunking: bool = False,
    doc_type: str = "book",
    vector_job_id: int | None = None,
) -> None:
    """Run PageIndex and/or Vector based on selected mode (separate paths)."""
    from pageindex.env_settings import settings

    mode = (mode or "hybrid").strip().lower()
    if mode not in {"pageindex", "vector", "hybrid"}:
        mode = "hybrid"
    doc_type_norm = (doc_type or "book").strip().lower()
    if doc_type_norm not in {"book", "notice_reply", "faq"}:
        doc_type_norm = "book"

    use_chunk_size = int(chunk_size if chunk_size is not None else settings.VECTOR_CHUNK_SIZE)
    use_chunk_overlap = int(
        chunk_overlap if chunk_overlap is not None else settings.VECTOR_CHUNK_OVERLAP
    )
    vec_job = int(vector_job_id) if vector_job_id is not None else int(job_id)

    log_info("=" * 50)
    log_info(
        "PIPELINE START mode=%s doc_type=%s doc_id=%s file=%s pageindex_job=%s vector_job=%s",
        mode,
        doc_type_norm,
        doc_id,
        safe_name,
        job_id if mode in {"pageindex", "hybrid"} else None,
        vec_job if mode in {"vector", "hybrid"} else None,
    )
    if mode in {"pageindex", "hybrid"}:
        log_info("  -> PageIndex service -> document_jobs / document_nodes")
    if mode in {"vector", "hybrid"}:
        log_info(
            "  -> Vector service -> vectors only | doc_type=%s (no_chunking=%s chunk_size=%s overlap=%s)",
            doc_type_norm,
            no_chunking,
            use_chunk_size,
            use_chunk_overlap,
        )
    log_info("=" * 50)

    async def _run() -> None:
        tasks = []
        labels = []

        if mode == "pageindex":
            _set_pageindex_progress(doc_id, status="processing", message="PageIndex queued...")
            _set_vector_progress(doc_id, status="skipped", message="Skipped (PageIndex-only mode)")
            tasks.append(asyncio.to_thread(_run_pageindex_branch, pdf_path, safe_name, doc_id))
            labels.append("pageindex")
        elif mode == "vector":
            _set_pageindex_progress(doc_id, status="skipped", message="Skipped (Vector-only mode)")
            _set_vector_progress(doc_id, status="processing", message="Vector queued...")
            tasks.append(
                _run_vector_branch(
                    pdf_path,
                    safe_name,
                    doc_id,
                    vec_job,
                    chunk_size=use_chunk_size,
                    chunk_overlap=use_chunk_overlap,
                    no_chunking=no_chunking,
                    doc_type=doc_type_norm,
                )
            )
            labels.append("vector")
        else:
            _set_pageindex_progress(doc_id, status="processing", message="PageIndex queued...")
            _set_vector_progress(doc_id, status="processing", message="Vector queued...")
            tasks.append(asyncio.to_thread(_run_pageindex_branch, pdf_path, safe_name, doc_id))
            tasks.append(
                _run_vector_branch(
                    pdf_path,
                    safe_name,
                    doc_id,
                    vec_job,
                    chunk_size=use_chunk_size,
                    chunk_overlap=use_chunk_overlap,
                    no_chunking=no_chunking,
                    doc_type=doc_type_norm,
                )
            )
            labels.extend(["pageindex", "vector"])

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for label, res in zip(labels, results):
            if isinstance(res, Exception):
                log_error("Pipeline %s raised: %s", label, res)

    try:
        asyncio.run(_run())
        _finalize_job_status(doc_id, mode)
        log_info("PIPELINE FINISHED mode=%s doc_type=%s doc_id=%s", mode, doc_type_norm, doc_id)
    except Exception as exc:
        log_exception("Pipeline crashed mode=%s doc_id=%s: %s", mode, doc_id, exc)
        try:
            _finalize_job_status(doc_id, mode)
        except Exception:
            pass
    finally:
        try:
            pdf_path.unlink(missing_ok=True)
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    from pageindex.cloud import get_api_key

    key = get_api_key()
    masked = f"{key[:6]}...{key[-4:]}" if len(key) > 12 else "(missing)"
    log_info("=" * 60)
    log_info("Server READY — http://127.0.0.1:8000")
    log_info("Logs: THIS terminal + %s", LOG_FILE)
    log_info("PAGEINDEX_API_KEY: %s", masked)
    try:
        opt = ConfigLoader().load({})
        log_info("Config pipeline: %s", getattr(opt, "pipeline", "pageindex"))
    except Exception as e:
        log_error("Config load failed: %s", e)
    if not key:
        log_error("PAGEINDEX_API_KEY is MISSING — set it in .env and restart!")

    try:
        from sqlalchemy import text

        from pageindex.db.database import Base, engine
        from pageindex.db.database import ensure_schema
        import pageindex.db.models  # Required for Base.metadata to discover the tables

        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        # Create additive tables first; ensure_schema rebuilds vectors to v2.
        Base.metadata.create_all(bind=engine)
        ensure_schema()
        log_info("Database tables verified.")
    except Exception as e:
        log_error("Failed to initialize database: %s", e)

    try:
        from pageindex.embedding_service import embedding_service

        embedding_service.load_model()
    except Exception as e:
        log_error("Embedding model not loaded at startup: %s", e)

    app.state.pageindex_progress = {}
    app.state.vector_progress = {}
    # In-memory registry for vector-only uploads (not stored in document_jobs).
    app.state.vector_jobs = {}
    log_info("=" * 60)
    yield
    log_info("Server stopped")


def friendly_error(exc: Exception) -> str:
    from pageindex.api_errors import format_user_error

    return format_user_error(exc)


app = FastAPI(title="PageIndex UI", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    log_info(">>> %s %s", request.method, request.url.path)
    try:
        response = await call_next(request)
        log_info("<<< %s %s -> %s (%.1fs)", request.method, request.url.path, response.status_code, time.time() - start)
        return response
    except Exception:
        log_exception("<<< %s %s -> ERROR (%.1fs)", request.method, request.url.path, time.time() - start)
        raise


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/usage")
async def usage_dashboard():
    return FileResponse(STATIC_DIR / "usage.html")


@app.get("/api/usage/jobs")
async def list_usage_jobs(limit: int = 30):
    from pageindex.usage.service import UsageService

    return JSONResponse(content={"jobs": UsageService().list_recent_jobs(limit=limit)})


@app.get("/api/usage/job/{job_id}")
async def get_usage_job(job_id: str):
    from pageindex.usage.service import UsageService

    data = UsageService().job_summary(job_id)
    if data.get("error"):
        raise HTTPException(status_code=404, detail=data["error"])
    return JSONResponse(content=data)


@app.get("/api/usage/document/{doc_id}")
async def get_usage_document(doc_id: str):
    from pageindex.usage.service import UsageService

    data = UsageService().document_summary(doc_id)
    if data.get("error"):
        raise HTTPException(status_code=404, detail=data["error"])
    return JSONResponse(content=data)


@app.get("/api/usage/page/{doc_id}/{page_number}")
async def get_usage_page(doc_id: str, page_number: int):
    from pageindex.usage.service import UsageService

    data = UsageService().page_summary(doc_id, page_number)
    if data.get("error"):
        raise HTTPException(status_code=404, detail=data["error"])
    return JSONResponse(content=data)


@app.get("/api/usage/credits/breakdown")
async def get_credits_breakdown(
    job_id: str | None = None,
    document_id: str | None = None,
):
    from pageindex.usage.service import UsageService

    data = UsageService().credits_breakdown(job_id=job_id, document_id=document_id)
    if data.get("error"):
        raise HTTPException(status_code=404, detail=data["error"])
    return JSONResponse(content=data)


@app.get("/api/usage/timeline/{job_id}")
async def get_usage_timeline(job_id: str):
    from pageindex.usage.service import UsageService

    data = UsageService().timeline(job_id)
    if data.get("error"):
        raise HTTPException(status_code=404, detail=data["error"])
    return JSONResponse(content=data)


@app.get("/api/usage/trace/{job_id}")
async def get_usage_trace(job_id: str, credits: float | None = None):
    from pageindex.usage.service import UsageService

    return JSONResponse(content=UsageService().reverse_trace(job_id, credits=credits))


@app.get("/api/usage/alerts/{job_id}")
async def get_usage_alerts(job_id: str):
    from pageindex.usage.store import get_store

    store = get_store()
    conn = store._connect()
    try:
        rows = conn.execute(
            "SELECT * FROM usage_alerts WHERE job_id = ? ORDER BY created_at DESC",
            (job_id,),
        ).fetchall()
        return JSONResponse(content={"alerts": [dict(r) for r in rows]})
    finally:
        conn.close()


@app.get("/api/health")
async def health():
    from pageindex.cloud import get_api_key

    key = get_api_key()
    return {
        "ok": bool(key),
        "pipeline": "pageindex",
        "api_base": os.getenv("PAGEINDEX_API_BASE", "https://api.pageindex.ai"),
        "openrouter_disabled": True,
        "api_key_set": bool(key),
    }


@app.post("/api/search")
async def search_document(body: dict[str, Any] = Body(...)):
    query = (body.get("query") or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Requires 'query'.")

    doc_id = body.get("doc_id")
    if doc_id:
        from pageindex.cloud import get_client

        log_info("SEARCH PageIndex API | doc_id=%s | query=%s", doc_id, query)
        loop = asyncio.get_running_loop()

        def _do_search():
            from pageindex.usage.meter import UsageMeter

            client = get_client()
            meter = UsageMeter(document_id=doc_id, document_name=doc_id, pipeline="pageindex")
            client.usage_meter = meter
            out = client.retrieve(doc_id, query, thinking=bool(body.get("thinking")))
            out["_usage"] = meter.complete(status="success")
            return out

        try:
            result = await loop.run_in_executor(executor, _do_search)
            log_info("SEARCH done | status=%s", result.get("status"))
        except Exception as exc:
            log_exception("SEARCH failed: %s", exc)
            raise HTTPException(status_code=500, detail=friendly_error(exc)) from exc
        return JSONResponse(content={"query": query, "doc_id": doc_id, "retrieval": result})

    from pageindex.vrag.pipeline import search as vrag_search

    structure = body.get("structure")
    if not structure:
        raise HTTPException(status_code=400, detail="Requires 'doc_id' or 'structure'.")
    log_info("SEARCH local VRAG | query=%s", query)
    hits = vrag_search(structure, query, top_k=int(body.get("top_k", 5)))
    return JSONResponse(content={"query": query, "hits": hits})


@app.post("/api/process")
async def process_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    mode: str = Form("hybrid"),
    chunk_size: int | None = Form(None),
    chunk_overlap: int | None = Form(None),
    no_chunking: bool = Form(False),
    doc_type: str = Form("book"),
):
    log_info("-" * 50)
    log_info("UPLOAD received: %s", file.filename)

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    mode_norm = (mode or "hybrid").strip().lower()
    if mode_norm not in {"pageindex", "vector", "hybrid"}:
        raise HTTPException(
            status_code=400,
            detail="mode must be one of: pageindex, vector, hybrid",
        )

    doc_type_norm = (doc_type or "book").strip().lower()
    if doc_type_norm not in {"book", "notice_reply", "faq"}:
        raise HTTPException(
            status_code=400,
            detail="doc_type must be one of: book, notice_reply, faq",
        )

    from pageindex.env_settings import settings

    use_chunk_size = int(chunk_size if chunk_size is not None else settings.VECTOR_CHUNK_SIZE)
    use_chunk_overlap = int(
        chunk_overlap if chunk_overlap is not None else settings.VECTOR_CHUNK_OVERLAP
    )
    # Chunk settings apply only to book vectorization.
    if doc_type_norm == "book":
        if use_chunk_size < 100:
            raise HTTPException(status_code=400, detail="chunk_size must be >= 100")
        if use_chunk_overlap < 0:
            raise HTTPException(status_code=400, detail="chunk_overlap must be >= 0")
        if use_chunk_overlap >= use_chunk_size and not no_chunking:
            raise HTTPException(status_code=400, detail="chunk_overlap must be < chunk_size")

    safe_name = Path(file.filename).name
    file_id = uuid.uuid4().hex[:8]
    pdf_path = UPLOAD_DIR / f"{file_id}_{safe_name}"

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    pdf_path.write_bytes(content)
    log_info(
        "Saved upload: %d bytes -> %s | mode=%s doc_type=%s no_chunking=%s chunk=%d overlap=%d",
        len(content),
        pdf_path.name,
        mode_norm,
        doc_type_norm,
        no_chunking,
        use_chunk_size,
        use_chunk_overlap,
    )

    from pageindex.db.database import SessionLocal
    from pageindex.db.repository import IngestionRepository

    doc_id = uuid.uuid4().hex
    # Vector path uses its own free job_id (never document_jobs.seq_id).
    vector_job_id = int(uuid.uuid4().int % (2**31 - 1))
    pageindex_job_id: int | None = None

    db = SessionLocal()
    try:
        # PageIndex path owns document_jobs — create only when PageIndex runs.
        if mode_norm in {"pageindex", "hybrid"}:
            repo = IngestionRepository(db)
            job = repo.upsert_job(doc_id, status="processing", file_name=safe_name)
            pageindex_job_id = int(job.seq_id)
            _set_pageindex_progress(
                doc_id,
                status="pending",
                message="Waiting for PageIndex service...",
            )
        else:
            _set_pageindex_progress(
                doc_id,
                status="skipped",
                message="Skipped (Vector-only mode)",
            )

        if mode_norm in {"vector", "hybrid"}:
            if not hasattr(app.state, "vector_jobs") or app.state.vector_jobs is None:
                app.state.vector_jobs = {}
            app.state.vector_jobs[doc_id] = {
                "seq_id": vector_job_id,
                "doc_id": doc_id,
                "file_name": safe_name,
                "status": "processing",
                "error_message": None,
                "doc_type": doc_type_norm,
                "created_at": _utc_now_iso(),
                "path": "vector",
            }
            if doc_type_norm == "book":
                vec_msg = (
                    "Waiting for Vector service (no chunking)..."
                    if no_chunking
                    else (
                        f"Waiting for Vector service "
                        f"(chunk={use_chunk_size}, overlap={use_chunk_overlap})..."
                    )
                )
            else:
                vec_msg = f"Waiting for Vector service ({doc_type_norm})..."
            _set_vector_progress(
                doc_id,
                status="pending",
                message=vec_msg,
                total=0,
                done=0,
            )
        else:
            _set_vector_progress(
                doc_id,
                status="skipped",
                message="Skipped (PageIndex-only mode)",
                total=0,
                done=0,
            )
    finally:
        db.close()

    # Background job_id: PageIndex seq when present, else vector correlator.
    pipeline_job_id = pageindex_job_id if pageindex_job_id is not None else vector_job_id

    background_tasks.add_task(
        _process_upload_job,
        pdf_path,
        safe_name,
        doc_id,
        pipeline_job_id,
        mode_norm,
        use_chunk_size,
        use_chunk_overlap,
        no_chunking,
        doc_type_norm,
        vector_job_id,
    )
    log_info(
        "Upload queued | mode=%s doc_type=%s doc_id=%s pageindex_job=%s vector_job=%s file=%s",
        mode_norm,
        doc_type_norm,
        doc_id,
        pageindex_job_id,
        vector_job_id,
        safe_name,
    )
    return JSONResponse(
        content={
            "message": f"Upload accepted. Mode={mode_norm} doc_type={doc_type_norm}.",
            "doc_id": doc_id,
            "job_id": pipeline_job_id,
            "pageindex_job_id": pageindex_job_id,
            "vector_job_id": vector_job_id if mode_norm in {"vector", "hybrid"} else None,
            "file_name": safe_name,
            "mode": mode_norm,
            "doc_type": doc_type_norm,
            "pageindex_status": "pending" if mode_norm in {"pageindex", "hybrid"} else "skipped",
            "vector_status": "pending" if mode_norm in {"vector", "hybrid"} else "skipped",
            "vector_chunk_size": use_chunk_size,
            "vector_chunk_overlap": use_chunk_overlap,
            "no_chunking": no_chunking,
        },
        status_code=202,
    )


@app.post("/api/ingestion/ingest/run")
async def run_ingestion(background_tasks: BackgroundTasks):
    """S3 batch ingestion — currently DISABLED (commented). Use drag-and-drop upload instead.

    To re-enable:
      1. Uncomment the block below
      2. Set AWS_* / S3_* in .env
      3. Remove the HTTP 410 raise
    """
    # --- DISABLED: S3 batch ingestion ---
    # from pageindex.ingestion_service import ingestion_service
    # if ingestion_service._is_running:
    #     raise HTTPException(status_code=409, detail="Ingestion is already running.")
    # background_tasks.add_task(ingestion_service.run)
    # return {"message": "Ingestion started in the background."}

    raise HTTPException(
        status_code=410,
        detail="S3 batch ingestion is disabled (commented out). Use the drag-and-drop PDF upload UI only.",
    )


@app.get("/api/ingestion/jobs")
async def list_jobs():
    from sqlalchemy.orm import load_only
    from pageindex.db.database import SessionLocal
    from pageindex.db.models import DocumentJob

    pageindex_progress = getattr(app.state, "pageindex_progress", {})
    vector_progress = getattr(app.state, "vector_progress", {})
    vector_jobs = getattr(app.state, "vector_jobs", {}) or {}

    db = SessionLocal()
    try:
        # PageIndex path jobs (document_jobs).
        pi_jobs = (
            db.query(DocumentJob)
            .options(
                load_only(
                    DocumentJob.seq_id,
                    DocumentJob.doc_id,
                    DocumentJob.file_name,
                    DocumentJob.status,
                    DocumentJob.error_message,
                    DocumentJob.created_at,
                )
            )
            .order_by(DocumentJob.seq_id.asc().nulls_last(), DocumentJob.created_at.asc())
            .limit(100)
            .all()
        )
        jobs: list[dict[str, Any]] = []
        seen: set[str] = set()
        for j in pi_jobs:
            seen.add(j.doc_id)
            jobs.append(
                {
                    "seq_id": j.seq_id,
                    "file_name": j.file_name,
                    "doc_id": j.doc_id,
                    "status": j.status,
                    "error_message": j.error_message,
                    "created_at": j.created_at.isoformat() if j.created_at else None,
                    "path": "pageindex",
                    "pageindex": pageindex_progress.get(j.doc_id) or {},
                    "vector": vector_progress.get(j.doc_id) or {},
                    "embedding": vector_progress.get(j.doc_id) or {},
                }
            )

        # Vector-only jobs (in-memory registry — not in document_jobs).
        for doc_id, vj in vector_jobs.items():
            if doc_id in seen:
                # Hybrid: enrich existing card with vector path info.
                for card in jobs:
                    if card["doc_id"] == doc_id:
                        card["path"] = "hybrid"
                        card["vector_job_id"] = vj.get("seq_id")
                        card["doc_type"] = vj.get("doc_type")
                        break
                continue
            jobs.append(
                {
                    "seq_id": vj.get("seq_id"),
                    "file_name": vj.get("file_name"),
                    "doc_id": doc_id,
                    "status": vj.get("status")
                    or (vector_progress.get(doc_id) or {}).get("status")
                    or "processing",
                    "error_message": vj.get("error_message"),
                    "created_at": vj.get("created_at"),
                    "path": "vector",
                    "doc_type": vj.get("doc_type"),
                    "pageindex": pageindex_progress.get(doc_id)
                    or {"status": "skipped", "message": "Skipped (Vector-only mode)"},
                    "vector": vector_progress.get(doc_id) or {},
                    "embedding": vector_progress.get(doc_id) or {},
                }
            )

        jobs.sort(key=lambda x: x.get("created_at") or "", reverse=True)
        return {"jobs": jobs[:100]}
    finally:
        db.close()


@app.get("/api/ingestion/jobs/{doc_id}")
async def get_job(doc_id: str, include_body: bool = True):
    """
    Full pipeline-style payload: structure, structure_vrag, validation, readiness.
    Tree is rebuilt from document_nodes (same shape as POST /api/process).
    """
    from pageindex.db.database import SessionLocal
    from pageindex.db.models import DocumentJob, DocumentNode
    from pageindex.db.tree_builder import build_structure_vrag_root
    from pageindex.job_results import merge_job_with_pipeline_result

    db = SessionLocal()
    try:
        job = db.query(DocumentJob).filter(DocumentJob.doc_id == doc_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found.")

        rows = (
            db.query(DocumentNode)
            .filter(DocumentNode.doc_id == doc_id)
            .order_by(DocumentNode.seq_id.asc().nulls_last(), DocumentNode.node_id.asc())
            .all()
        )
        structure_vrag = None
        if rows:
            structure_vrag = build_structure_vrag_root(rows, include_body=include_body)

        payload = merge_job_with_pipeline_result(
            job,
            job.results if isinstance(job.results, dict) else None,
            structure_vrag=structure_vrag,
            structure_native=job.results.get("structure") if isinstance(job.results, dict) else None,
        )
        payload["pageindex"] = getattr(app.state, "pageindex_progress", {}).get(doc_id) or {}
        payload["vector"] = getattr(app.state, "vector_progress", {}).get(doc_id) or {}
        payload["embedding"] = payload["vector"]
        return payload
    finally:
        db.close()


@app.get("/api/ingestion/nodes")
async def list_all_nodes(limit: int = 500, offset: int = 0):
    """Flat node list sorted by seq_id ASC, then node_id ASC."""
    from pageindex.db.database import SessionLocal
    from pageindex.db.models import DocumentNode

    db = SessionLocal()
    try:
        rows = (
            db.query(DocumentNode)
            .order_by(DocumentNode.seq_id.asc().nulls_last(), DocumentNode.node_id.asc())
            .offset(offset)
            .limit(limit)
            .all()
        )
        return {
            "nodes": [
                {
                    "seq_id": n.seq_id,
                    "doc_id": n.doc_id,
                    "file_name": n.file_name,
                    "node_id": n.node_id,
                    "parent_id": n.parent_id,
                    "type": n.type,
                    "title": n.title,
                    "level": n.level,
                    "retrieval_ready": n.retrieval_ready,
                }
                for n in rows
            ]
        }
    finally:
        db.close()


@app.get("/api/ingestion/jobs/{doc_id}/tree")
async def get_job_tree(doc_id: str, include_body: bool = False):
    from pageindex.db.database import SessionLocal
    from pageindex.db.models import DocumentNode
    from pageindex.db.tree_builder import build_tree_from_nodes

    db = SessionLocal()
    try:
        nodes = (
            db.query(DocumentNode)
            .filter(DocumentNode.doc_id == doc_id)
            .order_by(DocumentNode.seq_id.asc().nulls_last(), DocumentNode.node_id.asc())
            .all()
        )
        if not nodes:
            raise HTTPException(status_code=404, detail="Tree not found for this document.")

        return {"structure": build_tree_from_nodes(nodes, include_body=include_body)}
    finally:
        db.close()

