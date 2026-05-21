import asyncio
import contextlib
import json
import logging
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# Unbuffered output in terminal
os.environ.setdefault("PYTHONUNBUFFERED", "1")

from dotenv import load_dotenv
from fastapi import Body, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pageindex.log_util import log_error, log_exception, log_info, setup_logging
from pageindex.config import ConfigLoader, setup_pageindex_env

load_dotenv()
setup_pageindex_env()

logger = setup_logging()
log_info("=" * 60)
log_info("PageIndex app loaded — pipeline: PageIndex cloud API ONLY")
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
    log_info("=" * 60)
    yield
    log_info("Server stopped")


@contextlib.contextmanager
def capture_stdout_to_log():
    real_stdout = sys.stdout

    class StdoutToLog:
        def write(self, msg):
            if msg and msg.strip():
                log_info("[stdout] %s", msg.rstrip())

        def flush(self):
            real_stdout.flush()

    sys.stdout = StdoutToLog()
    try:
        yield
    finally:
        sys.stdout = real_stdout


def load_processing_options():
    return ConfigLoader().load({"pipeline": "pageindex"})


def process_pdf(pdf_path: str, opt, job_id: str | None = None) -> dict:
    """Always use PageIndex cloud API — never OpenRouter / legacy LLM."""
    from pageindex.cloud import build_cloud_index
    from pageindex.cloud import get_api_key
    from pageindex.usage.meter import UsageMeter

    if not get_api_key():
        raise RuntimeError(
            "PAGEINDEX_API_KEY is not set. Add it to .env — https://dash.pageindex.ai/api-keys"
        )

    jid = job_id or uuid.uuid4().hex
    meter = UsageMeter(
        job_id=jid,
        document_id=jid,
        document_name=Path(pdf_path).name,
        pipeline="pageindex",
    )

    log_info("=" * 40)
    log_info("PDF PROCESS START job_id=%s", jid)
    log_info("File: %s", pdf_path)
    log_info("Backend: PageIndex API (%s)", os.getenv("PAGEINDEX_API_BASE", "https://api.pageindex.ai"))
    log_info("=" * 40)

    try:
        with capture_stdout_to_log():
            result = build_cloud_index(pdf_path, meter=meter, job_id=jid)
    except Exception as exc:
        meter.fail(str(exc)[:500])
        raise

    log_info("PDF PROCESS DONE — doc_id=%s credits=%s", result.get("doc_id"), result.get("usage", {}).get("overview", {}).get("total_credits_used"))
    log_info("=" * 40)
    return result


def friendly_error(exc: Exception) -> str:
    msg = str(exc)
    low = msg.lower()
    if "openrouter" in low:
        return (
            "OpenRouter is disabled in this app. Restart with .\\start.ps1 "
            "and ensure PAGEINDEX_API_KEY is set in .env (not OPENROUTER_API_KEY)."
        )
    if "pageindex_api_key" in low or ("api key" in low and "pageindex" in low):
        return "Set PAGEINDEX_API_KEY in .env — https://dash.pageindex.ai/api-keys"
    if "401" in low or "403" in low or "authentication" in low or "invalid" in low and "key" in low:
        return "Invalid PAGEINDEX_API_KEY — check https://dash.pageindex.ai/api-keys"
    if "402" in low or "credit" in low or "quota" in low or "payment" in low:
        return "PageIndex API credits exhausted — add credits at https://dash.pageindex.ai"
    if "timed out" in low:
        return "PageIndex API timed out waiting for document processing. Try again."
    return msg


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
async def process_upload(file: UploadFile = File(...)):
    log_info("-" * 50)
    log_info("UPLOAD received: %s", file.filename)

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    safe_name = Path(file.filename).name
    file_id = uuid.uuid4().hex[:8]
    pdf_path = UPLOAD_DIR / f"{file_id}_{safe_name}"

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    pdf_path.write_bytes(content)
    log_info("Saved upload: %d bytes -> %s", len(content), pdf_path.name)

    try:
        opt = load_processing_options()
        log_info("Starting PageIndex cloud indexing (NOT OpenRouter)...")

        job_id = uuid.uuid4().hex
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(executor, process_pdf, str(pdf_path), opt, job_id)

        output_path = RESULTS_DIR / f"{Path(safe_name).stem}_structure.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        log_info("Wrote results: %s", output_path)
        log_info("doc_id=%s | pages=%s | job_id=%s", result.get("doc_id"), result.get("page_count"), result.get("job_id"))
        log_info("-" * 50)
        if result.get("usage"):
            result["usage_dashboard_url"] = f"/usage?job_id={result.get('job_id', job_id)}"
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as exc:
        log_exception("UPLOAD FAILED: %s", exc)
        raise HTTPException(status_code=500, detail=friendly_error(exc)) from exc
