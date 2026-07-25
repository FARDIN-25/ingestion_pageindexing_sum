"""Independent PageIndex service: PDF → PageIndex cloud API → document_jobs/nodes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from pageindex.cloud import build_cloud_index, get_api_key
from pageindex.db.database import SessionLocal
from pageindex.db.repository import IngestionRepository
from pageindex.job_results import slim_job_results
from pageindex.log_util import log_exception, log_info
from pageindex.usage.meter import UsageMeter


ProgressCallback = Callable[[dict[str, Any]], None]


class PageIndexUploadService:
    """Separate from Vector — owns PageIndex API + document_jobs / document_nodes only."""

    def process_pdf(
        self,
        pdf_path: str,
        *,
        doc_id: str,
        file_name: str | None = None,
        results_dir: Path | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        def emit(**fields: Any) -> None:
            if progress_callback:
                progress_callback({"doc_id": doc_id, **fields})

        if not get_api_key():
            raise RuntimeError(
                "PAGEINDEX_API_KEY is not set. Add it to .env — https://dash.pageindex.ai/api-keys"
            )

        db = SessionLocal()
        repo = IngestionRepository(db)
        try:
            emit(status="processing", message="Calling PageIndex API...")
            log_info("[pageindex] START doc_id=%s file=%s", doc_id, file_name or Path(pdf_path).name)

            repo.upsert_job(doc_id, status="processing", file_name=file_name)

            meter = UsageMeter(
                job_id=doc_id,
                document_id=doc_id,
                document_name=file_name or Path(pdf_path).name,
                pipeline="pageindex",
            )
            result = build_cloud_index(pdf_path, meter=meter, job_id=doc_id)

            if results_dir is not None:
                results_dir.mkdir(parents=True, exist_ok=True)
                stem = Path(file_name or pdf_path).stem
                out = results_dir / f"{stem}_structure.json"
                with open(out, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                log_info("[pageindex] Wrote results %s", out.name)

            structure = result.get("structure_vrag") or result.get("structure")
            flat_nodes: list[dict] = []
            if structure:
                roots = structure if isinstance(structure, list) else [structure]

                def flatten(nodes: list[dict]) -> None:
                    for n in nodes:
                        if isinstance(n, dict):
                            flat_nodes.append(n)
                            flatten(n.get("nodes") or [])

                flatten(roots)
                if flat_nodes:
                    repo.replace_nodes(doc_id, flat_nodes)

            repo.upsert_job(
                doc_id,
                status="completed",
                results=slim_job_results(result, job_doc_id=doc_id),
                file_name=file_name,
            )

            emit(
                status="completed",
                message=f"PageIndex completed ({len(flat_nodes)} nodes)",
                node_count=len(flat_nodes),
                page_count=result.get("page_count"),
            )
            log_info(
                "[pageindex] DONE doc_id=%s pages=%s nodes=%d",
                doc_id,
                result.get("page_count"),
                len(flat_nodes),
            )
            return result
        except Exception as exc:
            from pageindex.api_errors import format_user_error

            err = format_user_error(exc)[:4000]
            try:
                db.rollback()
                repo.upsert_job(doc_id, status="failed", error_message=err, file_name=file_name)
            except Exception:
                pass
            emit(status="failed", message=err[:300])
            log_exception("[pageindex] FAILED doc_id=%s: %s", doc_id, exc)
            raise
        finally:
            db.close()


pageindex_upload_service = PageIndexUploadService()
