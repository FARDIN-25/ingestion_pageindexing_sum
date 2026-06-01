"""Persist and merge job result payloads for document_jobs.results (JSONB)."""
from __future__ import annotations

import json
from typing import Any


def slim_job_results(
    result: dict[str, Any] | None,
    *,
    job_doc_id: str | None = None,
) -> dict[str, Any] | None:
    """Metadata for list views — full tree is served via GET /api/ingestion/jobs/{doc_id}."""
    if not result:
        return None
    validation = result.get("validation") if isinstance(result.get("validation"), dict) else {}
    readiness = result.get("readiness") if isinstance(result.get("readiness"), dict) else {}
    pageindex_doc_id = result.get("doc_id") or result.get("document_id")
    doc_id = job_doc_id or pageindex_doc_id
    out: dict[str, Any] = {
        "pipeline": result.get("pipeline"),
        "schema_version": result.get("schema_version"),
        "doc_id": doc_id,
        "document_id": doc_id,
        "pageindex_doc_id": pageindex_doc_id if job_doc_id and pageindex_doc_id != job_doc_id else None,
        "job_id": result.get("job_id"),
        "page_count": result.get("page_count"),
        "retrieval_ready": result.get("retrieval_ready"),
        "status": result.get("status"),
        "validation": {
            "valid": validation.get("valid"),
            "error_count": validation.get("error_count"),
            "errors": (validation.get("errors") or [])[:15],
            "chunk_count": validation.get("chunk_count"),
        },
        "readiness": {
            "retrieval_ready": readiness.get("retrieval_ready"),
            "ready_node_count": readiness.get("ready_node_count"),
            "candidate_node_count": readiness.get("candidate_node_count"),
            "validation_error_count": readiness.get("validation_error_count"),
        },
        "structure_url": f"/api/ingestion/jobs/{doc_id}" if doc_id else None,
    }

    structure = result.get("structure_vrag") or result.get("structure")
    if structure is not None:
        try:
            encoded = json.dumps(structure, ensure_ascii=False)
            if len(encoded) <= 1_500_000:
                out["structure_vrag"] = structure
                if result.get("structure") is not None:
                    out["structure"] = result.get("structure")
        except (TypeError, ValueError):
            pass
    return out


def merge_job_with_pipeline_result(
    job_row: Any,
    pipeline: dict[str, Any] | None,
    *,
    structure_vrag: dict[str, Any] | list[dict[str, Any]] | None = None,
    structure_native: Any = None,
) -> dict[str, Any]:
    """Full ingestion response shape (structure + validation + readiness)."""
    stored = job_row.results if isinstance(job_row.results, dict) else {}
    base: dict[str, Any] = {
        "seq_id": job_row.seq_id,
        "file_name": job_row.file_name,
        "doc_id": job_row.doc_id,
        "status": job_row.status,
        "error_message": job_row.error_message,
        "created_at": job_row.created_at.isoformat() if job_row.created_at else None,
    }
    if pipeline:
        base.update(pipeline)
    elif stored:
        base.update({k: v for k, v in stored.items() if k not in base})

    if structure_vrag is not None:
        base["structure_vrag"] = structure_vrag
        if isinstance(structure_vrag, dict) and structure_vrag.get("type") == "ROOT":
            base["structure"] = structure_native if structure_native is not None else structure_vrag
        else:
            base["structure"] = structure_native if structure_native is not None else structure_vrag
    elif stored.get("structure_vrag"):
        base["structure_vrag"] = stored["structure_vrag"]
        if stored.get("structure") is not None:
            base["structure"] = stored["structure"]
    return base
