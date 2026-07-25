"""Independent Vector service: PDF text → chunk+overlap → embeddings → vectors table ONLY.

Does not touch document_jobs / document_nodes (those belong to PageIndex).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from pageindex.db.models import Vector
from pageindex.embedding_service import embedding_service
from pageindex.env_settings import settings
from pageindex.log_util import log_exception, log_info
from pageindex.vector_chunking import chunk_text, content_sha256, estimate_token_count


ProgressCallback = Callable[[dict[str, Any]], None]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def extract_pdf_text(pdf_path: str) -> str:
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    try:
        parts: list[str] = []
        for i, page in enumerate(doc):
            text = (page.get_text("text") or "").strip()
            if text:
                parts.append(text)
            if (i + 1) % 25 == 0:
                log_info(
                    "Vector extract: read %d/%d pages from %s",
                    i + 1,
                    doc.page_count,
                    Path(pdf_path).name,
                )
        return "\n\n".join(parts)
    finally:
        doc.close()


class VectorService:
    """Separate from PageIndex — owns chunking/overlap + embedding persistence in `vectors` only."""

    def _replace_vector_rows_prep(
        self,
        db: Session,
        *,
        doc_id: str,
        chunks: list[str],
    ) -> list[tuple[str, int, str]]:
        """Delete prior vector rows for this doc; return (node_id, chunk_index, text)."""
        db.query(Vector).filter(Vector.doc_id == doc_id).delete(synchronize_session=False)
        db.commit()

        rows: list[tuple[str, int, str]] = []
        for i, text in enumerate(chunks):
            rows.append((f"v{i:04d}", i, text))
        return rows

    def _upsert_vector_rows(self, db: Session, items: list[dict[str, Any]]) -> int:
        if not items:
            return 0
        stmt = insert(Vector).values(items)
        # Use attribute names (chunk_meta), never .metadata — that is SQLAlchemy MetaData.
        stmt = stmt.on_conflict_do_update(
            constraint="uq_vectors_doc_node_chunk_model",
            set_={
                "job_id": stmt.excluded.job_id,
                "chunk_text": stmt.excluded.chunk_text,
                "content_hash": stmt.excluded.content_hash,
                "token_count": stmt.excluded.token_count,
                "embedding": stmt.excluded.embedding,
                "embedding_dim": stmt.excluded.embedding_dim,
                "status": stmt.excluded.status,
                "chunk_meta": stmt.excluded.chunk_meta,
                "updated_at": stmt.excluded.updated_at,
            },
        )
        db.execute(stmt)
        db.commit()
        return len(items)

    async def process_pdf(
        self,
        pdf_path: str,
        *,
        doc_id: str,
        file_name: str | None = None,
        job_id: int | None = None,
        progress_callback: ProgressCallback | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        no_chunking: bool = False,
    ) -> dict[str, Any]:
        from pageindex.db.database import SessionLocal

        use_chunk_size = int(chunk_size if chunk_size is not None else settings.VECTOR_CHUNK_SIZE)
        use_chunk_overlap = int(
            chunk_overlap if chunk_overlap is not None else settings.VECTOR_CHUNK_OVERLAP
        )
        # Free correlator — not document_jobs.seq_id.
        job_seq = int(job_id) if job_id is not None else int(uuid.uuid4().int % (2**31 - 1))
        safe_name = file_name or Path(pdf_path).name

        db = SessionLocal()
        try:

            def emit(**fields: Any) -> None:
                payload = {"doc_id": doc_id, "job_id": job_seq, **fields}
                if progress_callback:
                    progress_callback(payload)

            emit(status="processing", message="Extracting PDF text...", total=0, done=0)
            log_info(
                "[vector] START doc_id=%s job_id=%s file=%s no_chunking=%s chunk_size=%d overlap=%d",
                doc_id,
                job_seq,
                safe_name,
                no_chunking,
                use_chunk_size,
                use_chunk_overlap,
            )

            text = extract_pdf_text(pdf_path)
            if not text.strip():
                emit(status="failed", message="No extractable text for vectorization", total=0, done=0)
                raise RuntimeError("No extractable text for vectorization")

            chunks = chunk_text(
                text,
                chunk_size=use_chunk_size,
                chunk_overlap=use_chunk_overlap,
                no_chunking=no_chunking,
            )
            log_info(
                "[vector] Prepared doc_id=%s into %d chunk(s) (no_chunking=%s size=%d overlap=%d chars=%d)",
                doc_id,
                len(chunks),
                no_chunking,
                use_chunk_size,
                use_chunk_overlap,
                len(text),
            )
            emit(
                status="processing",
                message=(
                    "No chunking — embedding full document as 1 vector"
                    if no_chunking
                    else f"Created {len(chunks)} overlapping chunks"
                ),
                total=len(chunks),
                done=0,
                chunk_count=len(chunks),
            )

            node_rows = self._replace_vector_rows_prep(db, doc_id=doc_id, chunks=chunks)

            def build_items(embeddings: list[list[float]] | None, status: str) -> list[dict[str, Any]]:
                out: list[dict[str, Any]] = []
                for idx, (node_id, chunk_index, chunk) in enumerate(node_rows):
                    emb = embeddings[idx] if embeddings is not None else None
                    out.append(
                        {
                            "id": uuid.uuid4(),
                            "doc_id": doc_id,
                            "job_id": job_seq,
                            "node_id": node_id,
                            "chunk_index": chunk_index,
                            "chunk_text": chunk,
                            "content_hash": content_sha256(chunk),
                            "token_count": estimate_token_count(chunk),
                            "embedding": emb,
                            "embedding_dim": settings.EMBEDDING_DIM,
                            "model_name": embedding_service.model_name,
                            "status": status,
                            "chunk_meta": {
                                "doc_type": "book",
                                "chunk_size": use_chunk_size,
                                "chunk_overlap": use_chunk_overlap,
                                "no_chunking": no_chunking,
                                "source": "vector_service",
                                "file_name": safe_name,
                            },
                            "updated_at": _utcnow(),
                        }
                    )
                return out

            self._upsert_vector_rows(db, build_items(None, "pending"))

            def on_embed_progress(event: dict[str, int]) -> None:
                done = int(event.get("processed", 0))
                total = int(event.get("total", len(chunks)))
                emit(
                    status="processing",
                    message=(
                        f"Embedding batch {event.get('batch_index')}/{event.get('batch_count')} "
                        f"({done}/{total})"
                    ),
                    total=total,
                    done=done,
                    chunk_count=total,
                    model_name=embedding_service.model_name,
                )
                log_info("[vector] Embedding progress doc_id=%s %d/%d", doc_id, done, total)

            embeddings = await embedding_service.embed_texts(
                chunks,
                progress_callback=on_embed_progress,
            )
            n = self._upsert_vector_rows(db, build_items(embeddings, "completed"))

            emit(
                status="completed",
                message=f"Vectorized {n} chunks",
                total=n,
                done=n,
                chunk_count=n,
                model_name=embedding_service.model_name,
            )
            log_info("[vector] DONE doc_id=%s vectors=%d model=%s", doc_id, n, embedding_service.model_name)
            return {
                "doc_id": doc_id,
                "job_id": job_seq,
                "chunk_count": n,
                "status": "completed",
                "model_name": embedding_service.model_name,
                "no_chunking": no_chunking,
            }
        except Exception as exc:
            try:
                db.rollback()
            except Exception:
                pass
            log_exception("[vector] FAILED doc_id=%s: %s", doc_id, exc)
            if progress_callback:
                progress_callback(
                    {
                        "doc_id": doc_id,
                        "status": "failed",
                        "message": str(exc)[:400],
                        "total": 0,
                        "done": 0,
                    }
                )
            raise
        finally:
            db.close()


vector_service = VectorService()
