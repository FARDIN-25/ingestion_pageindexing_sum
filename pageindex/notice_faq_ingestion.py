"""Notice/reply + FAQ ingestion for the Vector/embedding path only.

Books continue to use ``vector_service.process_pdf`` unchanged; this module is
additive routing for ``doc_type`` in {notice_reply, faq, book}.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from pageindex.db.models import Vector
from pageindex.embedding_service import embedding_service
from pageindex.env_settings import settings
from pageindex.log_util import log_error, log_info
from pageindex.vector_chunking import content_sha256, estimate_token_count

# Easy to extend later — first match wins (order matters).
ISSUE_KEYWORDS: dict[str, list[str]] = {
    "belated_filing_interest": [
        "belated payment",
        "belated filing",
        "section 50",
        "delayed payment",
    ],
    "itc_mismatch": [
        "input tax credit",
        "itc mismatch",
        "wrongly availed",
        "wrongly utilized",
    ],
    "valuation_dispute": ["valuation", "undervaluation"],
    "eway_bill": ["e-way bill", "eway bill"],
    "return_mismatch": ["gstr-1", "gstr-3b mismatch", "return mismatch"],
}

# TODO: tune FAQ Q/A regex once a sample FAQ document is available.
_FAQ_QA_PATTERN = re.compile(
    r"(?:Q|Question)\s*[:\-]\s*(.*?)\s*(?:A|Answer)\s*[:\-]\s*(.*?)(?=(?:Q|Question)\s*[:\-]|$)",
    re.IGNORECASE | re.DOTALL,
)

_REF_NO_RE = re.compile(
    r"Reference\s*No\.?\s*[:\-]?\s*([A-Z0-9]+)",
    re.IGNORECASE,
)
_SECTION_RE = re.compile(r"(?:^|\b)[Ss]ection\s+(\d+)")
_TAX_PERIOD_LABELED_RE = re.compile(
    r"Tax\s*Period\s*[:\-]\s*([^\n\r]+)",
    re.IGNORECASE,
)
_TAX_PERIOD_RANGE_RE = re.compile(
    r"\b([A-Z]{3}\s+\d{4}\s*[-–]\s*[A-Z]{3}\s+\d{4})\b",
)
_GSTIN_RE = re.compile(r"GSTIN/ID\s*:\s*([A-Z0-9]+)", re.IGNORECASE)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _job_seq(job_id: str | int) -> int:
    return int(job_id)


def _embed_one_sync(text: str) -> list[float]:
    """Reuse embedding singleton without requiring an async caller."""
    embedding_service.load_model()
    return embedding_service._encode_sync([text])[0]


def split_notice_reply_pages(pdf_path: str) -> dict[str, Any]:
    """PAGE-BASED keyword detector. No AI / embeddings / LLM.

    Once ``GST / NOTICE`` is seen, every subsequent page is notice until
    ``GST / REPLY`` is seen; then every subsequent page is reply.
    """
    import fitz  # PyMuPDF

    notice_pages: list[str] = []
    reply_pages: list[str] = []
    current_section: str | None = None

    doc = fitz.open(pdf_path)
    try:
        log_info("[notice_reply] PDF opened: %s (%d pages)", pdf_path, doc.page_count)
        for page in doc:
            text = page.get_text() or ""
            upper = text.upper()

            if "GST / NOTICE" in upper:
                current_section = "notice"
            elif "GST / REPLY" in upper:
                current_section = "reply"

            if current_section == "notice":
                notice_pages.append(text)
            elif current_section == "reply":
                reply_pages.append(text)
    finally:
        doc.close()

    if not notice_pages:
        raise Exception(f"Notice section not found in {pdf_path}")
    if not reply_pages:
        raise Exception(f"Reply section not found in {pdf_path}")

    notice_text = "\n".join(notice_pages)
    reply_text = "\n".join(reply_pages)
    result = {
        "notice_text": notice_text,
        "reply_text": reply_text,
        "notice_page_count": len(notice_pages),
        "reply_page_count": len(reply_pages),
    }
    log_info(
        "[notice_reply] Section split: notice_pages=%d reply_pages=%d",
        result["notice_page_count"],
        result["reply_page_count"],
    )
    return result


def extract_notice_metadata(notice_text: str) -> dict[str, Any]:
    """Regex metadata extraction — failures return None fields, never raise."""
    text = notice_text or ""

    ref_m = _REF_NO_RE.search(text)
    section_m = _SECTION_RE.search(text)
    tax_m = _TAX_PERIOD_LABELED_RE.search(text) or _TAX_PERIOD_RANGE_RE.search(text)
    gstin_m = _GSTIN_RE.search(text)

    meta = {
        "reference_no": ref_m.group(1).strip() if ref_m else None,
        "notice_section": section_m.group(1).strip() if section_m else None,
        "tax_period": tax_m.group(1).strip() if tax_m else None,
        "gstin": gstin_m.group(1).strip() if gstin_m else None,
    }
    log_info(
        "[notice_reply] Metadata extracted: ref=%s section=%s period=%s gstin=%s",
        meta["reference_no"],
        meta["notice_section"],
        meta["tax_period"],
        meta["gstin"],
    )
    return meta


def classify_issue_category(notice_text: str) -> str:
    """Keyword dictionary classifier (not LLM / not embedding)."""
    lower = (notice_text or "").lower()
    for category, keywords in ISSUE_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                return category
    return "uncategorized"


def _find_by_content_hash(db: Session, content_hash: str) -> Vector | None:
    """Reuse content_hash index for global dedup (same hash → skip insert)."""
    return (
        db.query(Vector)
        .filter(Vector.content_hash == content_hash)
        .order_by(Vector.created_at.asc())
        .first()
    )


def _insert_vector_row(
    db: Session,
    *,
    doc_id: str,
    job_seq: int,
    node_id: str,
    chunk_index: int,
    chunk_text: str,
    content_hash: str,
    embedding: list[float] | None,
    embedding_dim: int | None,
    model_name: str | None,
    status: str,
    chunk_meta: dict[str, Any],
) -> uuid.UUID:
    """Insert into `vectors` only — never touches document_jobs / document_nodes."""
    # Replace any prior row for this doc/node (re-ingest with new text).
    db.query(Vector).filter(
        Vector.doc_id == doc_id,
        Vector.node_id == node_id,
    ).delete(synchronize_session=False)

    row_id = uuid.uuid4()
    db.add(
        Vector(
            id=row_id,
            doc_id=doc_id,
            job_id=job_seq,
            node_id=node_id,
            chunk_index=chunk_index,
            chunk_text=chunk_text,
            content_hash=content_hash,
            token_count=estimate_token_count(chunk_text),
            embedding=embedding,
            embedding_dim=embedding_dim,
            model_name=model_name,
            status=status,
            chunk_meta=chunk_meta,
            updated_at=_utcnow(),
        )
    )
    return row_id


def ingest_notice_reply(pdf_path: str, doc_id: str, job_id: str | int) -> dict[str, Any]:
    """Ingest a GST notice+reply PDF: embed notice only; store reply without embedding."""
    from pageindex.db.database import SessionLocal

    job_seq = _job_seq(job_id)
    db = SessionLocal()
    try:
        split = split_notice_reply_pages(pdf_path)
        notice_text = split["notice_text"]
        reply_text = split["reply_text"]

        metadata = extract_notice_metadata(notice_text)
        issue_category = classify_issue_category(notice_text)
        pair_id = metadata.get("reference_no") or uuid.uuid4().hex

        notice_hash = content_sha256(notice_text)
        reply_hash = content_sha256(reply_text)

        notice_existing = _find_by_content_hash(db, notice_hash)
        reply_existing = _find_by_content_hash(db, reply_hash)

        notice_row_id: uuid.UUID | str
        reply_row_id: uuid.UUID | str

        if notice_existing:
            notice_row_id = notice_existing.id
            log_info(
                "[notice_reply] Duplicate notice skipped content_hash=%s existing_id=%s",
                notice_hash[:12],
                notice_row_id,
            )
        else:
            notice_node_id = "nr_notice"
            embedding = _embed_one_sync(notice_text)
            log_info(
                "[notice_reply] Embedding generated for notice doc_id=%s dim=%s model=%s",
                doc_id,
                len(embedding),
                embedding_service.model_name,
            )
            notice_row_id = _insert_vector_row(
                db,
                doc_id=doc_id,
                job_seq=job_seq,
                node_id=notice_node_id,
                chunk_index=0,
                chunk_text=notice_text,
                content_hash=notice_hash,
                embedding=embedding,
                embedding_dim=len(embedding) or settings.EMBEDDING_DIM,
                model_name=embedding_service.model_name,
                status="embedded",
                chunk_meta={
                    "doc_type": "notice_reply",
                    "role": "notice",
                    "pair_id": pair_id,
                    "notice_section": metadata.get("notice_section"),
                    "tax_period": metadata.get("tax_period"),
                    "reference_no": metadata.get("reference_no"),
                    "gstin": metadata.get("gstin"),
                    "issue_category": issue_category,
                },
            )
            log_info("[notice_reply] Notice row inserted id=%s", notice_row_id)

        if reply_existing:
            reply_row_id = reply_existing.id
            log_info(
                "[notice_reply] Duplicate reply skipped content_hash=%s existing_id=%s",
                reply_hash[:12],
                reply_row_id,
            )
        else:
            reply_node_id = "nr_reply"
            reply_row_id = _insert_vector_row(
                db,
                doc_id=doc_id,
                job_seq=job_seq,
                node_id=reply_node_id,
                chunk_index=1,
                chunk_text=reply_text,
                content_hash=reply_hash,
                embedding=None,
                embedding_dim=None,
                model_name=None,
                status="not_embedded",
                chunk_meta={
                    "doc_type": "notice_reply",
                    "role": "reply",
                    "pair_id": pair_id,
                },
            )
            log_info("[notice_reply] Reply row inserted id=%s (not embedded)", reply_row_id)

        db.commit()
        return {
            "pair_id": pair_id,
            "notice_row_id": str(notice_row_id),
            "reply_row_id": str(reply_row_id),
            "issue_category": issue_category,
            "status": "success",
        }
    except Exception as exc:
        try:
            db.rollback()
        except Exception:
            pass
        log_error(
            "[notice_reply] FAILED doc_id=%s pdf_path=%s: %s",
            doc_id,
            pdf_path,
            exc,
        )
        raise Exception(
            f"notice_reply ingestion failed for doc_id={doc_id} pdf_path={pdf_path}: {exc}"
        ) from exc
    finally:
        db.close()


def _load_faq_text(pdf_path_or_text: str) -> str:
    path = Path(str(pdf_path_or_text))
    if path.is_file() and path.suffix.lower() == ".pdf":
        import fitz

        doc = fitz.open(str(path))
        try:
            log_info("[faq] PDF opened: %s (%d pages)", path, doc.page_count)
            parts = [(page.get_text() or "") for page in doc]
            return "\n".join(parts)
        finally:
            doc.close()
    return str(pdf_path_or_text or "")


def extract_faq_qa_pairs(text: str) -> list[tuple[str, str]]:
    """Extract (question, answer) pairs.

    TODO: tune ``_FAQ_QA_PATTERN`` once a sample FAQ document is available —
    current pattern matches ``Q:``/``Question:`` … ``A:``/``Answer:`` blocks.
    """
    pairs: list[tuple[str, str]] = []
    for match in _FAQ_QA_PATTERN.finditer(text or ""):
        q = (match.group(1) or "").strip()
        a = (match.group(2) or "").strip()
        if q and a:
            pairs.append((q, a))
    return pairs


def ingest_faq(pdf_path_or_text: str, doc_id: str, job_id: str | int) -> list[dict[str, Any]]:
    """Ingest FAQ Q/A pairs: embed questions only; store answers without embedding."""
    from pageindex.db.database import SessionLocal

    job_seq = _job_seq(job_id)
    db = SessionLocal()
    summaries: list[dict[str, Any]] = []
    try:
        text = _load_faq_text(pdf_path_or_text)
        pairs = extract_faq_qa_pairs(text)
        log_info("[faq] Extracted %d Q/A pair(s) for doc_id=%s", len(pairs), doc_id)
        if not pairs:
            raise Exception(
                f"No FAQ Q/A pairs found in {pdf_path_or_text} "
                "(expected Q:/Question: … A:/Answer: blocks)"
            )

        for i, (question, answer) in enumerate(pairs):
            pair_id = uuid.uuid4().hex
            q_hash = content_sha256(question)
            a_hash = content_sha256(answer)

            q_existing = _find_by_content_hash(db, q_hash)
            a_existing = _find_by_content_hash(db, a_hash)

            if q_existing:
                q_row_id = q_existing.id
                log_info(
                    "[faq] Duplicate question skipped content_hash=%s existing_id=%s",
                    q_hash[:12],
                    q_row_id,
                )
            else:
                q_node_id = f"fq{i:04d}_q"
                embedding = _embed_one_sync(question)
                log_info(
                    "[faq] Embedding generated for question %d doc_id=%s",
                    i + 1,
                    doc_id,
                )
                q_row_id = _insert_vector_row(
                    db,
                    doc_id=doc_id,
                    job_seq=job_seq,
                    node_id=q_node_id,
                    chunk_index=0,
                    chunk_text=question,
                    content_hash=q_hash,
                    embedding=embedding,
                    embedding_dim=len(embedding) or settings.EMBEDDING_DIM,
                    model_name=embedding_service.model_name,
                    status="embedded",
                    chunk_meta={
                        "doc_type": "faq",
                        "role": "question",
                        "pair_id": pair_id,
                    },
                )
                log_info("[faq] Question row inserted id=%s", q_row_id)

            if a_existing:
                a_row_id = a_existing.id
                log_info(
                    "[faq] Duplicate answer skipped content_hash=%s existing_id=%s",
                    a_hash[:12],
                    a_row_id,
                )
            else:
                a_node_id = f"fq{i:04d}_a"
                a_row_id = _insert_vector_row(
                    db,
                    doc_id=doc_id,
                    job_seq=job_seq,
                    node_id=a_node_id,
                    chunk_index=1,
                    chunk_text=answer,
                    content_hash=a_hash,
                    embedding=None,
                    embedding_dim=None,
                    model_name=None,
                    status="not_embedded",
                    chunk_meta={
                        "doc_type": "faq",
                        "role": "answer",
                        "pair_id": pair_id,
                    },
                )
                log_info("[faq] Answer row inserted id=%s (not embedded)", a_row_id)

            summaries.append(
                {
                    "pair_id": pair_id,
                    "question_row_id": str(q_row_id),
                    "answer_row_id": str(a_row_id),
                    "status": "success",
                }
            )

        db.commit()
        log_info("[faq] DONE doc_id=%s pairs=%d", doc_id, len(summaries))
        return summaries
    except Exception as exc:
        try:
            db.rollback()
        except Exception:
            pass
        log_error(
            "[faq] FAILED doc_id=%s source=%s: %s",
            doc_id,
            pdf_path_or_text,
            exc,
        )
        raise Exception(
            f"faq ingestion failed for doc_id={doc_id} source={pdf_path_or_text}: {exc}"
        ) from exc
    finally:
        db.close()


async def ingest_document(
    pdf_path: str,
    doc_id: str,
    job_id: str | int,
    doc_type: str,
    **book_kwargs: Any,
) -> Any:
    """Route Vector ingestion by document type.

    ``book`` delegates to the existing ``vector_service.process_pdf`` (unchanged).
    ``notice_reply`` / ``faq`` use the additive paths in this module.
    """
    import asyncio

    normalized = (doc_type or "book").strip().lower()
    if normalized == "notice_reply":
        return await asyncio.to_thread(ingest_notice_reply, pdf_path, doc_id, job_id)
    if normalized == "faq":
        return await asyncio.to_thread(ingest_faq, pdf_path, doc_id, job_id)
    if normalized == "book":
        # Existing book path — do not modify vector_service.process_pdf.
        from pageindex.vector_service import vector_service

        return await vector_service.process_pdf(
            pdf_path,
            doc_id=doc_id,
            job_id=int(job_id) if job_id is not None else None,
            **book_kwargs,
        )
    raise ValueError(f"Unknown doc_type: {doc_type}")
