"""Threshold alerts for credit usage anomalies."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .store import UsageStore


def _threshold(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


PAGE_COST_THRESHOLD = _threshold("ALERT_PAGE_CREDIT_THRESHOLD", 1.0)
JOB_COST_THRESHOLD = _threshold("ALERT_JOB_CREDIT_THRESHOLD", 25.0)
RETRY_THRESHOLD = _threshold("ALERT_RETRY_THRESHOLD", 5)


def check_event_alerts(
    store: UsageStore,
    *,
    job_id: str,
    document_id: str,
    page_number: int,
    operation: str,
    credits_used: float,
    status: str,
) -> list[str]:
    messages: list[str] = []
    if credits_used > PAGE_COST_THRESHOLD:
        msg = (
            f"Page {page_number} {operation} used {credits_used:.4f} credits "
            f"(threshold {PAGE_COST_THRESHOLD})"
        )
        store.insert_alert(
            job_id=job_id,
            document_id=document_id,
            alert_type="page_cost_high",
            message=msg,
            threshold=PAGE_COST_THRESHOLD,
            actual_value=credits_used,
        )
        messages.append(msg)
    if status == "failed" and credits_used > 0:
        msg = f"Wasted {credits_used:.4f} credits on failed {operation} (page {page_number})"
        store.insert_alert(
            job_id=job_id,
            document_id=document_id,
            alert_type="wasted_credits",
            message=msg,
            actual_value=credits_used,
        )
        messages.append(msg)
    return messages


def check_job_alerts(store: UsageStore, job_id: str) -> list[str]:
    job = store.get_job(job_id)
    if not job:
        return []
    messages: list[str] = []
    total = float(job.get("total_credits") or 0)
    if total > JOB_COST_THRESHOLD:
        msg = f"Job {job_id} total {total:.2f} credits exceeds threshold {JOB_COST_THRESHOLD}"
        store.insert_alert(
            job_id=job_id,
            document_id=job["document_id"],
            alert_type="job_cost_spike",
            message=msg,
            threshold=JOB_COST_THRESHOLD,
            actual_value=total,
        )
        messages.append(msg)
    retries = int(job.get("retries") or 0)
    if retries >= RETRY_THRESHOLD:
        msg = f"Job {job_id} has {retries} retries (threshold {RETRY_THRESHOLD})"
        store.insert_alert(
            job_id=job_id,
            document_id=job["document_id"],
            alert_type="retry_explosion",
            message=msg,
            threshold=float(RETRY_THRESHOLD),
            actual_value=float(retries),
        )
        messages.append(msg)
    return messages
