"""Credit usage meter — instrument pipeline stages with per-page attribution."""
from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Generator

from .alerts import check_event_alerts, check_job_alerts
from .constants import Accuracy, EventStatus, Operation
from .models import UsageEvent
from .pricing import (
    credits_for_poll,
    credits_for_upload,
    credits_from_tokens,
    estimate_tokens_from_text,
    local_zero_credits,
    parse_provider_usage,
)
from .store import UsageStore, get_store


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class UsageMeter:
    """Tracks credit events for a single processing job."""

    def __init__(
        self,
        *,
        job_id: str | None = None,
        document_id: str | None = None,
        document_name: str = "",
        pipeline: str = "pageindex",
        provider: str = "pageindex",
        model: str = "pageindex",
        store: UsageStore | None = None,
    ):
        self.job_id = job_id or uuid.uuid4().hex
        self.document_id = document_id or self.job_id
        self.document_name = document_name
        self.pipeline = pipeline
        self.provider = provider
        self.model = model
        self.store = store or get_store()
        self._job_created = False
        self._page_line_counts: dict[int, int] = {}

    def ensure_job(self, page_count: int = 0) -> None:
        if self._job_created:
            if page_count:
                job = self.store.get_job(self.job_id)
                if job and not job.get("page_count"):
                    self.store.finish_job(
                        self.job_id, status="running", page_count=page_count
                    )
            return
        self.store.create_job(
            job_id=self.job_id,
            document_id=self.document_id,
            document_name=self.document_name,
            pipeline=self.pipeline,
            page_count=page_count,
        )
        self._job_created = True

    def set_page_distribution(self, page_to_lines: dict[int, int]) -> None:
        self._page_line_counts = dict(page_to_lines)

    def _pages_for_span(self, page_start: int, page_end: int) -> list[int]:
        if page_start <= 0 and page_end <= 0:
            return [0]
        if page_end < page_start:
            page_end = page_start
        return list(range(page_start, page_end + 1)) or [page_start or 0]

    def record(
        self,
        operation: str | Operation,
        *,
        page_number: int = 0,
        page_start: int = 0,
        page_end: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        credits_used: float | None = None,
        accuracy: str | None = None,
        latency_ms: int = 0,
        cache_hit: bool = False,
        status: str = EventStatus.SUCCESS.value,
        error_message: str = "",
        metadata: dict[str, Any] | None = None,
        provider: str | None = None,
        model: str | None = None,
        api_response: Any = None,
    ) -> list[str]:
        """Record one event; splits across pages when page_start/page_end set."""
        self.ensure_job()
        op = operation.value if isinstance(operation, Operation) else operation

        acc = accuracy or Accuracy.ESTIMATED.value
        cred = credits_used
        inp, out = input_tokens, output_tokens

        if api_response is not None:
            parsed = parse_provider_usage(api_response)
            if parsed:
                cred, inp, out, acc = parsed[0], parsed[1], parsed[2], parsed[3]

        if cred is None:
            pages = len(self._pages_for_span(page_start, page_end)) if page_end else 1
            cred, acc = credits_from_tokens(
                inp, out, pages=pages, operation=op
            )

        event_ids: list[str] = []
        pages = self._pages_for_span(page_start, page_end) if (page_end or page_start) else [page_number]
        n_pages = max(len(pages), 1)
        credit_each = (cred or 0.0) / n_pages
        inp_each = inp // n_pages
        out_each = out // n_pages
        lat_each = latency_ms // n_pages

        for pg in pages:
            ev = UsageEvent(
                job_id=self.job_id,
                document_id=self.document_id,
                page_number=pg,
                operation=op,
                provider=provider or self.provider,
                model=model or self.model,
                input_tokens=inp_each,
                output_tokens=out_each,
                credits_used=round(credit_each, 6),
                latency_ms=lat_each,
                cache_hit=cache_hit,
                timestamp=_utc_now(),
                accuracy=acc,
                status=status,
                error_message=error_message,
                metadata=metadata or {},
            )
            eid = self.store.insert_event(ev)
            event_ids.append(eid)
            check_event_alerts(
                self.store,
                job_id=self.job_id,
                document_id=self.document_id,
                page_number=pg,
                operation=op,
                credits_used=credit_each,
                status=status,
            )
        return event_ids

    def record_local_stage(
        self,
        operation: str | Operation,
        *,
        page_start: int = 0,
        page_end: int = 0,
        text_sample: str = "",
        latency_ms: int = 0,
        cache_hit: bool = False,
        metadata: dict | None = None,
    ) -> None:
        """Local VRAG stage — labeled LOCAL (0 API credits) with token estimates for observability."""
        inp = estimate_tokens_from_text(text_sample)
        cred, acc = local_zero_credits()
        self.record(
            operation,
            page_start=page_start,
            page_end=page_end,
            input_tokens=inp,
            output_tokens=0,
            credits_used=cred,
            accuracy=acc,
            latency_ms=latency_ms,
            cache_hit=cache_hit,
            metadata=metadata,
            provider="local",
            model="vrag",
        )

    def record_upload(self, file_bytes: int, latency_ms: int) -> None:
        cred, acc = credits_for_upload(file_bytes)
        self.record(
            Operation.PDF_UPLOAD,
            credits_used=cred,
            accuracy=acc,
            latency_ms=latency_ms,
            input_tokens=0,
            metadata={"file_bytes": file_bytes},
        )

    def record_poll(self, attempt: int, latency_ms: int, *, api_response: Any = None) -> None:
        cred, acc = credits_for_poll(attempt)
        if api_response:
            parsed = parse_provider_usage(api_response)
            if parsed:
                cred, _, _, acc = parsed[0], parsed[1], parsed[2], parsed[3]
        self.record(
            Operation.CLOUD_POLL,
            credits_used=cred,
            accuracy=acc,
            latency_ms=latency_ms,
            metadata={"poll_attempt": attempt},
            api_response=api_response,
        )

    def record_retry(self, operation: str, page_number: int = 0, credits_wasted: float = 0) -> None:
        self.store.insert_event(
            UsageEvent(
                job_id=self.job_id,
                document_id=self.document_id,
                page_number=page_number,
                operation=Operation.RETRY.value,
                provider=self.provider,
                model=self.model,
                input_tokens=0,
                output_tokens=0,
                credits_used=credits_wasted,
                latency_ms=0,
                cache_hit=False,
                timestamp=_utc_now(),
                accuracy=Accuracy.ESTIMATED.value,
                status=EventStatus.RETRY.value,
                metadata={"retried_operation": operation},
            )
        )

    @contextmanager
    def track(
        self,
        operation: str | Operation,
        *,
        page_number: int = 0,
        page_start: int = 0,
        page_end: int = 0,
        text_sample: str = "",
        local: bool = False,
        metadata: dict | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        ctx: dict[str, Any] = {"api_response": None, "cache_hit": False}
        t0 = time.perf_counter()
        try:
            yield ctx
            ms = int((time.perf_counter() - t0) * 1000)
            if local:
                self.record_local_stage(
                    operation,
                    page_start=page_start,
                    page_end=page_end,
                    text_sample=text_sample,
                    latency_ms=ms,
                    cache_hit=ctx.get("cache_hit", False),
                    metadata=metadata,
                )
            else:
                self.record(
                    operation,
                    page_number=page_number,
                    page_start=page_start,
                    page_end=page_end,
                    input_tokens=estimate_tokens_from_text(text_sample),
                    latency_ms=ms,
                    cache_hit=ctx.get("cache_hit", False),
                    metadata=metadata,
                    api_response=ctx.get("api_response"),
                )
        except Exception as exc:
            ms = int((time.perf_counter() - t0) * 1000)
            self.record(
                operation,
                page_number=page_number,
                page_start=page_start,
                page_end=page_end,
                latency_ms=ms,
                status=EventStatus.FAILED.value,
                error_message=str(exc)[:500],
                metadata=metadata,
            )
            raise

    def complete(self, *, status: str = "success", page_count: int = 0, error: str = "") -> dict[str, Any]:
        self.ensure_job(page_count)
        self.store.finish_job(
            self.job_id,
            status=status,
            page_count=page_count or None,
            error_message=error,
        )
        check_job_alerts(self.store, self.job_id)
        return build_job_report(self.store, self.job_id)

    def fail(self, error: str, *, failed_pages: int = 0) -> dict[str, Any]:
        self.store.finish_job(
            self.job_id,
            status="failed",
            error_message=error,
            failed_pages=failed_pages,
        )
        self.record(
            Operation.FAILED_JOB,
            credits_used=0,
            accuracy=Accuracy.ESTIMATED.value,
            status=EventStatus.FAILED.value,
            error_message=error,
        )
        check_job_alerts(self.store, self.job_id)
        return build_job_report(self.store, self.job_id)


def build_job_report(store: UsageStore, job_id: str) -> dict[str, Any]:
    from .service import UsageService

    return UsageService(store).job_summary(job_id)
