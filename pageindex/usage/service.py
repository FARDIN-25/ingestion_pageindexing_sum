"""Query layer for usage APIs and dashboard payloads."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from .store import UsageStore


class UsageService:
    def __init__(self, store: UsageStore | None = None):
        from .store import get_store

        self.store = store or get_store()

    def job_summary(self, job_id: str) -> dict[str, Any]:
        job = self.store.get_job(job_id)
        if not job:
            return {"error": "job_not_found", "job_id": job_id}
        events = self.store.get_events(job_id=job_id)
        return self._assemble_report(job, events)

    def document_summary(self, document_id: str) -> dict[str, Any]:
        job = self.store.get_latest_job_for_document(document_id)
        if not job:
            return {"error": "document_not_found", "document_id": document_id}
        events = self.store.get_events(document_id=document_id, job_id=job["job_id"])
        return self._assemble_report(job, events)

    def page_summary(self, document_id: str, page_number: int) -> dict[str, Any]:
        job = self.store.get_latest_job_for_document(document_id)
        if not job:
            return {"error": "document_not_found"}
        events = self.store.get_events(
            document_id=document_id,
            job_id=job["job_id"],
            page_number=page_number,
        )
        total_credits = sum(e["credits_used"] for e in events)
        by_op: dict[str, float] = defaultdict(float)
        for e in events:
            by_op[e["operation"]] += e["credits_used"]
        return {
            "document_id": document_id,
            "job_id": job["job_id"],
            "page_number": page_number,
            "total_credits": round(total_credits, 6),
            "operations": dict(by_op),
            "events": events,
            "accuracy_note": _accuracy_note(events),
        }

    def credits_breakdown(
        self,
        *,
        job_id: str | None = None,
        document_id: str | None = None,
    ) -> dict[str, Any]:
        if job_id:
            job = self.store.get_job(job_id)
            events = self.store.get_events(job_id=job_id)
        elif document_id:
            job = self.store.get_latest_job_for_document(document_id)
            events = self.store.get_events(document_id=document_id) if job else []
        else:
            jobs = self.store.list_jobs(limit=1)
            job = jobs[0] if jobs else None
            events = self.store.get_events(job_id=job["job_id"]) if job else []

        if not job:
            return {"error": "not_found"}

        by_operation: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"credits": 0.0, "tokens": 0, "count": 0, "accuracy": set()}
        )
        for e in events:
            op = e["operation"]
            by_operation[op]["credits"] += e["credits_used"]
            by_operation[op]["tokens"] += e["input_tokens"] + e["output_tokens"]
            by_operation[op]["count"] += 1
            by_operation[op]["accuracy"].add(e["accuracy"])

        breakdown = []
        for op, data in sorted(by_operation.items(), key=lambda x: -x[1]["credits"]):
            acc = data["accuracy"]
            if acc == {"EXACT"}:
                label = "EXACT"
            elif acc == {"LOCAL"}:
                label = "LOCAL"
            elif "EXACT" in acc:
                label = "MIXED"
            else:
                label = "ESTIMATED"
            breakdown.append({
                "operation": op,
                "credits": round(data["credits"], 6),
                "tokens": data["tokens"],
                "event_count": data["count"],
                "accuracy": label,
            })

        return {
            "job_id": job["job_id"],
            "document_id": job["document_id"],
            "total_credits": round(float(job["total_credits"] or 0), 6),
            "breakdown": breakdown,
        }

    def timeline(self, job_id: str) -> dict[str, Any]:
        job = self.store.get_job(job_id)
        if not job:
            return {"error": "job_not_found"}
        events = self.store.get_events(job_id=job_id)
        points = []
        cumulative = 0.0
        for e in events:
            cumulative += e["credits_used"]
            points.append({
                "timestamp": e["timestamp"],
                "operation": e["operation"],
                "page_number": e["page_number"],
                "credits_delta": e["credits_used"],
                "credits_cumulative": round(cumulative, 6),
                "accuracy": e["accuracy"],
                "status": e["status"],
            })
        return {
            "job_id": job_id,
            "document_name": job["document_name"],
            "timeline": points,
            "stages": _stage_order(events),
        }

    def reverse_trace(self, job_id: str, credits: float | None = None) -> dict[str, Any]:
        """Given credits consumed, show which pages/operations attributed."""
        events = self.store.get_events(job_id=job_id)
        if credits is not None:
            target = credits
            ranked = sorted(events, key=lambda e: -e["credits_used"])
            picked = []
            total = 0.0
            for e in ranked:
                if total >= target - 1e-9:
                    break
                picked.append(e)
                total += e["credits_used"]
            events = picked
        attribution = [
            {
                "page_number": e["page_number"],
                "operation": e["operation"],
                "credits_used": e["credits_used"],
                "accuracy": e["accuracy"],
                "input_tokens": e["input_tokens"],
                "output_tokens": e["output_tokens"],
                "latency_ms": e["latency_ms"],
                "model": e["model"],
                "status": e["status"],
                "why": e.get("metadata", {}).get("note") or _why_credits(e),
            }
            for e in events
        ]
        return {
            "job_id": job_id,
            "credits_queried": credits,
            "attribution": attribution,
            "total_attributed": round(sum(a["credits_used"] for a in attribution), 6),
        }

    def list_recent_jobs(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.store.list_jobs(limit=limit)

    def _assemble_report(self, job: dict, events: list[dict]) -> dict[str, Any]:
        page_stats: dict[int, dict[str, Any]] = defaultdict(
            lambda: {
                "credits": 0.0,
                "tokens": 0,
                "operations": defaultdict(float),
                "events": [],
            }
        )
        for e in events:
            pg = int(e["page_number"])
            page_stats[pg]["credits"] += e["credits_used"]
            page_stats[pg]["tokens"] += e["input_tokens"] + e["output_tokens"]
            page_stats[pg]["operations"][e["operation"]] += e["credits_used"]
            page_stats[pg]["events"].append(e)

        page_rows = []
        for pg in sorted(page_stats.keys()):
            if pg <= 0 and len(page_stats) > 1:
                continue
            ps = page_stats[pg]
            page_rows.append({
                "page_number": pg,
                "total_credits": round(ps["credits"], 6),
                "total_tokens": ps["tokens"],
                "operations": {k: round(v, 6) for k, v in ps["operations"].items()},
                "accuracy": _accuracy_note(ps["events"]),
            })

        if not any(p["page_number"] > 0 for p in page_rows):
            page_rows = _synthetic_page_rows(job, events)

        credits_list = [p["total_credits"] for p in page_rows if p["page_number"] > 0]
        page_count = int(job.get("page_count") or 0) or len([p for p in page_rows if p["page_number"] > 0])
        total_credits = float(job.get("total_credits") or sum(e["credits_used"] for e in events))
        total_tokens = int(job.get("total_tokens") or sum(
            e["input_tokens"] + e["output_tokens"] for e in events
        ))
        started = job.get("started_at", "")
        finished = job.get("finished_at", "") or started
        duration_s = _duration_seconds(started, finished)

        cache_hits = int(job.get("cache_hits") or 0)
        cache_miss = int(job.get("cache_misses") or 0)
        without_cache = sum(e["credits_used"] for e in events if not e["cache_hit"])
        with_cache = total_credits
        saved = float(job.get("credits_saved_cache") or max(0, without_cache - with_cache))

        return {
            "job_id": job["job_id"],
            "document_id": job["document_id"],
            "document_name": job["document_name"],
            "pipeline": job["pipeline"],
            "status": job["status"],
            "accuracy_mix": job.get("accuracy_mix") or _accuracy_note(events),
            "overview": {
                "document_name": job["document_name"],
                "total_pages": page_count,
                "total_credits_used": round(total_credits, 6),
                "average_credits_per_page": round(total_credits / max(page_count, 1), 6),
                "max_credits_page": max(credits_list) if credits_list else 0,
                "min_credits_page": min(credits_list) if credits_list else 0,
                "total_tokens": total_tokens,
                "total_processing_time": _format_duration(duration_s),
                "total_processing_ms": int(duration_s * 1000),
                "status": job["status"],
                "retries": int(job.get("retries") or 0),
                "failed_pages": int(job.get("failed_pages") or 0),
            },
            "cache_accounting": {
                "without_cache_credits": round(without_cache, 6),
                "with_cache_credits": round(with_cache, 6),
                "credits_saved": round(saved, 6),
                "cache_hits": cache_hits,
                "cache_misses": cache_miss,
            },
            "page_breakdown": page_rows,
            "event_table": [
                {
                    "event_id": e.get("event_id", ""),
                    "page": e["page_number"],
                    "operation": e["operation"],
                    "model": e["model"] or "",
                    "input_tokens": e["input_tokens"],
                    "output_tokens": e["output_tokens"],
                    "credits": e["credits_used"],
                    "latency_ms": e["latency_ms"],
                    "status": e["status"],
                    "cache_hit": e["cache_hit"],
                    "accuracy": e["accuracy"],
                    "timestamp": e["timestamp"],
                }
                for e in events
            ],
            "cost_breakdown": self.credits_breakdown(job_id=job["job_id"]).get("breakdown", []),
            "timeline": self.timeline(job["job_id"]).get("timeline", []),
            "page_heatmap": [
                {"page": p["page_number"], "credits": p["total_credits"]}
                for p in page_rows
                if p["page_number"] > 0
            ],
        }


def _accuracy_note(events: list[dict]) -> str:
    acc = {e.get("accuracy") for e in events}
    if not acc:
        return "ESTIMATED"
    if acc == {"EXACT"}:
        return "EXACT"
    if acc == {"LOCAL"}:
        return "LOCAL (no API credits)"
    if "EXACT" in acc:
        return "MIXED (EXACT + ESTIMATED)"
    return "ESTIMATED"


def _why_credits(e: dict) -> str:
    parts = [e["operation"]]
    if e["input_tokens"]:
        parts.append(f"{e['input_tokens']} input tokens")
    if e["page_number"]:
        parts.append(f"page {e['page_number']}")
    parts.append(e["accuracy"])
    return "; ".join(parts)


def _stage_order(events: list[dict]) -> list[str]:
    seen: list[str] = []
    for e in events:
        op = e["operation"]
        if op not in seen:
            seen.append(op)
    return seen


def _duration_seconds(start: str, end: str) -> float:
    try:
        from datetime import datetime

        a = datetime.fromisoformat(start.replace("Z", "+00:00"))
        b = datetime.fromisoformat(end.replace("Z", "+00:00"))
        return max(0.0, (b - a).total_seconds())
    except Exception:
        return 0.0


def _synthetic_page_rows(job: dict, events: list[dict]) -> list[dict]:
    """Cloud jobs: allocate job total across pages for UI (ESTIMATED_ALLOCATION)."""
    page_count = int(job.get("page_count") or 0)
    total = float(job.get("total_credits") or sum(e["credits_used"] for e in events))
    if page_count <= 0 or total <= 0:
        return []
    per = total / page_count
    splits = [
        ("ocr_extraction", 0.12),
        ("structure_detection", 0.03),
        ("compression_generation", 0.21),
        ("micro_summary_generation", 0.04),
        ("keyword_generation", 0.02),
    ]
    norm = sum(x[1] for x in splits)
    rows = []
    for p in range(1, page_count + 1):
        ops = {n: round(per * (f / norm), 6) for n, f in splits}
        rows.append({
            "page_number": p,
            "total_credits": round(per, 6),
            "total_tokens": 0,
            "operations": ops,
            "accuracy": "ESTIMATED_ALLOCATION",
        })
    return rows


def _format_duration(seconds: float) -> str:
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s"
