"""Usage accounting data models."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class UsageEvent:
    job_id: str
    document_id: str
    page_number: int
    operation: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    credits_used: float
    latency_ms: int
    cache_hit: bool
    timestamp: str
    accuracy: str = "ESTIMATED"
    status: str = "success"
    error_message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    event_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["metadata"] = dict(self.metadata)
        return d


@dataclass
class JobRecord:
    job_id: str
    document_id: str
    document_name: str
    pipeline: str
    status: str
    page_count: int
    total_credits: float
    total_tokens: int
    total_latency_ms: int
    retries: int
    failed_pages: int
    cache_hits: int
    cache_misses: int
    credits_saved_cache: float
    accuracy_mix: str
    started_at: str
    finished_at: str = ""
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
