"""Credit pricing — EXACT when provider returns usage; else deterministic ESTIMATED."""
from __future__ import annotations

import os
from typing import Any

from .constants import Accuracy, Operation


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


# PageIndex cloud estimates (credits) — override via .env; always labeled ESTIMATED unless API returns exact
RATES = {
    "per_page_ocr": _env_float("CREDIT_RATE_PAGE_OCR", 0.12),
    "per_page_structure": _env_float("CREDIT_RATE_PAGE_STRUCTURE", 0.03),
    "per_page_compression": _env_float("CREDIT_RATE_PAGE_COMPRESSION", 0.21),
    "per_page_summary": _env_float("CREDIT_RATE_PAGE_SUMMARY", 0.04),
    "per_page_metadata": _env_float("CREDIT_RATE_PAGE_METADATA", 0.02),
    "per_1k_input_tokens": _env_float("CREDIT_RATE_PER_1K_INPUT", 0.01),
    "per_1k_output_tokens": _env_float("CREDIT_RATE_PER_1K_OUTPUT", 0.015),
    "upload_base": _env_float("CREDIT_RATE_UPLOAD_BASE", 0.05),
    "poll_per_call": _env_float("CREDIT_RATE_POLL", 0.002),
}

OPERATION_PAGE_RATE: dict[str, str] = {
    Operation.OCR_EXTRACTION.value: "per_page_ocr",
    Operation.STRUCTURE_DETECTION.value: "per_page_structure",
    Operation.COMPRESSION.value: "per_page_compression",
    Operation.MICRO_SUMMARY.value: "per_page_summary",
    Operation.ALIAS_GENERATION.value: "per_page_metadata",
    Operation.KEYWORD_GENERATION.value: "per_page_metadata",
    Operation.DOCUMENT_PARSING.value: "per_page_structure",
    Operation.HEADING_DETECTION.value: "per_page_structure",
    Operation.CHUNK_GENERATION.value: "per_page_structure",
}


def estimate_tokens_from_text(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def credits_from_tokens(
    input_tokens: int,
    output_tokens: int = 0,
    *,
    pages: int = 1,
    operation: str | None = None,
) -> tuple[float, str]:
    """Return (credits, accuracy label)."""
    credits = 0.0
    if operation and operation in OPERATION_PAGE_RATE:
        rate_key = OPERATION_PAGE_RATE[operation]
        credits += RATES[rate_key] * max(pages, 1)
    credits += (input_tokens / 1000.0) * RATES["per_1k_input_tokens"]
    credits += (output_tokens / 1000.0) * RATES["per_1k_output_tokens"]
    return round(credits, 6), Accuracy.ESTIMATED.value


def credits_for_upload(file_bytes: int) -> tuple[float, str]:
    mb = file_bytes / (1024 * 1024)
    est = RATES["upload_base"] + mb * 0.01
    return round(est, 6), Accuracy.ESTIMATED.value


def credits_for_poll(attempt: int) -> tuple[float, str]:
    return round(RATES["poll_per_call"] * attempt, 6), Accuracy.ESTIMATED.value


def parse_provider_usage(body: Any) -> tuple[float, int, int, str] | None:
    """
    Extract EXACT credits/tokens from PageIndex API response if present.
    Returns (credits, input_tokens, output_tokens, accuracy) or None.
    """
    if not isinstance(body, dict):
        return None
    usage = body.get("usage") or body.get("credit_usage") or body.get("billing")
    if not isinstance(usage, dict):
        # top-level fields
        if "credits_used" in body or "credits" in body:
            usage = body
        else:
            return None
    credits = usage.get("credits_used") or usage.get("credits") or usage.get("total_credits")
    if credits is None:
        return None
    inp = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
    out = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
    return float(credits), inp, out, Accuracy.EXACT.value


def local_zero_credits() -> tuple[float, str]:
    return 0.0, Accuracy.LOCAL.value
