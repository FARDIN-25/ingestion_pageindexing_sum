"""Character-based sliding-window chunking with overlap for the Vector service."""

from __future__ import annotations

import hashlib
import re


def content_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def estimate_token_count(text: str) -> int:
    """Cheap token estimate (~4 chars/token). Good enough for metadata."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def chunk_text(
    text: str,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    no_chunking: bool = False,
) -> list[str]:
    """
    Split text into overlapping windows.

    Strategy:
    - Prefer paragraph boundaries (\\n\\n), then sentence-ish punctuation.
    - Fall back to hard character windows.
    - Adjacent chunks overlap by `chunk_overlap` characters.
    - Overlap is clamped to < chunk_size.
    - If no_chunking=True, return the full cleaned text as a single chunk.
    """
    cleaned = re.sub(r"[ \t]+", " ", (text or "")).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    if not cleaned:
        return []

    if no_chunking:
        return [cleaned]

    size = max(100, int(chunk_size))
    overlap = max(0, min(int(chunk_overlap), size - 1))

    if len(cleaned) <= size:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    n = len(cleaned)

    while start < n:
        end = min(start + size, n)
        if end < n:
            window = cleaned[start:end]
            # Prefer break near the end of the window.
            break_at = _best_break(window)
            if break_at >= size // 3:
                end = start + break_at
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(0, end - overlap)
        # Avoid infinite loops on pathological overlap.
        if start >= end:
            start = end

    return chunks


def _best_break(window: str) -> int:
    """Return local index to break at, or -1."""
    # Paragraph break
    idx = window.rfind("\n\n")
    if idx >= 0:
        return idx + 2
    # Sentence-ish
    for sep in (". ", "? ", "! ", ".\n", "?\n", "!\n"):
        idx = window.rfind(sep)
        if idx >= 0:
            return idx + len(sep)
    # Single newline
    idx = window.rfind("\n")
    if idx >= 0:
        return idx + 1
    # Space
    idx = window.rfind(" ")
    if idx >= 0:
        return idx + 1
    return -1
