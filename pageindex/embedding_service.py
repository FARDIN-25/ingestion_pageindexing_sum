"""Local embedding model singleton (sentence-transformers).

Loaded once at startup; encode work runs in a worker thread so the asyncio
event loop stays free — same pattern as offloading PageIndex work.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any
from typing import Callable

from pageindex.env_settings import settings
from pageindex.log_util import log_error, log_info


class EmbeddingService:
    def __init__(self) -> None:
        self._model: Any | None = None
        self._lock = threading.Lock()
        self._load_error: str | None = None

    @property
    def model_name(self) -> str:
        return settings.EMBEDDING_MODEL_NAME

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def load_model(self) -> None:
        """Load the embedding model once (idempotent, thread-safe)."""
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                from sentence_transformers import SentenceTransformer

                device = (settings.EMBEDDING_DEVICE or "cpu").strip().lower()
                log_info(
                    "Loading embedding model %s on %s ...",
                    settings.EMBEDDING_MODEL_NAME,
                    device,
                )
                self._model = SentenceTransformer(
                    settings.EMBEDDING_MODEL_NAME,
                    device=device,
                )
                self._load_error = None
                log_info("Embedding model ready: %s", settings.EMBEDDING_MODEL_NAME)
            except Exception as exc:
                self._load_error = str(exc)[:500]
                log_error("Failed to load embedding model: %s", self._load_error)
                raise

    def _encode_sync(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            self.load_model()
        assert self._model is not None
        # normalize_embeddings=True → unit vectors, good for cosine / IVFFlat
        vectors = self._model.encode(
            texts,
            batch_size=len(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [row.tolist() for row in vectors]

    async def embed_texts(
        self,
        texts: list[str],
        progress_callback: Callable[[dict[str, int]], None] | None = None,
    ) -> list[list[float]]:
        """Embed texts in batches without blocking the event loop."""
        if not texts:
            return []

        if self._model is None:
            await asyncio.to_thread(self.load_model)

        batch_size = max(1, int(settings.EMBEDDING_BATCH_SIZE or 32))
        out: list[list[float]] = []
        total = len(texts)
        batch_count = (total + batch_size - 1) // batch_size
        for batch_index, i in enumerate(range(0, total, batch_size), start=1):
            batch = texts[i : i + batch_size]
            log_info(
                "Embedding batch %d/%d (%d texts)",
                batch_index,
                batch_count,
                len(batch),
            )
            vectors = await asyncio.to_thread(self._encode_sync, batch)
            out.extend(vectors)
            if progress_callback is not None:
                progress_callback(
                    {
                        "processed": len(out),
                        "total": total,
                        "batch_index": batch_index,
                        "batch_count": batch_count,
                    }
                )
        return out


embedding_service = EmbeddingService()
