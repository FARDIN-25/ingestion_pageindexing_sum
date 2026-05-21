"""SQLite persistence for credit usage events and jobs."""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import JobRecord, UsageEvent

_DEFAULT_DB = Path(__file__).resolve().parents[2] / "data" / "usage.db"
_lock = threading.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class UsageStore:
    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else _DEFAULT_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with _lock:
            conn = self._connect()
            try:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS jobs (
                        job_id TEXT PRIMARY KEY,
                        document_id TEXT NOT NULL,
                        document_name TEXT NOT NULL,
                        pipeline TEXT NOT NULL,
                        status TEXT NOT NULL,
                        page_count INTEGER DEFAULT 0,
                        total_credits REAL DEFAULT 0,
                        total_tokens INTEGER DEFAULT 0,
                        total_latency_ms INTEGER DEFAULT 0,
                        retries INTEGER DEFAULT 0,
                        failed_pages INTEGER DEFAULT 0,
                        cache_hits INTEGER DEFAULT 0,
                        cache_misses INTEGER DEFAULT 0,
                        credits_saved_cache REAL DEFAULT 0,
                        accuracy_mix TEXT DEFAULT 'ESTIMATED',
                        started_at TEXT NOT NULL,
                        finished_at TEXT,
                        error_message TEXT DEFAULT ''
                    );

                    CREATE TABLE IF NOT EXISTS usage_events (
                        event_id TEXT PRIMARY KEY,
                        job_id TEXT NOT NULL,
                        document_id TEXT NOT NULL,
                        page_number INTEGER NOT NULL DEFAULT 0,
                        operation TEXT NOT NULL,
                        provider TEXT NOT NULL,
                        model TEXT NOT NULL DEFAULT '',
                        input_tokens INTEGER DEFAULT 0,
                        output_tokens INTEGER DEFAULT 0,
                        credits_used REAL NOT NULL DEFAULT 0,
                        latency_ms INTEGER DEFAULT 0,
                        cache_hit INTEGER DEFAULT 0,
                        accuracy TEXT NOT NULL DEFAULT 'ESTIMATED',
                        status TEXT NOT NULL DEFAULT 'success',
                        error_message TEXT DEFAULT '',
                        metadata TEXT DEFAULT '{}',
                        timestamp TEXT NOT NULL,
                        FOREIGN KEY (job_id) REFERENCES jobs(job_id)
                    );

                    CREATE INDEX IF NOT EXISTS idx_events_job ON usage_events(job_id);
                    CREATE INDEX IF NOT EXISTS idx_events_doc ON usage_events(document_id);
                    CREATE INDEX IF NOT EXISTS idx_events_page ON usage_events(document_id, page_number);
                    CREATE INDEX IF NOT EXISTS idx_events_op ON usage_events(operation);

                    CREATE TABLE IF NOT EXISTS usage_alerts (
                        alert_id TEXT PRIMARY KEY,
                        job_id TEXT,
                        document_id TEXT,
                        alert_type TEXT NOT NULL,
                        message TEXT NOT NULL,
                        threshold REAL,
                        actual_value REAL,
                        created_at TEXT NOT NULL
                    );
                    """
                )
                conn.commit()
            finally:
                conn.close()

    def create_job(
        self,
        *,
        job_id: str,
        document_id: str,
        document_name: str,
        pipeline: str,
        page_count: int = 0,
    ) -> JobRecord:
        rec = JobRecord(
            job_id=job_id,
            document_id=document_id,
            document_name=document_name,
            pipeline=pipeline,
            status="running",
            page_count=page_count,
            total_credits=0.0,
            total_tokens=0,
            total_latency_ms=0,
            retries=0,
            failed_pages=0,
            cache_hits=0,
            cache_misses=0,
            credits_saved_cache=0.0,
            accuracy_mix="",
            started_at=_utc_now(),
        )
        with _lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO jobs (
                        job_id, document_id, document_name, pipeline, status,
                        page_count, started_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        rec.job_id,
                        rec.document_id,
                        rec.document_name,
                        rec.pipeline,
                        rec.status,
                        rec.page_count,
                        rec.started_at,
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        return rec

    def insert_event(self, event: UsageEvent) -> str:
        eid = event.event_id or uuid.uuid4().hex
        event.event_id = eid
        with _lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO usage_events (
                        event_id, job_id, document_id, page_number, operation,
                        provider, model, input_tokens, output_tokens, credits_used,
                        latency_ms, cache_hit, accuracy, status, error_message,
                        metadata, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        eid,
                        event.job_id,
                        event.document_id,
                        event.page_number,
                        event.operation,
                        event.provider,
                        event.model,
                        event.input_tokens,
                        event.output_tokens,
                        event.credits_used,
                        event.latency_ms,
                        1 if event.cache_hit else 0,
                        event.accuracy,
                        event.status,
                        event.error_message,
                        json.dumps(event.metadata),
                        event.timestamp or _utc_now(),
                    ),
                )
                conn.execute(
                    """
                    UPDATE jobs SET
                        total_credits = total_credits + ?,
                        total_tokens = total_tokens + ? + ?,
                        total_latency_ms = total_latency_ms + ?,
                        cache_hits = cache_hits + ?,
                        cache_misses = cache_misses + ?
                    WHERE job_id = ?
                    """,
                    (
                        event.credits_used,
                        event.input_tokens,
                        event.output_tokens,
                        event.latency_ms,
                        1 if event.cache_hit else 0,
                        0 if event.cache_hit else 1,
                        event.job_id,
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        return eid

    def finish_job(
        self,
        job_id: str,
        *,
        status: str,
        page_count: int | None = None,
        error_message: str = "",
        retries: int = 0,
        failed_pages: int = 0,
    ) -> None:
        with _lock:
            conn = self._connect()
            try:
                job = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
                if not job:
                    return
                events = conn.execute(
                    "SELECT accuracy, credits_used, cache_hit FROM usage_events WHERE job_id = ?",
                    (job_id,),
                ).fetchall()
                accuracies = {r["accuracy"] for r in events}
                if "EXACT" in accuracies and len(accuracies) > 1:
                    mix = "MIXED"
                elif accuracies == {"EXACT"}:
                    mix = "EXACT"
                elif accuracies == {"LOCAL"}:
                    mix = "LOCAL"
                else:
                    mix = "ESTIMATED"

                without_cache = sum(r["credits_used"] for r in events if not r["cache_hit"])
                with_cache = sum(r["credits_used"] for r in events)
                saved = max(0.0, without_cache - with_cache) if events else 0.0

                params: list[Any] = [status, _utc_now(), mix, saved, retries, failed_pages, error_message, job_id]
                sql = """
                    UPDATE jobs SET
                        status = ?, finished_at = ?, accuracy_mix = ?,
                        credits_saved_cache = ?, retries = ?, failed_pages = ?,
                        error_message = ?
                """
                if page_count is not None:
                    sql += ", page_count = ?"
                    params.insert(-1, page_count)
                sql += " WHERE job_id = ?"
                conn.execute(sql, params)
                conn.commit()
            finally:
                conn.close()

    def insert_alert(
        self,
        *,
        job_id: str,
        document_id: str,
        alert_type: str,
        message: str,
        threshold: float | None = None,
        actual_value: float | None = None,
    ) -> None:
        with _lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO usage_alerts (
                        alert_id, job_id, document_id, alert_type, message,
                        threshold, actual_value, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        uuid.uuid4().hex,
                        job_id,
                        document_id,
                        alert_type,
                        message,
                        threshold,
                        actual_value,
                        _utc_now(),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_events(
        self,
        *,
        job_id: str | None = None,
        document_id: str | None = None,
        page_number: int | None = None,
        limit: int = 5000,
    ) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            q = "SELECT * FROM usage_events WHERE 1=1"
            params: list[Any] = []
            if job_id:
                q += " AND job_id = ?"
                params.append(job_id)
            if document_id:
                q += " AND document_id = ?"
                params.append(document_id)
            if page_number is not None:
                q += " AND page_number = ?"
                params.append(page_number)
            q += " ORDER BY timestamp ASC LIMIT ?"
            params.append(limit)
            rows = conn.execute(q, params).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                d["cache_hit"] = bool(d.get("cache_hit"))
                d["metadata"] = json.loads(d.get("metadata") or "{}")
                out.append(d)
            return out
        finally:
            conn.close()

    def list_jobs(self, limit: int = 50) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_latest_job_for_document(self, document_id: str) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM jobs WHERE document_id = ? ORDER BY started_at DESC LIMIT 1",
                (document_id,),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()


_store: UsageStore | None = None


def get_store() -> UsageStore:
    global _store
    if _store is None:
        _store = UsageStore()
    return _store
