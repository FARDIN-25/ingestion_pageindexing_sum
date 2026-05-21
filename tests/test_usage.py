"""Credit usage accounting tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pageindex.usage.meter import UsageMeter
from pageindex.usage.service import UsageService
from pageindex.usage.store import UsageStore


@pytest.fixture
def store(tmp_path):
    return UsageStore(tmp_path / "test_usage.db")


def test_event_and_job_totals(store):
    meter = UsageMeter(
        job_id="job1",
        document_id="doc1",
        document_name="test.pdf",
        pipeline="vrag",
        store=store,
    )
    meter.record_upload(1000, latency_ms=5)
    meter.record_local_stage(
        "ocr_extraction",
        page_start=1,
        page_end=1,
        text_sample="GST registration certificate download",
        latency_ms=10,
    )
    report = meter.complete(status="success", page_count=2)
    assert report["overview"]["total_pages"] == 2
    assert "accuracy_mix" in report
    assert len(report["event_table"]) >= 2


def test_reverse_trace(store):
    meter = UsageMeter(job_id="job2", document_id="doc2", document_name="a.pdf", store=store)
    meter.record(
        "compression_generation",
        page_number=4,
        credits_used=0.21,
        accuracy="ESTIMATED",
        input_tokens=100,
    )
    meter.record(
        "ocr_extraction",
        page_number=5,
        credits_used=0.12,
        accuracy="ESTIMATED",
    )
    meter.complete(page_count=5)
    trace = UsageService(store).reverse_trace("job2", credits=0.33)
    assert trace["total_attributed"] >= 0.33 - 0.01


def test_timeline_and_breakdown(store):
    meter = UsageMeter(job_id="job3", document_id="doc3", document_name="b.pdf", store=store)
    meter.record("indexing", credits_used=1.0, accuracy="ESTIMATED")
    meter.record("compression_generation", credits_used=2.0, accuracy="ESTIMATED")
    meter.complete(page_count=1)
    svc = UsageService(store)
    tl = svc.timeline("job3")
    assert len(tl["timeline"]) == 2
    bd = svc.credits_breakdown(job_id="job3")
    assert sum(x["credits"] for x in bd["breakdown"]) == pytest.approx(3.0, rel=0.01)
