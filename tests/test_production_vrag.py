"""Production VRAG integration tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pageindex.config import ConfigLoader
from pageindex.vrag import build_index, search, validate_index
from pageindex.vrag.processing import ContentDeduplicator
from pageindex.vrag.schema import SCHEMA_VERSION, NODE_TYPES
from pageindex.vrag.processing import contains_garbage_artifact, is_garbage_line
from pageindex.vrag.processing import is_paragraph_title, sha256_content

PDF = ROOT / "uploads" / "56a6a816_GST_Filing_and_Practice.pdf"
if not PDF.exists():
    candidates = list((ROOT / "uploads").glob("*GST*.pdf"))
    PDF = candidates[0] if candidates else PDF

FIXTURE_QUERIES = ROOT / "tests" / "fixtures" / "retrieval_queries.json"


def test_schema_types():
    assert "CONTENT" in NODE_TYPES
    assert SCHEMA_VERSION == "2.3"


def test_overlap_detection():
    a = "GST registration certificate download from portal step by step"
    b = "GST registration certificate download from portal procedure"
    from pageindex.vrag.processing import jaccard_similarity
    assert jaccard_similarity(a, b) > 0.5


def test_dedup_rejects_exact_duplicate():
    d = ContentDeduplicator(0.85)
    text = "Same exact content for testing deduplication path."
    ok1, h1 = d.register(text)
    ok2, h2 = d.register(text)
    assert ok1 and not ok2
    assert h1 == sha256_content(text)


def test_garbage():
    assert is_garbage_line("(Detected as legal book / act)")
    assert contains_garbage_artifact("Detected as legal book / act in body")


def test_paragraph_not_title():
    assert is_paragraph_title(
        "For the effective administration and implementation of the GST Act certain provisions"
    )


@pytest.mark.skipif(not PDF.exists(), reason="Sample PDF missing")
def test_full_build_passes_validation():
    opt = ConfigLoader().load({"pipeline": "vrag", "fail_on_validation": "yes"})
    result = build_index(str(PDF), opt=opt)
    assert result["schema_version"] == "2.3"
    assert result["retrieval_ready"] is True
    assert result["validation"]["valid"]
    assert result["retrieval_chunk_count"] > 20
    root = result["structure"]
    assert root["type"] == "ROOT"
    assert any(c["type"] == "FRONT_MATTER" for c in root.get("nodes", []))


@pytest.mark.skipif(not PDF.exists(), reason="Sample PDF missing")
def test_no_duplicate_hashes_in_flat_nodes():
    opt = ConfigLoader().load({"fail_on_validation": "no"})
    result = build_index(str(PDF), opt=opt)
    seen = set()
    for n in result["flat_nodes"]:
        if not n.get("retrieval_ready"):
            continue
        h = n["content_hash"]
        assert h not in seen, n["title"]
        seen.add(h)
        assert n["type"] == "CONTENT"
        assert len(n["title"]) <= 90


@pytest.mark.skipif(not PDF.exists(), reason="Sample PDF missing")
def test_retrieval_harness_passes():
    opt = ConfigLoader().load({"fail_on_validation": "no", "run_retrieval_tests": "yes"})
    result = build_index(str(PDF), opt=opt)
    assert result["retrieval_tests"]["pass_rate"] >= 0.9


@pytest.mark.skipif(not PDF.exists(), reason="Sample PDF missing")
@pytest.mark.parametrize("case", json.loads(FIXTURE_QUERIES.read_text(encoding="utf-8")))
def test_fixture_retrieval(case):
    opt = ConfigLoader().load({"fail_on_validation": "no", "run_retrieval_tests": "no"})
    result = build_index(str(PDF), opt=opt)
    hits = search(result["structure"], case["query"], top_k=5)
    assert hits, case["query"]
    blob = " ".join(
        f"{h.get('title','')} {h.get('micro_summary','')} {h.get('path','')}" for h in hits[:3]
    ).lower()
    assert any(n.lower() in blob for n in case["needles"]), f"{case['query']} -> {hits[0]['title']}"


@pytest.mark.skipif(not PDF.exists(), reason="Sample PDF missing")
def test_explainable_search_fields():
    opt = ConfigLoader().load({"fail_on_validation": "no", "run_retrieval_tests": "no"})
    result = build_index(str(PDF), opt=opt)
    hits = search(result["structure"], "GSTR2A", top_k=1)
    assert hits[0].get("score_breakdown")
    assert hits[0].get("matched_content_type")
    assert "traversal" in hits[0]


@pytest.mark.skipif(not PDF.exists(), reason="Sample PDF missing")
def test_reg06_quality_gate():
    opt = ConfigLoader().load({"fail_on_validation": "no", "run_retrieval_tests": "no"})
    result = build_index(str(PDF), opt=opt)
    hits = search(result["structure"], "how to download GST registration certificate", top_k=3)
    titles = " ".join(h["title"] for h in hits).lower()
    assert "6.2" in titles or "reg" in titles
