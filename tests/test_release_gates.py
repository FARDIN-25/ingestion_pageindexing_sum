"""Release blocker gates — readiness, schema, overlap, artifacts."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pageindex.vrag.validation import apply_retrieval_readiness
from pageindex.vrag.processing import contains_garbage_artifact, sanitize_raw_content, strip_physical_index
from pageindex.vrag.schema import normalize_cloud_structure
from pageindex.vrag.processing import is_synthetic_title, jaccard_similarity
from pageindex.vrag.validation import validate_index
from pageindex.vrag.schema import SCHEMA_VERSION, export_node
from pageindex.config import ConfigLoader

PDF = ROOT / "uploads" / "56a6a816_GST_Filing_and_Practice.pdf"
if not PDF.exists():
    PDF = next((ROOT / "uploads").glob("*GST*.pdf"), PDF)


def test_schema_version():
    assert SCHEMA_VERSION == "2.3"


def test_physical_index_stripped():
    raw = "Intro text\n<physical_index_10>\nMore content"
    clean = sanitize_raw_content(raw)
    assert "physical_index" not in clean.lower()
    assert not contains_garbage_artifact(clean)


def test_synthetic_title_rejected():
    assert is_synthetic_title("Engagement Terms, Reporting Standards, Materiality")
    assert not is_synthetic_title("6.2 FORM GST REG-06")


def test_overlap_15_percent_detected():
    a = "GST registration certificate download portal step one two three"
    b = "GST registration certificate download portal step one two four"
    assert jaccard_similarity(a, b) > 0.15


def test_readiness_false_on_errors():
    root = {
        "node_id": "r1",
        "type": "ROOT",
        "title": "ROOT",
        "path": "ROOT",
        "nodes": [{
            "node_id": "c1",
            "type": "CONTENT",
            "title": "6.1 Test",
            "path": "ROOT > 6.1 Test",
            "parent_id": "r1",
            "raw_content": "x" * 100,
            "compressed_content": "y" * 80,
            "micro_summary": "Test section.",
            "aliases": ["6.1"],
            "keywords": ["test"],
            "synonyms": ["6.1"],
            "content_hash": "",
            "nodes": [],
        }],
    }
    report = apply_retrieval_readiness(root, ["content_hash mismatch on 6.1 Test"])
    assert report["retrieval_ready"] is False
    assert root["nodes"][0]["retrieval_ready"] is False


def test_cloud_normalizer_maps_legacy_text():
    cloud = [{
        "title": "Engagement Terms, Reporting Standards",
        "text": "6.2 FORM GST REG-06\nCertificate download steps from portal.",
        "summary": "Covers registration certificate.",
        "page_index": 12,
        "nodes": [],
    }]
    root = normalize_cloud_structure(cloud)
    exported = export_node(root)

    def first_content(node: dict) -> dict | None:
        if node.get("type") == "CONTENT" and node.get("raw_content"):
            return node
        for ch in node.get("nodes") or []:
            found = first_content(ch)
            if found:
                return found
        return None

    child = first_content(exported)
    assert child is not None
    assert "text" not in child or not child.get("text")
    assert child.get("raw_content")
    assert child.get("compressed_content")
    assert child.get("micro_summary")
    assert "6.2" in child.get("title", "") or "REG" in child.get("title", "").upper()


@pytest.mark.skipif(not PDF.exists(), reason="Sample PDF missing")
def test_full_build_retrieval_ready():
    from pageindex.vrag import build_index

    opt = ConfigLoader().load({
        "fail_on_validation": "yes",
        "run_retrieval_tests": "yes",
        "overlap_adjacent_threshold": 0.15,
    })
    result = build_index(str(PDF), opt=opt)
    assert result["retrieval_ready"] is True
    assert result["readiness"]["retrieval_ready"] is True
    assert result["schema_version"] == "2.3"
    for n in result["flat_nodes"]:
        assert n["retrieval_ready"] is True
        assert n.get("raw_content")
        assert n.get("compressed_content")
        assert n.get("micro_summary")
        assert "summary" not in n
        assert "prefix_summary" not in n
        assert "text" not in n
