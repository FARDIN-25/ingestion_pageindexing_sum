"""Core VRAG pipeline tests."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pageindex.config import ConfigLoader
from pageindex.vrag import build_vrag_index, search, validate_tree
from pageindex.vrag.processing import extract_document, is_garbage_line

PDF = ROOT / "uploads" / "68bf0ef9_GST_Filing_and_Practice.pdf"


def test_garbage_detection():
    assert is_garbage_line("(Detected as legal book / act)")
    assert is_garbage_line("This instruction asks for a concise summary")
    assert not is_garbage_line("6.2 FORM GST REG- 06")


@pytest.mark.skipif(not PDF.exists(), reason="Sample PDF not found")
def test_build_index_no_hallucination():
    opt = ConfigLoader().load({"pipeline": "vrag", "fail_on_validation": "no"})
    result = build_vrag_index(str(PDF), opt=opt)

    def walk(nodes):
        for n in nodes:
            title = (n.get("title") or "").lower()
            assert "detected as" not in title
            assert "instruction asks" not in title
            raw = n.get("raw_content") or ""
            assert "detected as legal" not in raw.lower()
            if n.get("nodes"):
                walk(n["nodes"])

    structure = result["structure"]
    nodes = structure if isinstance(structure, list) else [structure]
    walk(nodes)
    errors = validate_tree(result["structure"])
    assert len(errors) < 5, errors[:3]


@pytest.mark.skipif(not PDF.exists(), reason="Sample PDF not found")
def test_no_duplicate_content_hashes():
    opt = ConfigLoader().load({"pipeline": "vrag", "fail_on_validation": "no"})
    result = build_vrag_index(str(PDF), opt=opt)
    hashes = []
    for n in result["flat_nodes"]:
        h = n.get("content_hash")
        if h:
            assert h not in hashes
            hashes.append(h)


@pytest.mark.skipif(not PDF.exists(), reason="Sample PDF not found")
def test_retrieval_gst_reg():
    opt = ConfigLoader().load({"pipeline": "vrag", "fail_on_validation": "no"})
    result = build_vrag_index(str(PDF), opt=opt)
    hits = search(result["structure"], "GST REG 06", top_k=3)
    assert hits
    top = hits[0]
    assert top["type"] not in ("ROOT", "FRONT_MATTER")
    blob = f"{top['title']} {top.get('micro_summary','')}".lower()
    assert "reg" in blob or "06" in blob
