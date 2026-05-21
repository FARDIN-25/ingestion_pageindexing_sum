"""Retrieval harness for production vectorless RAG index."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pageindex.config import ConfigLoader
from pageindex.vrag import build_vrag_index, search
from pageindex.vrag.processing import is_garbage_line

PDF = ROOT / "uploads" / "68bf0ef9_GST_Filing_and_Practice.pdf"

RETRIEVAL_QUERIES = [
    ("what is GST REG 06", ["reg", "06", "registration"]),
    ("how to download GST registration certificate", ["certificate", "registration", "download"]),
    ("GSTR-1 due date", ["gstr", "due"]),
    ("what is GSTR2A", ["gstr", "2a"]),
    ("input tax credit verification", ["tax", "credit"]),
]


def test_garbage_filter():
    assert is_garbage_line("(Detected as legal book / act)")
    assert not is_garbage_line("6.2 FORM GST REG- 06")


@pytest.mark.skipif(not PDF.exists(), reason="Sample GST PDF missing")
def test_build_schema_and_validation():
    opt = ConfigLoader().load({"pipeline": "vrag", "fail_on_validation": "yes"})
    result = build_vrag_index(str(PDF), opt=opt)
    assert result["pipeline"] == "vrag"
    assert result["validation"]["valid"]
    assert result["retrieval_chunk_count"] > 10

    for leaf in result["flat_nodes"]:
        assert leaf.get("raw_content")
        assert leaf.get("compressed_content")
        assert leaf.get("micro_summary")
        assert leaf.get("content_hash")
        assert leaf["token_count_raw"] >= leaf["token_count_compressed"] or leaf["token_count_compressed"] > 0
        assert "detected as" not in (leaf.get("title") or "").lower()


@pytest.mark.skipif(not PDF.exists(), reason="Sample GST PDF missing")
@pytest.mark.parametrize("query,needles", RETRIEVAL_QUERIES)
def test_lexical_retrieval(query, needles):
    opt = ConfigLoader().load({"pipeline": "vrag", "fail_on_validation": "no"})
    result = build_vrag_index(str(PDF), opt=opt)
    hits = search(result["structure"], query, top_k=3)
    assert hits, f"No hits for: {query}"
    top = hits[0]
    blob = f"{top.get('title','')} {top.get('micro_summary','')}".lower()
    assert any(n in blob for n in needles), f"Weak match for {query}: {top}"
