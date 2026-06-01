"""Automated retrieval-quality checks (titles, TOC, compression, traversal)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pageindex.vrag.cloud_normalize import normalize_cloud_structure, resolve_cloud_title
from pageindex.vrag.processing import (
    compress_text,
    compression_ratio,
    is_legal_section_reference_title,
    is_paragraph_title,
    is_table_of_contents_content,
    sanitize_raw_content,
    strip_physical_index,
)
from pageindex.vrag.schema import enrich_node, export_node, empty_node
from pageindex.vrag.validation import apply_retrieval_readiness, validate_index
from pageindex.vrag.pipeline import LexicalRetriever
from pageindex.vrag.processing import sha256_content, micro_summary_from_content


def test_reject_legal_section_fragment_titles():
    assert is_legal_section_reference_title("Section 44A B of the Act.")
    assert is_legal_section_reference_title("Section 44A DA:")
    assert is_paragraph_title("Section 44A E or section 44BB or section 44BBB")
    assert not is_legal_section_reference_title("Introduction")
    assert not is_legal_section_reference_title("Provisions of section 44AB")


def test_toc_detected_and_not_retrieval_ready():
    raw = "Introduction .......... 1\nBackground .......... 5\nPenalty .......... 90\n"
    assert is_table_of_contents_content("Contents", raw)
    cloud = [{"title": "Contents", "text": raw, "nodes": []}]
    root = normalize_cloud_structure(cloud)

    def find_toc(node: dict) -> dict | None:
        if node.get("type") == "TABLE_OF_CONTENTS":
            return node
        for ch in node.get("nodes") or []:
            found = find_toc(ch)
            if found:
                return found
        return None

    toc = find_toc(root)
    assert toc is not None
    assert toc.get("retrieval_ready") is False


def test_physical_index_stripped_from_content():
    raw = "Intro\n<physical_index_5>\nMore"
    clean = sanitize_raw_content(raw)
    assert "physical_index" not in clean.lower()


def test_compressed_content_ratio():
    raw = (
        "FORM GST REG-06 provides registration certificate. "
        "Download from portal. The study material has been prepared by university faculty. "
        "Registration shall be granted under section 7 of the CGST Act."
    ) * 3
    compressed = compress_text(raw, target_ratio=0.70, min_ratio=0.60)
    ratio = compression_ratio(raw, compressed)
    assert 0.50 <= ratio <= 0.92
    assert "REG-06" in compressed or "registration" in compressed.lower()


def _minimal_ready_tree() -> dict:
    root = empty_node("ROOT", "ROOT", "ROOT", 0)
    root["node_id"] = "r1"
    doc = empty_node("Doc", "DOCUMENT", "ROOT/DOCUMENT", 1, "r1")
    doc["node_id"] = "d1"
    ch = empty_node("Ch1", "CHAPTER", "ROOT/DOCUMENT/CHAPTER", 2, "d1")
    ch["node_id"] = "ch1"
    sec = empty_node("Body", "SECTION", "ROOT/DOCUMENT/CHAPTER/SECTION", 3, "ch1")
    sec["node_id"] = "s1"
    raw = (
        "FORM GST REG-06 certificate download procedure on the GST portal for taxpayers. "
        "Registration certificate must be downloaded after approval under rule 8. "
    ) * 5
    leaf = empty_node("6.2 REG-06", "CONTENT", "ROOT > 6.2", 7, "s1")
    leaf["node_id"] = "c1"
    leaf["raw_content"] = raw
    enrich_node(leaf)
    leaf["micro_summary"] = micro_summary_from_content(leaf["title"], raw)
    leaf["page_start"] = 1
    leaf["page_end"] = 2
    leaf["char_start"] = 0
    leaf["char_end"] = len(raw)
    sec["nodes"] = [leaf]
    ch["nodes"] = [sec]
    doc["nodes"] = [ch]
    root["nodes"] = [doc]
    leaf["children"] = []
    sec["children"] = ["c1"]
    ch["children"] = ["s1"]
    doc["children"] = ["ch1"]
    root["children"] = ["d1"]
    return root


def test_strict_readiness_requires_all_gates():
    root = _minimal_ready_tree()
    errors = validate_index(root, strict=False)
    readiness = apply_retrieval_readiness(root, errors)
    assert readiness["retrieval_ready"] is True
    leaf = root["nodes"][0]["nodes"][0]["nodes"][0]["nodes"][0]
    assert leaf["retrieval_ready"] is True
    assert leaf.get("compressed_content")
    assert leaf.get("keywords")


def test_lexical_retrieval_title_keyword_traversal():
    root = _minimal_ready_tree()
    apply_retrieval_readiness(root, [])
    for n in root["nodes"][0]["nodes"][0]["nodes"][0]["nodes"]:
        n["retrieval_ready"] = True
        n["is_retrieval_chunk"] = True

    retriever = LexicalRetriever(root)
    hits = retriever.search("GST REG 06", top_k=3)
    assert hits
    assert hits[0].get("matched_title") or hits[0]["score_breakdown"].get("title_contains_query")

    hits_kw = retriever.search("registration certificate download", top_k=3)
    assert hits_kw
    trav = hits_kw[0].get("traversal") or {}
    assert trav.get("parent_id") or trav.get("parent_title")


def test_cloud_title_not_from_paragraph():
    item = {
        "title": "Section 44A B of the Act.",
        "text": "Introduction\n\nBackground material about tax audit provisions.",
    }
    title = resolve_cloud_title(item)
    assert title != "Section 44A B of the Act."
    assert title in ("Introduction", "") or "Introduction" in title


def test_export_includes_compressed():
    root = _minimal_ready_tree()
    leaf = root["nodes"][0]["nodes"][0]["nodes"][0]["nodes"][0]
    out = export_node(leaf, include_children=False)
    assert out.get("compressed_content")
    assert out.get("node_id") and out.get("parent_id") and out.get("path")
