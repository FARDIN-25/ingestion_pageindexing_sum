"""Ingestion schema: compressed_content, hierarchy, micro_summary."""

from __future__ import annotations



import sys

from pathlib import Path



import pytest



ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))



from pageindex.vrag.schema import (

    SCHEMA_VERSION,

    export_node,

    empty_node,

    enrich_node,

    FORBIDDEN_EXPORT_FIELDS,

)

from pageindex.vrag.validation import validate_index, ValidationError

from pageindex.vrag.processing import line_range_char_span, micro_summary_from_content

from pageindex.vrag.processing import DocLine, sha256_content





def test_schema_version_24():

    assert SCHEMA_VERSION == "2.4"





def test_export_node_includes_compressed():

    node = empty_node("6.1 Test", "CONTENT", "ROOT > 6.1", 7, "p1")

    node["node_id"] = "c1"

    node["raw_content"] = "GST registration certificate download steps from portal." * 4

    enrich_node(node)

    node["micro_summary"] = micro_summary_from_content(node["title"], node["raw_content"])

    node["aliases"] = ["6.1"]

    node["keywords"] = ["gst"]

    node["synonyms"] = ["6.1"]

    node["page_start"] = 1

    node["page_end"] = 2

    node["char_start"] = 10

    node["char_end"] = 80

    node["retrieval_ready"] = True



    out = export_node(node, include_children=False)

    for field in FORBIDDEN_EXPORT_FIELDS:

        assert field not in out

    assert out["raw_content"]

    assert out["compressed_content"]

    assert out["micro_summary"]

    assert len(out["micro_summary"].split(". ")) <= 3 or len(out["micro_summary"]) < 300





def test_micro_summary_from_raw_only():

    raw = "FORM GST REG-06 provides the registration certificate. Download from the GST portal."

    micro = micro_summary_from_content("6.2 REG-06", raw)

    assert "compressed" not in micro.lower()

    assert len(micro.split(". ")) <= 3





def test_line_char_span_offsets():

    lines = [

        DocLine(1, 1, 0, "Hello", char_start=0, char_end=5),

        DocLine(1, 2, 1, "World", char_start=6, char_end=11),

    ]

    cs, ce = line_range_char_span(lines, 0, 2)

    assert cs == 0

    assert ce == 11





def test_validate_requires_compressed_field():

    root = empty_node("ROOT", "ROOT", "ROOT", 0)

    root["node_id"] = "r1"

    doc = empty_node("Doc", "DOCUMENT", "ROOT/DOCUMENT", 1, "r1")

    doc["node_id"] = "d1"

    ch = empty_node("Ch", "CHAPTER", "path", 2, "d1")

    ch["node_id"] = "ch1"

    sec = empty_node("Body", "SECTION", "path", 3, "ch1")

    sec["node_id"] = "s1"

    leaf = empty_node("6.1", "CONTENT", "path", 7, "s1")

    leaf["node_id"] = "c1"

    leaf["raw_content"] = "x" * 120

    leaf["micro_summary"] = "Test section about GST forms."

    leaf["content_hash"] = sha256_content(leaf["raw_content"])

    leaf["aliases"] = ["6.1"]

    leaf["keywords"] = ["gst"]

    leaf["synonyms"] = ["6.1"]

    leaf["page_start"] = 1

    leaf["page_end"] = 1

    leaf["char_start"] = 1

    leaf["char_end"] = 50

    sec["nodes"] = [leaf]

    ch["nodes"] = [sec]

    doc["nodes"] = [ch]

    root["nodes"] = [doc]



    errors = validate_index(root, strict=False)

    assert any("Missing compressed_content" in e for e in errors)





def test_valid_minimal_tree_passes():

    root = empty_node("ROOT", "ROOT", "ROOT", 0)

    root["node_id"] = "r1"

    doc = empty_node("Doc", "DOCUMENT", "ROOT/DOCUMENT", 1, "r1")

    doc["node_id"] = "d1"

    ch = empty_node("Ch", "CHAPTER", "ROOT/DOCUMENT/CHAPTER", 2, "d1")

    ch["node_id"] = "ch1"

    sec = empty_node("Body", "SECTION", "ROOT/DOCUMENT/CHAPTER/SECTION", 3, "ch1")

    sec["node_id"] = "s1"

    raw = (
        "FORM GST REG-06 certificate download procedure on the GST portal for taxpayers. "
        "The registration certificate must be downloaded after approval under rule 8. "
    ) * 5

    leaf = empty_node("6.2 REG-06", "CONTENT", "path", 7, "s1")

    leaf["node_id"] = "c1"

    leaf["raw_content"] = raw

    leaf["micro_summary"] = micro_summary_from_content(leaf["title"], raw)

    enrich_node(leaf)

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



    errors = validate_index(root, strict=True)

    assert not errors

