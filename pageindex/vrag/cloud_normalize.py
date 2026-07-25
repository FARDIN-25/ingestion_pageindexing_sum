"""Preserve PageIndex cloud tree nesting under strict VRAG hierarchy."""
from __future__ import annotations

import re
from typing import Any

from .processing import (
    clean_title,
    contains_garbage_artifact,
    container_micro_summary,
    is_legal_section_reference_title,
    is_paragraph_title,
    is_structural_heading_line,
    is_synthetic_title,
    is_table_of_contents_content,
    is_valid_heading_title,
    micro_summary_from_content,
    sanitize_raw_content,
)
from .schema import (
    CONTAINER_TYPES,
    FRONT_MATTER_TYPES,
    LEVEL,
    VALID_PARENT,
    empty_node,
    enrich_node,
    finalize_children,
    normalize_type,
)
from .path_utils import rebuild_paths

RE_GENERIC_SECTION = re.compile(r"^section[_\s-]*\d+$", re.I)
RE_CHAPTER = re.compile(
    r"^(?:chapter|unit|part)\s+([IVXLC\d]+)(?:\s*[:\-–]\s*(.+))?$",
    re.I,
)
RE_BOOK_SECTION = re.compile(
    r"^(?:introduction|background|definitions?|provisions?|summary|conclusion|"
    r"appendix|glossary|preface|acknowledg(?:e)?ments?|table\s+of\s+contents|"
    r"contents|index|foreword|overview|penalty|liability|profession|business|"
    r"sales|turnover|gross\s+receipts|specified\s+date|tax\s+audit)\b",
    re.I,
)
RE_APPENDIX = re.compile(r"^appendix\b", re.I)


def _cloud_page_span(item: dict) -> tuple[int, int]:
    ps = int(item.get("page_start") or item.get("page_index") or item.get("start_index") or 0)
    pe = int(item.get("page_end") or item.get("end_index") or ps)
    if pe < ps and ps > 0:
        pe = ps
    return ps, max(pe, ps)


def _cloud_raw(item: dict) -> str:
    raw = (item.get("raw_content") or item.get("text") or item.get("content") or "").strip()
    if not raw:
        legacy = (item.get("summary") or item.get("prefix_summary") or "").strip()
        if legacy and len(legacy.split()) > 30:
            raw = legacy
    return sanitize_raw_content(raw)


def resolve_cloud_title(item: dict, *, fallback_type: str = "TOPIC") -> str:
    """Use only declared structural titles — never paragraph fragments."""
    for key in ("title", "heading", "section_title", "name"):
        declared = (item.get(key) or "").strip()
        if RE_GENERIC_SECTION.match(declared):
            declared = ""
        if not declared:
            continue
        if is_legal_section_reference_title(declared):
            continue
        if is_paragraph_title(declared) and not is_valid_heading_title(declared):
            continue
        if not is_synthetic_title(declared):
            return clean_title(declared, fallback_type)[:85]

    raw = _cloud_raw(item)
    for line in raw.split("\n")[:6]:
        line = line.strip()
        if not line or len(line) > 100:
            continue
        if is_valid_heading_title(line):
            return clean_title(line, fallback_type)[:85]

    m = re.match(r"^(\d+\.\d+(?:\.\d+)?)\s+(.+)$", raw)
    if m and len(m.group(2)) < 80 and is_valid_heading_title(f"{m.group(1)} {m.group(2)}"):
        return clean_title(f"{m.group(1)} {m.group(2)}", fallback_type)[:85]

    for candidate in (item.get("title") or "", raw.split("\n")[0] if raw else ""):
        m = RE_CHAPTER.match(candidate.strip())
        if m:
            suffix = (m.group(2) or "").strip()
            return clean_title(f"Chapter {m.group(1)} {suffix}".strip(), "CHAPTER")[:85]
        m = RE_BOOK_SECTION.match(candidate.strip())
        if m and len(candidate.strip()) < 90:
            return clean_title(candidate.strip(), fallback_type)[:85]

    return ""


def _infer_container_type(title: str, *, has_children: bool) -> str:
    t = title.strip()
    if not t and has_children:
        return "TOPIC"
    if RE_APPENDIX.match(t):
        return "APPENDIX"
    if RE_CHAPTER.match(t):
        return "CHAPTER"
    if RE_BOOK_SECTION.match(t):
        low = t.lower()
        if "contents" in low or "table of" in low:
            return "TABLE_OF_CONTENTS"
        if "preface" in low or "foreword" in low:
            return "PREFACE"
        if "syllabus" in low:
            return "SYLLABUS"
        if "objective" in low:
            return "OBJECTIVES"
        if "reading" in low or "reference" in low or "glossary" in low:
            return "REFERENCES"
        return "TOPIC"
    if re.match(r"^\d+\.\d+\.\d+", t):
        return "SUBTOPIC"
    if re.match(r"^\d+\.\d+", t):
        return "TOPIC"
    if re.match(r"^section\s+", t, re.I) and len(t) < 60:
        return "SECTION"
    if has_children:
        return "TOPIC"
    return "TOPIC"


def _is_front_matter_title(title: str) -> bool:
    t = title.lower().strip()
    return bool(
        RE_BOOK_SECTION.match(title)
        and any(
            k in t
            for k in (
                "preface",
                "syllabus",
                "contents",
                "table of",
                "objective",
                "reading",
                "acknowledg",
                "glossary",
                "foreword",
                "committee",
                "publication",
            )
        )
    )


def _allowed_child(parent_type: str, child_type: str) -> bool:
    allowed = VALID_PARENT.get(parent_type, frozenset())
    return child_type in allowed


def _fallback_container_title(ntype: str, item: dict) -> str:
    ps, pe = _cloud_page_span(item)
    if ntype == "CHAPTER":
        return f"Chapter (pp. {ps}-{pe})" if pe > ps else f"Chapter (p. {ps})"
    if ntype == "TOPIC":
        return f"Topic (pp. {ps}-{pe})" if pe > ps else "Topic block"
    return ntype.replace("_", " ").title()


def map_cloud_subtree(item: dict, parent: dict, counter: list[int]) -> dict | None:
    """Recursively map one cloud node, preserving nested children."""
    if not isinstance(item, dict):
        return None

    children_in = [c for c in (item.get("nodes") or []) if isinstance(c, dict)]
    raw = _cloud_raw(item)
    has_body = len(raw) >= 20 and not contains_garbage_artifact(raw)
    title = resolve_cloud_title(item)

    if children_in:
        ntype = _infer_container_type(title, has_children=True)
        if not title:
            title = _fallback_container_title(ntype, item)
        if not _allowed_child(normalize_type(parent.get("type", "")), ntype):
            ntype = "TOPIC" if _allowed_child(parent.get("type", ""), "TOPIC") else "CONTENT"
        out = empty_node(
            title,
            ntype,
            f"{parent.get('path', 'ROOT')} > {title}",
            LEVEL.get(ntype, 4),
            parent["node_id"],
        )
        counter[0] += 1
        out["node_id"] = item.get("node_id") or f"cloud_{counter[0]:05d}"
        ps, pe = _cloud_page_span(item)
        out["page_start"], out["page_end"] = ps, pe
        out["micro_summary"] = container_micro_summary(title, [])
        if ntype == "TABLE_OF_CONTENTS" or is_table_of_contents_content(title, raw):
            out["type"] = "TABLE_OF_CONTENTS"
            out["is_front_matter"] = True
            out["retrieval_ready"] = False
        for ch in children_in:
            mapped = map_cloud_subtree(ch, out, counter)
            if mapped:
                out["nodes"].append(mapped)
        if not out["nodes"] and has_body:
            return _map_content_leaf(item, parent, counter, title or _fallback_container_title("TOPIC", item), raw)
        out["children"] = [c["node_id"] for c in out["nodes"]]
        out["children_ids"] = out["children"]
        titles = [c.get("title", "") for c in out["nodes"]]
        out["micro_summary"] = container_micro_summary(title, titles)
        return out

    if has_body:
        if not title or is_paragraph_title(title) or is_legal_section_reference_title(title):
            for line in raw.split("\n")[:8]:
                if is_valid_heading_title(line.strip()):
                    title = clean_title(line.strip(), "CONTENT")[:85]
                    break
        # PageIndex often returns truncated / paragraph-like titles on real body
        # nodes. Never drop has_body leaves — otherwise document_nodes lose text.
        if not title or is_paragraph_title(title) or is_legal_section_reference_title(title):
            declared = (
                (item.get("title") or item.get("heading") or item.get("section_title") or "")
                .strip()
            )
            fallback = declared or (title or "").strip()
            if not fallback:
                for line in raw.split("\n")[:8]:
                    line = line.strip()
                    if line:
                        fallback = line
                        break
            title = clean_title(fallback or "Content", "CONTENT")[:85]
        if is_table_of_contents_content(title, raw):
            toc = empty_node(
                clean_title(title, "TABLE_OF_CONTENTS")[:85],
                "TABLE_OF_CONTENTS",
                f"{parent.get('path')} > {title}",
                LEVEL["TABLE_OF_CONTENTS"],
                parent["node_id"],
            )
            counter[0] += 1
            toc["node_id"] = item.get("node_id") or f"cloud_{counter[0]:05d}"
            toc["raw_content"] = raw
            toc["is_front_matter"] = True
            toc["retrieval_ready"] = False
            toc["micro_summary"] = "Table of contents (navigation only)."
            parent["nodes"].append(toc)
            return toc
        return _map_content_leaf(item, parent, counter, title, raw)
    return None


def _map_content_leaf(
    item: dict,
    parent: dict,
    counter: list[int],
    title: str,
    raw: str,
) -> dict:
    ptype = normalize_type(parent.get("type", ""))
    parent_out = None
    attach_parent = parent
    if (
        ptype in ("UNIT", "SECTION", "CHAPTER")
        and re.match(r"^\d+\.\d+", title)
        and _allowed_child(ptype, "TOPIC")
    ):
        wrap = empty_node(
            clean_title(title, "TOPIC")[:85],
            "TOPIC",
            f"{parent.get('path')} > {title}",
            LEVEL["TOPIC"],
            parent["node_id"],
        )
        counter[0] += 1
        wrap["node_id"] = f"cloud_{counter[0]:05d}"
        attach_parent = wrap
        parent_out = wrap

    out = empty_node(
        clean_title(title, "CONTENT")[:85],
        "CONTENT",
        f"{attach_parent.get('path')} > {title}",
        LEVEL["CONTENT"],
        attach_parent["node_id"],
    )
    counter[0] += 1
    out["node_id"] = item.get("node_id") or f"cloud_{counter[0]:05d}"
    ps, pe = _cloud_page_span(item)
    out["page_start"], out["page_end"] = ps, pe
    out["raw_content"] = raw
    micro = (item.get("micro_summary") or item.get("prefix_summary") or "").strip()
    if not micro or len(micro.split()) > 40:
        micro = micro_summary_from_content(out["title"], raw)
    out["micro_summary"] = micro
    enrich_node(out)
    out["is_front_matter"] = (
        parent.get("type") == "FRONT_MATTER"
        or normalize_type(parent.get("type", "")) in FRONT_MATTER_TYPES
        or _is_front_matter_title(title)
    )
    out["retrieval_ready"] = False
    out["is_retrieval_chunk"] = False

    if parent_out is not None:
        parent_out["nodes"] = [out]
        parent_out["children"] = [out["node_id"]]
        parent_out["children_ids"] = out["children"]
        return parent_out
    return out


def _cloud_top_level_items(structure: Any) -> list[dict]:
    if isinstance(structure, list):
        return [x for x in structure if isinstance(x, dict)]
    if isinstance(structure, dict):
        if structure.get("nodes"):
            return [x for x in structure["nodes"] if isinstance(x, dict)]
        if structure.get("title") or structure.get("text"):
            return [structure]
    return []


def _ensure_chapter(document: dict, chapters: dict[str, dict], key: str, title: str) -> dict:
    if key in chapters:
        return chapters[key]
    ch = empty_node(
        title or "Chapter",
        "CHAPTER",
        f"{document.get('path')}/CHAPTER/{title[:40]}",
        LEVEL["CHAPTER"],
        document["node_id"],
    )
    ch["node_id"] = f"ch_{len(chapters) + 1:04d}"
    ch["micro_summary"] = container_micro_summary(ch["title"], [])
    document["nodes"].append(ch)
    chapters[key] = ch
    return ch


def normalize_cloud_structure(structure: Any) -> dict[str, Any]:
    """
    ROOT → DOCUMENT → FRONT_MATTER | CHAPTER* → nested cloud subtree.
    Preserves PageIndex API nesting; never flattens all nodes under one SECTION.
    """
    counter = [0]
    root = empty_node("ROOT", "ROOT", "ROOT", LEVEL["ROOT"])
    root["node_id"] = "root_0001"

    document = empty_node("Document", "DOCUMENT", "ROOT/DOCUMENT", LEVEL["DOCUMENT"], root["node_id"])
    document["node_id"] = "doc_0001"
    root["nodes"].append(document)

    fm = empty_node(
        "Front Matter",
        "FRONT_MATTER",
        "ROOT/DOCUMENT/FRONT_MATTER",
        LEVEL["FRONT_MATTER"],
        document["node_id"],
    )
    fm["node_id"] = "fm_0001"

    chapters: dict[str, dict] = {}
    default_chapter: dict | None = None

    for item in _cloud_top_level_items(structure):
        title_guess = resolve_cloud_title(item) or _cloud_raw(item)[:0]
        raw_preview = _cloud_raw(item)

        if _is_front_matter_title(title_guess) or is_table_of_contents_content(title_guess, raw_preview):
            mapped = map_cloud_subtree(item, fm, counter)
            if mapped:
                fm["nodes"].append(mapped)
            continue

        if RE_APPENDIX.match(title_guess):
            appendix = empty_node(
                title_guess or "Appendix",
                "APPENDIX",
                f"{document['path']}/APPENDIX",
                LEVEL["APPENDIX"],
                document["node_id"],
            )
            appendix["node_id"] = f"app_{len(document['nodes']):04d}"
            mapped = map_cloud_subtree(item, appendix, counter)
            if mapped:
                appendix["nodes"].append(mapped)
            document["nodes"].append(appendix)
            continue

        ch_key = "main"
        ch_title = "Document Body"
        m = RE_CHAPTER.match(title_guess)
        if m:
            ch_key = m.group(1).upper()
            ch_title = title_guess or f"Chapter {ch_key}"
        chapter = _ensure_chapter(document, chapters, ch_key, ch_title)
        if m and not chapter["nodes"]:
            chapter["title"] = ch_title
        mapped = map_cloud_subtree(item, chapter, counter)
        if mapped:
            chapter["nodes"].append(mapped)

    document["nodes"].insert(0, fm)

    document["children"] = [c["node_id"] for c in document["nodes"]]
    root["children"] = [document["node_id"]]
    fm["children"] = [c["node_id"] for c in fm["nodes"]]
    for ch in chapters.values():
        ch["children"] = [c["node_id"] for c in ch.get("nodes") or []]

    rebuild_paths(root)
    finalize_children(root)
    return root
