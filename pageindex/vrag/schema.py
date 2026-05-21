from __future__ import annotations
# --- config.py ---
"""Build-time configuration for vectorless RAG pipeline."""

from dataclasses import dataclass
from typing import Any


@dataclass
class BuildConfig:
    compression_target_ratio: float = 0.70
    compression_min_ratio: float = 0.60
    overlap_reject_threshold: float = 0.85
    overlap_adjacent_threshold: float = 0.15
    fail_on_validation: bool = True
    run_retrieval_tests: bool = True
    min_chunk_chars: int = 25
    max_chunk_chars: int = 6000
    max_title_chars: int = 85
    domain: str = "generic"

    @classmethod
    def from_dict(cls, cfg: dict[str, Any] | None) -> "BuildConfig":
        if not cfg:
            return cls()
        return cls(
            compression_target_ratio=float(cfg.get("compression_target_ratio", 0.70)),
            compression_min_ratio=float(cfg.get("compression_min_ratio", 0.60)),
            overlap_reject_threshold=float(cfg.get("overlap_reject_threshold", 0.85)),
            overlap_adjacent_threshold=float(cfg.get("overlap_adjacent_threshold", 0.15)),
            fail_on_validation=str(cfg.get("fail_on_validation", "yes")).lower() in ("yes", "true", "1"),
            run_retrieval_tests=str(cfg.get("run_retrieval_tests", "yes")).lower() in ("yes", "true", "1"),
            min_chunk_chars=int(cfg.get("min_chunk_chars", 25)),
            max_chunk_chars=int(cfg.get("max_chunk_chars", 6000)),
            max_title_chars=int(cfg.get("max_title_chars", 85)),
            domain=str(cfg.get("domain", "generic")),
        )

    @classmethod
    def from_opt(cls, opt: Any) -> "BuildConfig":
        if opt is None:
            return cls()
        if isinstance(opt, dict):
            return cls.from_dict(opt)
        return cls.from_dict({k: v for k, v in vars(opt).items() if not k.startswith("_")})


# --- models.py ---
"""Production vectorless RAG index schema (v2.2)."""

from typing import Any

SCHEMA_VERSION = "2.3"


NODE_TYPES = frozenset({
    "ROOT", "FRONT_MATTER",
    "PREFACE", "COURSE_INFO", "OBJECTIVES", "SYLLABUS", "REFERENCES",
    "SECTION", "UNIT", "TOPIC", "SUBTOPIC", "CONTENT",
})

LEVEL = {
    "ROOT": 0,
    "FRONT_MATTER": 1,
    "PREFACE": 2,
    "COURSE_INFO": 2,
    "OBJECTIVES": 2,
    "SYLLABUS": 2,
    "REFERENCES": 2,
    "SECTION": 2,
    "UNIT": 3,
    "TOPIC": 4,
    "SUBTOPIC": 5,
    "CONTENT": 6,
}

VALID_PARENT: dict[str, frozenset[str]] = {
    "ROOT": frozenset({"FRONT_MATTER", "SECTION"}),
    "FRONT_MATTER": frozenset({
        "PREFACE", "COURSE_INFO", "OBJECTIVES", "SYLLABUS", "REFERENCES", "CONTENT",
    }),
    "SECTION": frozenset({"UNIT", "TOPIC", "SUBTOPIC", "CONTENT"}),
    "UNIT": frozenset({"TOPIC", "SUBTOPIC", "CONTENT"}),
    "TOPIC": frozenset({"SUBTOPIC", "CONTENT"}),
    "SUBTOPIC": frozenset({"CONTENT"}),
}

CONTAINER_TYPES = frozenset({"ROOT", "FRONT_MATTER", "SECTION", "UNIT", "TOPIC", "SUBTOPIC"})
FRONT_MATTER_TYPES = frozenset({"PREFACE", "COURSE_INFO", "OBJECTIVES", "SYLLABUS", "REFERENCES"})
RETRIEVAL_TYPES = frozenset({"CONTENT", "PREFACE", "OBJECTIVES", "SYLLABUS", "REFERENCES"})
BODY_RETRIEVAL_TYPES = frozenset({"CONTENT"})

# Legacy alias map
_TYPE_ALIASES = {"SYLLABUS_OVERVIEW": "SYLLABUS", "CONTENT_NODE": "CONTENT"}

NODE_FIELDS = [
    "node_id", "parent_id", "children", "type", "level", "title", "path",
    "raw_content", "compressed_content", "micro_summary",
    "aliases", "keywords", "synonyms",
    "page_start", "page_end", "char_start", "char_end",
    "token_count_raw", "token_count_compressed", "content_hash",
    "retrieval_ready",
]


def normalize_type(node_type: str) -> str:
    return _TYPE_ALIASES.get(node_type, node_type)


def empty_node(
    title: str,
    node_type: str,
    path: str,
    level: int,
    parent_id: str | None = None,
) -> dict[str, Any]:
    ntype = normalize_type(node_type)
    return {
        "node_id": "",
        "parent_id": parent_id,
        "children": [],
        "type": ntype,
        "level": level,
        "title": title,
        "path": path,
        "raw_content": "",
        "compressed_content": "",
        "micro_summary": "",
        "aliases": [],
        "keywords": [],
        "synonyms": [],
        "page_start": 0,
        "page_end": 0,
        "char_start": 0,
        "char_end": 0,
        "token_count_raw": 0,
        "token_count_compressed": 0,
        "content_hash": "",
        "retrieval_ready": False,
        "nodes": [],
        # Legacy fields for PostgreSQL export
        "children_ids": [],
        "is_retrieval_chunk": False,
        "is_front_matter": ntype in FRONT_MATTER_TYPES,
    }


def export_node(node: dict[str, Any], include_children: bool = True) -> dict[str, Any]:
    children = node.get("nodes") or []
    out: dict[str, Any] = {}
    # Explicit 3-content schema only — no summary/prefix_summary/text
    node.pop("text", None)
    node.pop("summary", None)
    node.pop("prefix_summary", None)
    for k in NODE_FIELDS:
        if k == "children":
            out[k] = [c["node_id"] for c in children if c.get("node_id")]
        elif k in ("aliases", "keywords", "synonyms"):
            out[k] = list(node.get(k) or [])
        elif k == "retrieval_ready":
            out[k] = bool(node.get("retrieval_ready") or node.get("is_retrieval_chunk"))
        else:
            default = "" if k not in (
                "level", "page_start", "page_end", "char_start", "char_end",
                "token_count_raw", "token_count_compressed",
            ) else 0
            out[k] = node.get(k, default)
    out["children_ids"] = out["children"]
    out["is_retrieval_chunk"] = out["retrieval_ready"]
    out["is_front_matter"] = node.get("type") in FRONT_MATTER_TYPES
    if include_children and children:
        out["nodes"] = [export_node(c) for c in children]
    return out


def finalize_children(node: dict[str, Any]) -> None:
    children = node.get("nodes") or []
    ids = [c["node_id"] for c in children if c.get("node_id")]
    node["children"] = ids
    node["children_ids"] = ids
    for c in children:
        finalize_children(c)


# --- metadata.py ---
"""Lexical retrieval metadata: aliases, keywords, synonyms, content_hash."""

import re


FORM_PATTERNS = [
    (re.compile(r"GST\s*REG[- ]?0?6", re.I), [
        "FORM GST REG-06", "GST REG 06", "GST REG-06",
        "GST registration certificate", "registration certificate download",
    ]),
    (re.compile(r"GSTR[- ]?1\b", re.I), ["GSTR-1", "GSTR 1", "GSTR1", "GSTR1 return"]),
    (re.compile(r"GSTR[- ]?2A", re.I), ["GSTR2A", "GSTR-2A", "GSTR 2A"]),
    (re.compile(r"GSTR[- ]?2B", re.I), ["GSTR2B", "GSTR-2B", "GSTR 2B"]),
    (re.compile(r"GSTR[- ]?3B", re.I), ["GSTR-3B", "GSTR 3B", "GSTR3B"]),
    (re.compile(r"GST\s*CMP[- ]?0?8", re.I), ["GST CMP-08", "CMP08", "CMP-08"]),
    (re.compile(r"GST\s*PMT[- ]?0?6", re.I), ["GST PMT-06", "PMT06", "PMT-06", "payment challan"]),
    (re.compile(r"DRC[- ]?0?3", re.I), ["DRC03", "DRC-03", "demand recovery"]),
    (re.compile(r"input tax credit|\bitc\b", re.I), ["input tax credit", "ITC", "ITC verification"]),
]


def token_count(text: str) -> int:
    return len(text.split()) if text else 0


def content_hash(text: str) -> str:
    from .processing import sha256_content
    return sha256_content(text)


def extract_aliases(title: str, raw: str) -> list[str]:
    aliases: list[str] = []
    seen: set[str] = set()
    blob = f"{title} {raw[:2500]}"

    def add(a: str):
        a = re.sub(r"\s+", " ", a.strip())
        if a and len(a) > 2 and a.lower() not in seen:
            seen.add(a.lower())
            aliases.append(a)

    add(title)
    for pat, forms in FORM_PATTERNS:
        if pat.search(blob):
            for f in forms:
                add(f)

    m = re.match(r"^(\d+\.\d+(?:\.\d+)?)\s+(.+)$", title)
    if m:
        add(m.group(1))
        short = m.group(2)[:60].strip()
        if short and not re.search(r"\b(the|for the|according)\b", short, re.I):
            add(f"{m.group(1)} {short}")

    return aliases[:14]


def extract_keywords(title: str, raw: str, compressed: str) -> list[str]:
    kws: list[str] = []
    seen: set[str] = set()
    blob = f"{title} {compressed[:2000]} {raw[:1200]}".lower()

    def add(k: str):
        k = k.strip()
        if k and k not in seen and len(k) > 3:
            seen.add(k)
            kws.append(k)

    if "download" in blob and ("certificate" in blob or "reg" in blob):
        add("download GST registration certificate")
        add("GST portal certificate download")
    if "due date" in blob or "when filing" in blob:
        add("GSTR-1 due date")
        add("return filing due date")
    if re.search(r"gstr[- ]?1", blob):
        add("GSTR-1 return filing")
    if re.search(r"gstr[- ]?2a", blob):
        add("GSTR-2A reconciliation")
        add("GSTR2A auto-drafted return")
    if re.search(r"gstr[- ]?2b", blob):
        add("GSTR-2B matching")
    if re.search(r"gstr[- ]?3b", blob):
        add("GSTR-3B monthly return")
    if re.search(r"reg[- ]?0?6", blob):
        add("GST REG 06")
        add("registration certificate form")
    if re.search(r"pmt[- ]?0?6", blob):
        add("GST PMT-06 payment")
    if re.search(r"cmp[- ]?0?8", blob):
        add("GST CMP-08 composition return")
    if re.search(r"drc[- ]?0?3", blob):
        add("DRC-03 demand payment")
    if re.search(r"input tax credit|\bitc\b", blob):
        add("input tax credit verification")
        add("ITC matching verification")

    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]{2,}", title)
    if len(words) >= 2 and len(title) < 60:
        add(" ".join(words[:5]).lower())

    if not kws and title:
        add(title.lower()[:80])
    return kws[:16]


def _extract_synonyms(aliases: list[str], title: str) -> list[str]:
    syn: list[str] = []
    seen: set[str] = set()

    def add(s: str):
        k = s.lower().strip()
        if k and k not in seen and len(k) > 2:
            seen.add(k)
            syn.append(s)

    for a in aliases:
        add(a)
        add(re.sub(r"[-\s]+", " ", a))
        add(re.sub(r"[-\s]+", "", a))
    add(title)
    return syn[:18]


def enrich_node(node: dict) -> None:
    raw = node.get("raw_content") or ""
    if not raw:
        return
    node["content_hash"] = content_hash(raw)
    node["token_count_raw"] = token_count(raw)
    comp = node.get("compressed_content") or ""
    node["token_count_compressed"] = token_count(comp)
    aliases = extract_aliases(node.get("title", ""), raw)
    node["aliases"] = aliases
    node["keywords"] = extract_keywords(node.get("title", ""), raw, comp)
    syns = _extract_synonyms(aliases, node.get("title", ""))
    node["synonyms"] = syns if syns else list(aliases[:8])


# --- schema_normalizer.py ---
"""Normalize legacy cloud/LLM tree nodes to explicit 3-content schema."""

import re
from typing import Any



from .path_utils import rebuild_paths

LEGACY_CONTENT_KEYS = frozenset({"text", "summary", "prefix_summary", "content"})

RE_PREFACE = re.compile(r"^preface\b", re.I)
RE_SECTION = re.compile(r"^section\s+([a-z])", re.I)
RE_UNIT = re.compile(r"^unit\s+([ivxlc\d]+)", re.I)


def _infer_structural_title(node: dict) -> str:
    from .processing import is_synthetic_title, is_paragraph_title
    title = (node.get("title") or "").strip()
    if title and not is_synthetic_title(title) and not is_paragraph_title(title):
        return title[:85]
    raw = (node.get("raw_content") or node.get("text") or "").strip()
    for line in raw.split("\n")[:8]:
        line = line.strip()
        if not line or len(line) > 100:
            continue
        if re.match(r"^(?:SECTION|UNIT)\s+", line, re.I):
            return line[:85]
        if re.match(r"^\d+\.\d+(?:\.\d+)?\s+\S", line):
            return line[:85]
        if re.match(r"^(?:FORM\s+)?GST\s", line, re.I) and len(line) < 80:
            return line[:85]
    if re.match(r"^\d+\.\d+", title):
        return title[:85]
    return "Content"


def _page_span(node: dict) -> tuple[int, int, int, int]:
    ps = int(node.get("page_start") or node.get("page_index") or node.get("start_index") or 0)
    pe = int(node.get("page_end") or node.get("page_index") or node.get("end_index") or ps)
    cs = int(node.get("char_start") or 0)
    ce = int(node.get("char_end") or 0)
    if pe < ps and ps > 0:
        pe = ps
    return ps, pe, cs, ce


def _is_front_matter_title(title: str) -> bool:
    t = title.lower().strip()
    return bool(
        RE_PREFACE.match(t)
        or "syllabus" in t
        or "objective" in t
        or "course coordinator" in t
        or "suggested reading" in t
        or t in ("structure", "introduction", "table of contents")
    )


def _infer_cloud_node_type(node: dict, has_body: bool) -> str:
    declared = normalize_type(node.get("type") or "")
    title = (node.get("title") or "").strip()

    if RE_SECTION.match(title):
        return "SECTION"
    if RE_UNIT.match(title):
        return "UNIT"
    if RE_PREFACE.match(title):
        return "PREFACE"
    if _is_front_matter_title(title):
        return "PREFACE" if "preface" in title.lower() else "SYLLABUS"
    if re.match(r"^\d+\.\d+\.\d+", title):
        return "SUBTOPIC"
    if re.match(r"^\d+\.\d+", title):
        return "TOPIC"
    if has_body:
        return "CONTENT"
    if declared in ("SECTION", "UNIT", "TOPIC", "SUBTOPIC", "CONTENT", "PREFACE", "SYLLABUS"):
        return declared
    return "TOPIC"


def normalize_node(
    node: Any,
    parent: dict,
    counter: list[int],
) -> dict | None:
    from .processing import sanitize_raw_content, contains_garbage_artifact
    if not isinstance(node, dict):
        return None

    children_in = node.get("nodes") or []
    raw = (node.get("raw_content") or "").strip()
    if not raw:
        legacy_text = (node.get("text") or node.get("content") or "").strip()
        if legacy_text:
            raw = sanitize_raw_content(legacy_text)

    has_body = len(raw) >= 20
    ntype = _infer_cloud_node_type(node, has_body)
    if has_body and ntype in ("TOPIC", "SUBTOPIC", "PREFACE", "SYLLABUS", "UNIT"):
        ntype = "CONTENT"

    title = _infer_structural_title(node)
    parent_path = parent.get("path") or "ROOT"
    path = f"{parent_path} > {title}"
    level = LEVEL.get(ntype, 4)

    out = empty_node(title, ntype, path, level, parent["node_id"])
    counter[0] += 1
    out["node_id"] = node.get("node_id") or f"cloud_{counter[0]:05d}"
    out["parent_id"] = parent["node_id"]

    ps, pe, cs, ce = _page_span(node)
    out["page_start"], out["page_end"] = ps, pe
    out["char_start"], out["char_end"] = cs, ce

    if has_body and not contains_garbage_artifact(raw):
        out["type"] = "CONTENT"
        out["level"] = LEVEL["CONTENT"]
        out["raw_content"] = raw
        legacy_summary = (node.get("summary") or node.get("prefix_summary") or "").strip()
        comp = (node.get("compressed_content") or "").strip()
        if not comp:
            from .processing import compress_text
            comp = compress_text(raw)
        out["compressed_content"] = comp
        micro = (node.get("micro_summary") or "").strip()
        if not micro or len(micro) > 400:
            from .processing import micro_summary_from_content
            micro = micro_summary_from_content(title, comp)
        elif len(legacy_summary) < 400:
            micro = legacy_summary[:400]
        out["micro_summary"] = micro
        enrich_node(out)
        out["retrieval_ready"] = False

    for ch in children_in:
        child = normalize_node(ch, out, counter)
        if child:
            out["nodes"].append(child)

    out["children"] = [c["node_id"] for c in out["nodes"]]
    out["children_ids"] = out["children"]
    return out


def normalize_cloud_structure(structure: Any) -> dict[str, Any]:
    """
    Convert PageIndex cloud tree to strict VRAG hierarchy:
    ROOT → FRONT_MATTER | SECTION → (UNIT | TOPIC | CONTENT)*

    Cloud API nodes are never attached directly under ROOT.
    """
    counter = [0]
    root = empty_node("ROOT", "ROOT", "ROOT", 0)
    root["node_id"] = "root_0001"

    fm = empty_node("Front Matter", "FRONT_MATTER", "ROOT/FRONT_MATTER", LEVEL["FRONT_MATTER"], root["node_id"])
    fm["node_id"] = "fm_0001"
    root["nodes"].append(fm)

    section = empty_node("Document", "SECTION", "ROOT/SECTION/DOCUMENT", LEVEL["SECTION"], root["node_id"])
    section["node_id"] = "sec_0001"
    root["nodes"].append(section)

    items: list[Any] = []
    if isinstance(structure, list):
        items = structure
    elif isinstance(structure, dict):
        if structure.get("nodes"):
            items = structure["nodes"]
        elif structure.get("title") or structure.get("text"):
            items = [structure]

    for item in items:
        if not isinstance(item, dict):
            continue
        title = (item.get("title") or "").strip()
        parent = fm if _is_front_matter_title(title) else section
        ch = normalize_node(item, parent, counter)
        if ch:
            parent["nodes"].append(ch)

    root["children"] = [c["node_id"] for c in root["nodes"]]
    fm["children"] = [c["node_id"] for c in fm["nodes"]]
    section["children"] = [c["node_id"] for c in section["nodes"]]
    rebuild_paths(root)
    return root


def strip_legacy_fields(node: dict[str, Any]) -> dict[str, Any]:
    for key in list(node.keys()):
        if key in LEGACY_CONTENT_KEYS:
            del node[key]
    if "nodes" in node:
        node["nodes"] = [strip_legacy_fields(c) for c in node["nodes"]]
    return node
