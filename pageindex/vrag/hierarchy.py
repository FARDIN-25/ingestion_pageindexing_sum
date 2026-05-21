from __future__ import annotations
"""Production hierarchy builder — strict tree, CONTENT leaves only."""

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pageindex.usage.meter import UsageMeter

from .processing import semantic_chunks
from .processing import compress_text
from .schema import BuildConfig
from .processing import ContentDeduplicator
from .processing import DocLine
from .processing import clean_title, detect_headings
from .schema import enrich_node
from .schema import (
    CONTAINER_TYPES,
    FRONT_MATTER_TYPES,
    LEVEL,
    empty_node,
    finalize_children,
)
from .processing import container_micro_summary, micro_summary_from_content
from .path_utils import rebuild_paths
from .processing import is_paragraph_title, is_synthetic_title

ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10}


def _unit_number(meta: dict) -> int | None:
    u = (meta.get("unit") or "").upper().strip()
    if not u:
        return None
    if u.isdigit():
        return int(u)
    return ROMAN.get(u)


def _topic_unit_number(meta: dict) -> int | None:
    t = meta.get("topic") or ""
    m = re.match(r"^(\d+)", t)
    return int(m.group(1)) if m else None


def _new_id(prefix: str, counter: list[int]) -> str:
    counter[0] += 1
    return f"{prefix}_{counter[0]:04d}"


def _page_range(doc_lines: list[DocLine], start: int, end: int) -> tuple[int, int]:
    if start >= len(doc_lines) or end <= start:
        return (0, 0)
    pages = [doc_lines[i].page for i in range(start, min(end, len(doc_lines)))]
    return (min(pages), max(pages)) if pages else (0, 0)


def _structural_wrapper_type(title: str) -> str:
    if re.match(r"^\d+\.\d+\.\d+", title):
        return "SUBTOPIC"
    if re.match(r"^\d+\.\d+", title):
        return "TOPIC"
    return "TOPIC"


class HierarchyBuilder:
    def __init__(self, cfg: BuildConfig, meter: Any | None = None):
        self.cfg = cfg
        self._id = [0]
        self.dedup = ContentDeduplicator(cfg.overlap_reject_threshold)
        self.skipped_duplicates = 0
        self.meter = meter

    def build(self, doc_lines: list[DocLine]) -> dict[str, Any]:
        headings = detect_headings(doc_lines)
        if not headings:
            headings = [{"line_idx": 0, "type": "PREFACE", "title": "Document", "level": 2, "meta": {}}]

        ends = [headings[i + 1]["line_idx"] for i in range(len(headings) - 1)] + [len(doc_lines)]
        first_section = next((i for i, h in enumerate(headings) if h["type"] == "SECTION"), None)

        root = empty_node("ROOT", "ROOT", "ROOT", LEVEL["ROOT"])
        root["node_id"] = _new_id("root", self._id)

        fm = self._container("Front Matter", "FRONT_MATTER", "ROOT/FRONT_MATTER", root)
        root["nodes"].append(fm)

        section: dict | None = None
        unit: dict | None = None
        fm_seen: set[str] = set()

        for i, h in enumerate(headings):
            start, end = h["line_idx"], ends[i]
            ntype = h["type"]
            title = clean_title(h["title"], ntype)
            meta = h.get("meta") or {}
            in_body = first_section is not None and i >= first_section

            if ntype == "SECTION":
                section = self._container(title, "SECTION", f"ROOT/SECTION/{title[:24]}", root)
                root["nodes"].append(section)
                unit = None
                continue

            if not in_body:
                fm_type = ntype if ntype in FRONT_MATTER_TYPES else "PREFACE"
                if fm_type in fm_seen:
                    continue
                fm_seen.add(fm_type)
                self._attach_content_chunks(
                    fm, fm_type, f"{fm['path']}/{fm_type}", doc_lines, start, end, title,
                    front_matter=True,
                )
                continue

            if section is None:
                section = self._container("SECTION MAIN", "SECTION", "ROOT/SECTION/MAIN", root)
                root["nodes"].append(section)
                unit = None

            if ntype == "UNIT":
                unit = self._container(title, "UNIT", f"{section['path']}/UNIT/{title[:40]}", section, meta)
                section["nodes"].append(unit)
                continue

            parent = self._resolve_parent(section, unit, meta)
            self._attach_content_chunks(
                parent, _structural_wrapper_type(title), parent["path"],
                doc_lines, start, end, title,
            )

        if not section and not fm["nodes"]:
            self._attach_content_chunks(
                fm, "PREFACE", f"{fm['path']}/PREFACE", doc_lines, 0, len(doc_lines), "Document",
                front_matter=True,
            )

        self._finalize_tree(root)
        rebuild_paths(root)
        finalize_children(root)
        self.skipped_duplicates = (
            self.dedup.stats.rejected_exact
            + self.dedup.stats.rejected_overlap
            + self.dedup.stats.rejected_parent_child
        )
        return root

    def _container(
        self,
        title: str,
        ntype: str,
        path: str,
        parent: dict,
        meta: dict | None = None,
    ) -> dict:
        node = empty_node(title, ntype, path, LEVEL[ntype], parent["node_id"])
        node["node_id"] = _new_id(ntype[:3].lower(), self._id)
        node["parent_id"] = parent["node_id"]
        if meta:
            node["_meta"] = meta
        return node

    def _resolve_parent(self, section: dict | None, unit: dict | None, meta: dict) -> dict:
        topic_unit = _topic_unit_number(meta)
        if topic_unit is not None and section:
            for child in section.get("nodes") or []:
                if child.get("type") != "UNIT":
                    continue
                un = _unit_number(child.get("_meta") or {})
                if un == topic_unit:
                    return child
        if unit:
            return unit
        return section  # type: ignore

    def _attach_content_chunks(
        self,
        parent: dict,
        wrapper_hint: str,
        path_prefix: str,
        doc_lines: list[DocLine],
        start: int,
        end: int,
        default_title: str,
        front_matter: bool = False,
    ) -> None:
        for title, raw, s, e in semantic_chunks(
            doc_lines,
            start,
            end,
            default_title,
            max_chars=self.cfg.max_chunk_chars,
            overlap_adjacent_threshold=self.cfg.overlap_adjacent_threshold,
        ):
            if is_synthetic_title(title) and not re.match(r"^\d+\.\d+", title):
                continue
            if is_paragraph_title(title) and not re.match(r"^\d+\.\d+", title):
                continue
            if re.match(r"^[\d.,\s]+$", title) or len(re.findall(r"[A-Za-z]", title)) < 4:
                continue

            accepted, chash = self.dedup.register(raw)
            if not accepted:
                if self.meter:
                    from pageindex.usage.constants import Operation

                    ps = doc_lines[s].page if s < len(doc_lines) else 0
                    pe = doc_lines[min(e - 1, len(doc_lines) - 1)].page if e > 0 else ps
                    self.meter.record_local_stage(
                        Operation.DEDUPE_VALIDATION,
                        page_start=ps,
                        page_end=pe,
                        metadata={"rejected_duplicate": True},
                    )
                continue

            ps = doc_lines[s].page if s < len(doc_lines) else 0
            pe = doc_lines[min(e - 1, len(doc_lines) - 1)].page if e > 0 else ps
            if self.meter:
                from pageindex.usage.constants import Operation

                self.meter.record_local_stage(
                    Operation.CHUNK_GENERATION,
                    page_start=ps,
                    page_end=pe,
                    text_sample=raw[:2000],
                )

            wrap_type = wrapper_hint
            if re.match(r"^\d+\.\d+\.\d+", title):
                wrap_type = "SUBTOPIC"
            elif re.match(r"^\d+\.\d+", title):
                wrap_type = "TOPIC"

            # Structural wrapper (no raw content) when numbered topic under UNIT/SECTION
            attach_parent = parent
            if (
                not front_matter
                and parent.get("type") in ("UNIT", "SECTION")
                and wrap_type in ("TOPIC", "SUBTOPIC")
                and re.match(r"^\d+\.\d+", title)
            ):
                wrapper = empty_node(
                    clean_title(title, wrap_type)[: self.cfg.max_title_chars],
                    wrap_type,
                    f"{path_prefix}/{title[:50]}",
                    LEVEL[wrap_type],
                    parent["node_id"],
                )
                wrapper["node_id"] = _new_id("wrap", self._id)
                wrapper["parent_id"] = parent["node_id"]
                parent["nodes"].append(wrapper)
                attach_parent = wrapper

            # Check for sibling overlap to fix ISSUE 1
            sibling_merged = False
            for sib in attach_parent.setdefault("nodes", []):
                if sib.get("type") == "CONTENT":
                    sraw = sib.get("raw_content", "")
                    sec_key = re.match(r"^(\d+(?:\.\d+)*)", title.strip())
                    sib_sec = re.match(r"^(\d+(?:\.\d+)*)", sib.get("title", "").strip())
                    if sec_key and sib_sec and sec_key.group(1) == sib_sec.group(1):
                        from .processing import jaccard_similarity, sha256_content
                        if jaccard_similarity(sraw, raw) >= self.cfg.overlap_adjacent_threshold:
                            sib["raw_content"] = sraw + "\n" + raw
                            sib["content_hash"] = sha256_content(sib["raw_content"])
                            sib["compressed_content"] = compress_text(
                                sib["raw_content"],
                                target_ratio=self.cfg.compression_target_ratio,
                                min_ratio=self.cfg.compression_min_ratio,
                            )
                            sib["micro_summary"] = micro_summary_from_content(sib["title"], sib["compressed_content"])
                            pe_new = _page_range(doc_lines, s, e)[1]
                            sib["page_end"] = max(sib.get("page_end", 0), pe_new)
                            sib["char_end"] = max(sib.get("char_end", 0), e)
                            sibling_merged = True
                            break
            
            if sibling_merged:
                continue

            leaf = empty_node(
                clean_title(title, "CONTENT")[: self.cfg.max_title_chars],
                "CONTENT",
                f"{attach_parent['path']}/CONTENT/{title[:50]}",
                LEVEL["CONTENT"],
                attach_parent["node_id"],
            )
            leaf["node_id"] = _new_id("content", self._id)
            leaf["parent_id"] = attach_parent["node_id"]
            leaf["raw_content"] = raw
            leaf["content_hash"] = chash
            import time as _time

            t_comp = _time.perf_counter()
            leaf["compressed_content"] = compress_text(
                raw,
                target_ratio=self.cfg.compression_target_ratio,
                min_ratio=self.cfg.compression_min_ratio,
            )
            if self.meter:
                from pageindex.usage.constants import Operation

                self.meter.record_local_stage(
                    Operation.COMPRESSION,
                    page_start=ps,
                    page_end=pe,
                    text_sample=leaf["compressed_content"][:2000],
                    latency_ms=int((_time.perf_counter() - t_comp) * 1000),
                )
            leaf["micro_summary"] = micro_summary_from_content(leaf["title"], leaf["compressed_content"])
            if self.meter:
                from pageindex.usage.constants import Operation

                self.meter.record_local_stage(
                    Operation.MICRO_SUMMARY,
                    page_start=ps,
                    page_end=pe,
                    text_sample=leaf["micro_summary"],
                )
            leaf["page_start"], leaf["page_end"] = _page_range(doc_lines, s, e)
            leaf["char_start"], leaf["char_end"] = s, e
            leaf["retrieval_ready"] = False
            leaf["is_retrieval_chunk"] = False
            leaf["is_front_matter"] = front_matter or attach_parent.get("type") == "FRONT_MATTER"
            enrich_node(leaf)
            if self.meter:
                from pageindex.usage.constants import Operation

                self.meter.record_local_stage(
                    Operation.ALIAS_GENERATION,
                    page_start=leaf["page_start"],
                    page_end=leaf["page_end"],
                    text_sample=" ".join(leaf.get("aliases") or [])[:500],
                )
                self.meter.record_local_stage(
                    Operation.KEYWORD_GENERATION,
                    page_start=leaf["page_start"],
                    page_end=leaf["page_end"],
                    text_sample=" ".join(leaf.get("keywords") or [])[:500],
                )
            attach_parent["nodes"].append(leaf)

    def _finalize_tree(self, node: dict) -> None:
        if node.get("_meta") is not None:
            del node["_meta"]
        if node["type"] in CONTAINER_TYPES:
            node["raw_content"] = ""
            node["compressed_content"] = ""
            node["content_hash"] = ""
            node["retrieval_ready"] = False
            node["is_retrieval_chunk"] = False
            titles = [c.get("title", "") for c in node.get("nodes") or []]
            node["micro_summary"] = container_micro_summary(node.get("title", ""), titles)
        for c in node.get("nodes") or []:
            pr = (node.get("raw_content") or "").strip()
            cr = (c.get("raw_content") or "").strip()
            if pr and cr and self.dedup.child_duplicates_parent(pr, cr):
                self.dedup.stats.rejected_parent_child += 1
                c["raw_content"] = ""
                c["retrieval_ready"] = False
            self._finalize_tree(c)
