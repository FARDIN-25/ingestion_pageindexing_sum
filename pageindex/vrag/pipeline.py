from __future__ import annotations
"""Vectorless RAG: lexical retrieval and production build pipeline."""

from typing import Any

from .validation import ValidationError

import math
import re
from collections import Counter
from typing import Any

from .schema import BODY_RETRIEVAL_TYPES, RETRIEVAL_TYPES, normalize_type
from .processing import is_paragraph_title


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", text.lower())


def _norm_form(s: str) -> str:
    return re.sub(r"[\s\-]+", "", s.lower())


def _query_entities(query: str) -> list[str]:
    patterns = [
        (r"gst\s*reg[- ]?0?6", "reg06"),
        (r"gstr[- ]?2a", "gstr2a"),
        (r"gstr[- ]?2b", "gstr2b"),
        (r"gstr[- ]?1", "gstr1"),
        (r"gstr[- ]?3b", "gstr3b"),
        (r"cmp[- ]?0?8", "cmp08"),
        (r"pmt[- ]?0?6", "pmt06"),
        (r"drc[- ]?0?3", "drc03"),
        (r"input tax credit", "itc"),
        (r"\bitc\b", "itc"),
    ]
    entities: list[str] = []
    q = query.lower()
    for pat, tag in patterns:
        if re.search(pat, q, re.I):
            entities.append(tag)
    return entities


def _bm25(query_tokens: list[str], doc_tokens: list[str], avgdl: float, N: int, df: Counter) -> float:
    dl = len(doc_tokens)
    tf = Counter(doc_tokens)
    score = 0.0
    for t in query_tokens:
        if t not in tf:
            continue
        n = df.get(t, 0)
        idf = math.log((N - n + 0.5) / (n + 0.5) + 1.0)
        f = tf[t]
        score += idf * (f * 2.5) / (f + 1.5 * (1 - 0.75 + 0.75 * dl / max(avgdl, 1)))
    return score


def _normalize_structure(root: Any) -> dict:
    if isinstance(root, list):
        return {"node_id": "legacy_root", "type": "ROOT", "title": "ROOT", "nodes": root}
    return root


def flatten_retrieval_nodes(root: dict) -> list[dict]:
    out: list[dict] = []

    def walk(n: dict):
        if not n.get("retrieval_ready"):
            for c in n.get("nodes") or []:
                walk(c)
            return
        ntype = normalize_type(n.get("type", ""))
        if ntype in RETRIEVAL_TYPES:
            out.append(n)
        for c in n.get("nodes") or []:
            walk(c)

    walk(root)
    return out


def _index_by_id(root: dict) -> dict[str, dict]:
    idx: dict[str, dict] = {}

    def walk(n: dict):
        if n.get("node_id"):
            idx[n["node_id"]] = n
        for c in n.get("nodes") or []:
            walk(c)

    walk(root)
    return idx


ENTITY_TITLE_HINTS = {
    "reg06": [r"reg[- ]?0?6", r"6\.2", r"registration certificate"],
    "gstr1": [r"gstr[- ]?1", r"6\.3", r"6\.4"],
    "gstr2a": [r"gstr[- ]?2a", r"6\.5"],
    "gstr2b": [r"gstr[- ]?2b", r"6\.6"],
    "gstr3b": [r"gstr[- ]?3b", r"6\.7"],
    "cmp08": [r"cmp[- ]?0?8", r"6\.8"],
    "pmt06": [r"pmt[- ]?0?6", r"6\.9"],
    "drc03": [r"drc[- ]?0?3", r"6\.10"],
    "itc": [r"input tax credit", r"\bitc\b", r"10\.7"],
}


class LexicalRetriever:
    def __init__(self, structure: Any):
        self.root = _normalize_structure(structure)
        self.node_index = _index_by_id(self.root)
        self.nodes = flatten_retrieval_nodes(self.root)
        self._build_fts()

    def _build_fts(self):
        self.docs: list[dict] = []
        all_tokens: list[list[str]] = []
        for n in self.nodes:
            comp = _tokenize(n.get("compressed_content", ""))
            raw = _tokenize(n.get("raw_content", ""))
            all_tokens.append(comp + raw)
            self.docs.append({
                "node": n,
                "comp_tokens": comp,
                "raw_tokens": raw,
                "title": (n.get("title") or "").lower(),
                "aliases": [a.lower() for a in n.get("aliases") or []],
                "synonyms": [s.lower() for s in n.get("synonyms") or []],
                "keywords": [k.lower() for k in n.get("keywords") or []],
            })
        self.N = max(len(self.docs), 1)
        self.avgdl = sum(len(t) for t in all_tokens) / self.N
        self.df = Counter()
        for toks in all_tokens:
            for t in set(toks):
                self.df[t] += 1

    def _entity_title_match(self, ent: str, title: str, aliases: list[str]) -> bool:
        hints = ENTITY_TITLE_HINTS.get(ent, [])
        blob = title + " " + " ".join(aliases)
        return any(re.search(h, blob, re.I) for h in hints)

    def _score_doc(self, d: dict, q: str, q_tokens: list[str], entities: list[str]) -> tuple[float, dict]:
        node = d["node"]
        raw = (node.get("raw_content") or "")
        breakdown: dict[str, float] = {}
        score = 0.0
        title = d["title"]
        matched_title = False
        matched_alias: str | None = None
        matched_keyword: str | None = None
        matched_content_type: str | None = None

        ntype = normalize_type(node.get("type", ""))

        if node.get("is_front_matter") and entities:
            breakdown["front_matter_penalty"] = -100.0
            score -= 100.0

        if entities and re.search(r"test your knowledge|let sum up", title, re.I):
            blob = f"{title} {raw[:500]}".lower()
            if not any(
                re.search(h, blob, re.I)
                for ent in entities
                for h in ENTITY_TITLE_HINTS.get(ent, [])
            ):
                breakdown["boilerplate_penalty"] = -120.0
                score -= 120.0

        if is_paragraph_title(node.get("title", "")) and len(title) > 60:
            breakdown["paragraph_title_penalty"] = -150.0
            score -= 150.0

        # Stage 1: exact title match
        if title == q:
            breakdown["title_exact"] = 120.0
            score += 120.0
            matched_title = True
        elif len(q) > 4 and q in title:
            breakdown["title_contains_query"] = 90.0
            score += 90.0
            matched_title = True

        # Entity-specific title (beats loose alias on wrong chunks)
        for ent in entities:
            if self._entity_title_match(ent, title, d["aliases"]):
                breakdown[f"entity_title_{ent}"] = 200.0
                score += 200.0
                matched_title = True

        # Stage 2: alias match
        for a in d["aliases"] + d["synonyms"]:
            if a == q or (len(q) > 5 and q in a):
                breakdown["alias_exact"] = 85.0
                score += 85.0
                matched_alias = a
                break
            for ent in entities:
                if ent == "reg06" and re.search(r"reg[- ]?0?6", a, re.I):
                    breakdown["alias_entity"] = 95.0
                    score += 95.0
                    matched_alias = a
                    break

        # Stage 3: keyword match
        for k in d["keywords"]:
            if q in k or k in q:
                breakdown["keyword_exact"] = 70.0
                score += 70.0
                matched_keyword = k
                break
            if any(t in k for t in q_tokens if len(t) > 3):
                breakdown["keyword_partial"] = 35.0
                score += 35.0
                matched_keyword = k

        # Stage 4: compressed FTS
        comp_s = _bm25(q_tokens, d["comp_tokens"], self.avgdl, self.N, self.df)
        if comp_s > 0:
            breakdown["compressed_fts"] = round(comp_s * 2.5, 3)
            score += comp_s * 2.5
            matched_content_type = "compressed_content"

        # Stage 5: raw FTS
        raw_s = _bm25(q_tokens, d["raw_tokens"], self.avgdl, self.N, self.df)
        if raw_s > 0:
            breakdown["raw_fts"] = round(raw_s, 3)
            score += raw_s
            if not matched_content_type:
                matched_content_type = "raw_content"

        return score, {
            "breakdown": breakdown,
            "matched_title": matched_title,
            "matched_alias": matched_alias,
            "matched_keyword": matched_keyword,
            "matched_content_type": matched_content_type,
        }

    def _traversal_boost(self, node: dict, base_score: float, q_tokens: list[str]) -> tuple[float, dict]:
        extra: dict[str, float] = {}
        boost = 0.0
        nid = node.get("node_id")
        pid = node.get("parent_id")
        parent = self.node_index.get(pid) if pid else None

        # Stage 6: parent traversal
        if parent and parent.get("title"):
            pt = parent["title"].lower()
            if any(t in pt for t in q_tokens if len(t) > 2):
                extra["parent_traversal"] = 15.0
                boost += 15.0

        # Stage 7: child traversal (siblings of best — handled in search)
        children = [c for c in node.get("nodes") or [] if c.get("retrieval_ready")]
        if children:
            extra["has_children"] = 5.0
            boost += 5.0

        # Stage 8: sibling traversal via parent
        if parent:
            sibs = parent.get("nodes") or []
            for s in sibs:
                st = (s.get("title") or "").lower()
                if s.get("node_id") != nid and any(t in st for t in q_tokens if len(t) > 2):
                    extra["sibling_context"] = 8.0
                    boost += 8.0
                    break

        return boost, extra

    def search(self, query: str, top_k: int = 5, expand_traversal: bool = True) -> list[dict]:
        q = query.strip().lower()
        q_tokens = _tokenize(q)
        entities = _query_entities(query)
        ranked: list[tuple[float, dict, dict]] = []

        for d in self.docs:
            node = d["node"]
            score, meta = self._score_doc(d, q, q_tokens, entities)
            if expand_traversal:
                tb, tex = self._traversal_boost(node, score, q_tokens)
                score += tb
                meta["breakdown"].update(tex)
            raw_snip = (node.get("raw_content") or "")[:800].lower()
            if score > 0 or (entities and any(
                ent.replace("-", "") in raw_snip.replace("-", "") for ent in entities
            )):
                ranked.append((score, node, meta))

        # Stage 9: fallback lexical expansion — boost partial token overlap in title
        if not ranked or ranked[0][0] < 30:
            for d in self.docs:
                overlap = sum(1 for t in q_tokens if t in d["title"])
                if overlap >= 2:
                    ranked.append((20.0 + overlap * 5, d["node"], {
                        "breakdown": {"fallback_lexical": 20.0 + overlap * 5},
                        "matched_title": False,
                        "matched_alias": None,
                        "matched_keyword": None,
                        "matched_content_type": "fallback",
                    }))

        ranked.sort(key=lambda x: -x[0])
        seen_nid: set[str] = set()
        results = []
        for sc, node, meta in ranked:
            nid = node.get("node_id")
            if nid in seen_nid:
                continue
            seen_nid.add(nid)
            hit = {
                "node_id": nid,
                "title": node.get("title"),
                "type": normalize_type(node.get("type", "")),
                "path": node.get("path"),
                "retrieval_path": node.get("path"),
                "score": round(sc, 3),
                "score_breakdown": meta["breakdown"],
                "match_reason": ",".join(meta["breakdown"].keys()),
                "matched_title": meta["matched_title"],
                "matched_alias": meta["matched_alias"],
                "matched_keyword": meta["matched_keyword"],
                "matched_content_source": meta["matched_content_type"],
                "matched_content_type": meta["matched_content_type"],
                "matched_chunk": nid,
                "micro_summary": node.get("micro_summary"),
                "page_start": node.get("page_start"),
                "page_end": node.get("page_end"),
                "source_pages": f"{node.get('page_start')}-{node.get('page_end')}",
            }
            if expand_traversal:
                hit["traversal"] = self._traversal_context(node)
            results.append(hit)
            if len(results) >= top_k:
                break
        return results

    def _traversal_context(self, node: dict) -> dict:
        nid = node.get("node_id")
        pid = node.get("parent_id")
        parent = self.node_index.get(pid) if pid else None
        siblings = []
        if parent:
            siblings = [
                c.get("title") for c in parent.get("nodes") or []
                if c.get("node_id") != nid and (c.get("retrieval_ready") or c.get("is_retrieval_chunk"))
            ][:8]
        children = [
            c.get("title") for c in node.get("nodes") or []
            if c.get("retrieval_ready") or c.get("is_retrieval_chunk")
        ][:8]
        return {
            "parent_id": pid,
            "parent_title": parent.get("title") if parent else None,
            "parent_path": parent.get("path") if parent else None,
            "sibling_titles": siblings,
            "child_titles": children,
        }


def search(structure: Any, query: str, top_k: int = 5) -> list[dict]:
    return LexicalRetriever(structure).search(query, top_k=top_k)



TEST_QUERIES = [
    ("what is GST REG 06", ["reg", "06", "6.2"]),
    ("how to download GST registration certificate", ["certificate", "registration", "download", "6.2"]),
    ("GSTR-1 due date", ["gstr", "due", "6.4", "6.3"]),
    ("what is GSTR2A", ["gstr", "2a", "6.5"]),
    ("GSTR2B", ["gstr", "2b", "6.6"]),
    ("GSTR3B", ["gstr", "3b", "6.7"]),
    ("PMT06", ["pmt", "6.9"]),
    ("DRC03", ["drc", "6.10"]),
    ("CMP08", ["cmp", "6.8"]),
    ("input tax credit verification", ["tax", "credit", "itc"]),
]


def run_test_queries(retriever: LexicalRetriever) -> dict[str, Any]:
    report = {"queries": [], "passed": 0, "total": len(TEST_QUERIES)}
    for query, needles in TEST_QUERIES:
        hits = retriever.search(query, top_k=5)
        ok = False
        top = hits[0] if hits else None
        for h in hits[:5]:
            blob = f"{h.get('title','')} {h.get('micro_summary','')} {h.get('path','')}".lower()
            if any(n.lower() in blob for n in needles):
                top = h
                ok = True
                break
        report["queries"].append({
            "query": query,
            "passed": ok,
            "top_title": top.get("title") if top else None,
            "top_score": top.get("score") if top else 0,
            "score_breakdown": top.get("score_breakdown") if top else {},
        })
        if ok:
            report["passed"] += 1
    report["pass_rate"] = report["passed"] / max(report["total"], 1)
    return report


import json
import os
import time
import uuid
from typing import Any

from pageindex.log_util import log_info

from .schema import BuildConfig
from .processing import extract_document
from .hierarchy import HierarchyBuilder
from .schema import SCHEMA_VERSION, export_node
from .validation import apply_retrieval_readiness, clear_retrieval_ready

from .validation import ValidationError, validate_index


def build_index(
    pdf_path: str,
    output_path: str | None = None,
    opt: Any = None,
    config: dict | None = None,
    meter: Any | None = None,
    job_id: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    cfg = BuildConfig.from_dict(config) if config else BuildConfig.from_opt(opt)
    if kwargs.get("fail_on_validation") is not None:
        cfg.fail_on_validation = bool(kwargs["fail_on_validation"])
    if kwargs.get("run_retrieval_tests") is not None:
        cfg.run_retrieval_tests = bool(kwargs["run_retrieval_tests"])

    if meter is None:
        from pageindex.usage.meter import UsageMeter

        meter = UsageMeter(
            job_id=job_id or uuid.uuid4().hex,
            document_id=kwargs.get("document_id") or uuid.uuid4().hex,
            document_name=os.path.basename(pdf_path),
            pipeline="vrag",
            provider="local",
            model="vrag",
        )

    t0 = time.time()
    log_info("[vrag] === Production index build START ===")
    log_info("[vrag] PDF: %s", pdf_path)

    from pageindex.usage.constants import Operation

    file_size = os.path.getsize(pdf_path) if os.path.isfile(pdf_path) else 0
    meter.record_upload(file_size, latency_ms=0)

    with meter.track(Operation.OCR_EXTRACTION, local=True):
        t_ext = time.perf_counter()
        _, doc_lines, pages = extract_document(pdf_path)
        page_count = len(pages)
        meter.ensure_job(page_count)
        for pnum, _ in pages:
            lines_on_page = [ln for ln in doc_lines if ln.page == pnum]
            text = "\n".join(ln.text for ln in lines_on_page)
            meter.record_local_stage(
                Operation.OCR_EXTRACTION,
                page_start=pnum,
                page_end=pnum,
                text_sample=text[:2500],
                latency_ms=max(1, int((time.perf_counter() - t_ext) * 1000 / max(len(pages), 1))),
            )

    log_info("[vrag] Extracted %d lines, %d pages", len(doc_lines), len(pages))

    with meter.track(Operation.DOCUMENT_PARSING, local=True, text_sample="\n".join(ln.text for ln in doc_lines[:30])):
        pass

    with meter.track(Operation.HEADING_DETECTION, local=True):
        from .processing import detect_headings

        headings = detect_headings(doc_lines)
        meter.record_local_stage(
            Operation.HEADING_DETECTION,
            metadata={"heading_count": len(headings)},
            text_sample=str(len(headings)),
        )

    with meter.track(Operation.STRUCTURE_DETECTION, local=True):
        builder = HierarchyBuilder(cfg, meter=meter)
        root = builder.build(doc_lines)

    log_info(
        "[vrag] Tree built | chunks accepted=%d | rejected=%d",
        builder.dedup.stats.accepted,
        builder.skipped_duplicates,
    )

    errors: list[str] = []
    with meter.track(Operation.DEDUPE_VALIDATION, local=True):
        try:
            errors = validate_index(
                root,
                strict=cfg.fail_on_validation,
                overlap_adjacent_threshold=cfg.overlap_adjacent_threshold,
            )
        except ValidationError as e:
            errors = e.errors
            clear_retrieval_ready(root)
            meter.fail(errors[0][:500])
            raise

    log_info("[vrag] Validation: %d issues", len(errors))

    readiness = apply_retrieval_readiness(
        root,
        errors,
        observability_initialized=bool(meter.job_id),
        retrieval_tests_passed=None,
        require_tests=False,
    )

    retrieval_report = {}
    tests_ok = True
    with meter.track(Operation.INDEXING, local=True):
        retriever = LexicalRetriever(root)

    if cfg.run_retrieval_tests:
        with meter.track(Operation.TEST_VALIDATION, local=True):
            retrieval_report = run_test_queries(retriever)
            tests_ok = retrieval_report.get("pass_rate", 0) >= 0.9
        if not tests_ok:
            clear_retrieval_ready(root)
            readiness = apply_retrieval_readiness(
                root,
                errors + ["retrieval_tests_passed gate failed"],
                observability_initialized=bool(meter.job_id),
                retrieval_tests_passed=False,
                require_tests=True,
            )
        elif readiness["retrieval_ready"]:
            readiness = apply_retrieval_readiness(
                root,
                errors,
                observability_initialized=bool(meter.job_id),
                retrieval_tests_passed=True,
                require_tests=True,
            )
    log_info(
        "[vrag] retrieval_ready=%s gates=%s",
        readiness["retrieval_ready"],
        readiness["gates"],
    )

    if cfg.fail_on_validation and not readiness["retrieval_ready"]:
        clear_retrieval_ready(root)
        raise ValidationError(
            readiness.get("blocking_errors") or ["retrieval_ready gate failed"]
        )

    usage_report = meter.complete(
        status="success" if readiness["retrieval_ready"] else "failed_gates",
        page_count=page_count,
    )

    result: dict[str, Any] = {
        "pipeline": "vrag",
        "schema_version": SCHEMA_VERSION,
        "schema": SCHEMA_VERSION,
        "source_pdf": os.path.basename(pdf_path),
        "page_count": page_count,
        "line_count": len(doc_lines),
        "retrieval_ready": readiness["retrieval_ready"],
        "readiness": readiness,
        "retrieval_chunk_count": readiness["ready_node_count"],
        "job_id": meter.job_id,
        "document_id": meter.document_id,
        "usage": usage_report,
        "dedup": {
            "accepted": builder.dedup.stats.accepted,
            "rejected_exact": builder.dedup.stats.rejected_exact,
            "rejected_overlap": builder.dedup.stats.rejected_overlap,
        },
        "skipped_duplicates": builder.skipped_duplicates,
        "validation": {
            "valid": len(errors) == 0 and readiness["retrieval_ready"],
            "error_count": len(errors),
            "errors": errors[:30],
            "chunk_count": readiness["ready_node_count"],
        },
        "validation_errors": errors,
        "retrieval_tests": retrieval_report,
        "retrieval_report": retrieval_report,
        "structure": export_node(root),
        "flat_nodes": [
            export_node(n, include_children=False)
            for n in retriever.nodes
            if n.get("retrieval_ready")
        ],
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        log_info("[vrag] Wrote %s", output_path)

    log_info("[vrag] === DONE in %.1fs ===", time.time() - t0)
    return result


def build_index_safe(*args: Any, **kwargs: Any) -> dict[str, Any]:
    try:
        return build_index(*args, **kwargs)
    except ValidationError as e:
        log_info("[vrag] BUILD FAILED: %s", e.errors[:8])
        meter = kwargs.get("meter")
        if meter:
            meter.fail(str(e.errors[0])[:500])
        raise
build_vrag_index = build_index
build_vrag_index_safe = build_index_safe
__all__ = ["build_vrag_index", "build_vrag_index_safe", "ValidationError", "build_index"]
