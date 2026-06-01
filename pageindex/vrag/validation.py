from __future__ import annotations
"""Production index validation and retrieval readiness gating."""

import re
from typing import Any

from .schema import (
    CONTAINER_TYPES,
    FORBIDDEN_EXPORT_FIELDS,
    NODE_TYPES,
    RETRIEVAL_TYPES,
    VALID_PARENT,
    _split_micro_sentences,
    normalize_type,
)
from .processing import (
    compression_ratio,
    contains_garbage_artifact,
    is_legal_section_reference_title,
    is_paragraph_title,
    is_synthetic_title,
    jaccard_similarity,
    sha256_content,
    token_set,
)

MULTI_TOPIC_PATTERNS = [
    re.compile(r"gstr[- ]?1\b", re.I),
    re.compile(r"gstr[- ]?2a", re.I),
    re.compile(r"gstr[- ]?2b", re.I),
    re.compile(r"gstr[- ]?3b", re.I),
    re.compile(r"reg[- ]?0?6", re.I),
    re.compile(r"drc[- ]?0?3", re.I),
]

OVERLAP_DUPLICATE_THRESHOLD = 0.85
# Full pairwise checks on 1000-page docs can exhaust RAM (O(n²) × large raw_content).
MAX_GLOBAL_OVERLAP_NODES = 800
OVERLAP_SAMPLE_CHARS = 2000
MAX_GLOBAL_OVERLAP_ERRORS = 25


def _global_overlap_duplicate_errors(retrieval_nodes: list[dict], threshold: float) -> list[str]:
    """Near-duplicate detection using token samples (not full raw_content pairs)."""
    if len(retrieval_nodes) <= 1:
        return []
    if len(retrieval_nodes) > MAX_GLOBAL_OVERLAP_NODES:
        return []

    samples: list[tuple[dict, set[str]]] = []
    for node in retrieval_nodes:
        raw = (node.get("raw_content") or "")[:OVERLAP_SAMPLE_CHARS]
        if len(raw) < 40:
            continue
        toks = token_set(raw)
        if toks:
            samples.append((node, toks))

    errors: list[str] = []
    for i, (node_a, ta) in enumerate(samples):
        ha = node_a.get("content_hash")
        for node_b, tb in samples[i + 1 :]:
            if ha and ha == node_b.get("content_hash"):
                continue
            union = len(ta | tb)
            if union and len(ta & tb) / union >= threshold:
                errors.append(
                    f"Overlap duplicate:\n"
                    f"Node A: {node_a.get('path')} [{node_a.get('node_id')}]\n"
                    f"Node B: {node_b.get('path')} [{node_b.get('node_id')}]"
                )
                if len(errors) >= MAX_GLOBAL_OVERLAP_ERRORS:
                    return errors
    return errors


class ValidationError(Exception):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__(f"{len(errors)} validation error(s):\n" + "\n".join(errors[:3]))


def validate_index(
    root: dict[str, Any],
    strict: bool = True,
    *,
    overlap_adjacent_threshold: float = 0.15,
) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    all_ids: set[str] = set()
    content_hashes: dict[str, str] = {}
    retrieval_nodes: list[dict] = []
    parent_map: dict[str, str | None] = {}
    sibling_chunks: list[tuple[str, str, str]] = []

    def collect_ids(node: dict) -> None:
        if node.get("node_id"):
            all_ids.add(node["node_id"])
        for c in node.get("nodes") or []:
            collect_ids(c)

    collect_ids(root)

    def walk(node: dict, parent: dict | None, path_parts: list[str]):
        nid = node.get("node_id") or ""
        ntype = normalize_type(node.get("type", ""))
        title = (node.get("title") or "").strip()

        def _err(msg: str, subject_node: dict | None = None) -> str:
            n = subject_node or node
            n_path = n.get("path") or n.get("title") or "Unknown"
            n_id = n.get("node_id") or "?"
            n_type = normalize_type(n.get("type", ""))
            ps = n.get("page_start") or 0
            pe = n.get("page_end") or 0
            return f"{msg} | node_id={n_id} | type={n_type} | pages={ps}-{pe} | path={n_path}"

        if ntype not in NODE_TYPES:
            errors.append(_err(f"Unknown type: {ntype}"))

        if not nid:
            errors.append(_err("Missing node_id"))
        elif nid in seen_ids:
            errors.append(_err(f"Duplicate node_id: {nid}"))
        else:
            seen_ids.add(nid)

        pid = parent.get("node_id") if parent else None
        parent_map[nid] = pid
        if pid and pid not in all_ids:
            errors.append(_err(f"Orphan node: parent {pid} missing"))

        if not title and ntype != "ROOT":
            errors.append(_err("Empty title"))

        if is_synthetic_title(title) and ntype in ("TOPIC", "SUBTOPIC", "CONTENT"):
            errors.append(_err(f"Synthetic title (not structural): {title[:60]}"))

        if is_legal_section_reference_title(title) and ntype in ("TOPIC", "SUBTOPIC", "CONTENT"):
            errors.append(_err(f"Legal section prose used as title: {title[:60]}"))

        if ntype == "TABLE_OF_CONTENTS" and node.get("retrieval_ready"):
            errors.append(_err("Table of contents must not be retrieval_ready"))

        if parent:
            ptype = normalize_type(parent.get("type", ""))
            allowed = VALID_PARENT.get(ptype, frozenset())
            if ntype not in allowed:
                errors.append(_err(f"Invalid hierarchy: {ntype} under {ptype}"))
            if node.get("parent_id") != parent.get("node_id"):
                errors.append(_err(f"parent_id mismatch: expected {parent.get('node_id')}"))

        children = node.get("nodes") or []
        cids = node.get("children") or node.get("children_ids") or []
        actual_ids = [c.get("node_id") for c in children if c.get("node_id")]
        if set(cids) != set(actual_ids):
            errors.append(_err("children mismatch"))

        if not (node.get("path") or "").strip() and ntype != "ROOT":
            errors.append(_err("Missing path"))

        raw = (node.get("raw_content") or "").strip()

        for leg in FORBIDDEN_EXPORT_FIELDS:
            if (node.get(leg) or "").strip():
                errors.append(_err(f"Forbidden field '{leg}' must not be stored on nodes"))

        for leg in ("text", "summary", "prefix_summary"):
            if (node.get(leg) or "").strip() and ntype in RETRIEVAL_TYPES:
                errors.append(_err(f"Legacy field '{leg}' present — use raw_content and micro_summary only"))

        if ntype in CONTAINER_TYPES:
            if len(raw) > 50:
                errors.append(_err(f"Container {ntype} must not store raw content"))
            if node.get("retrieval_ready") or node.get("is_retrieval_chunk"):
                errors.append(_err(f"Container marked retrieval_ready"))

        if ntype == "CONTENT" and raw and not node.get("is_front_matter"):
            retrieval_nodes.append(node)

        if ntype in RETRIEVAL_TYPES and raw and not node.get("is_front_matter"):
            ps, pe = int(node.get("page_start") or 0), int(node.get("page_end") or 0)
            cs, ce = int(node.get("char_start") or 0), int(node.get("char_end") or 0)
            if pe > 0 and ps > pe:
                errors.append(_err(f"Invalid pages {ps}-{pe}"))
            if pe > 0 and ce > 0 and cs >= ce:
                errors.append(_err(f"Invalid char range {cs}-{ce}"))

            micro = (node.get("micro_summary") or "").strip()
            if not micro:
                errors.append(_err("Missing micro_summary"))
            elif len(_split_micro_sentences(micro)) > 3:
                errors.append(_err("micro_summary must be <= 3 sentences"))
            elif raw and micro.lower() not in raw.lower():
                micro_tokens = {t for t in re.findall(r"[a-z0-9]{4,}", micro.lower())}
                raw_tokens = set(re.findall(r"[a-z0-9]{4,}", raw.lower()[:4000]))
                if micro_tokens and not micro_tokens & raw_tokens:
                    errors.append(_err("micro_summary must be grounded in raw_content"))

            if len(raw) < 20:
                errors.append(_err("raw_content too short for retrieval node"))
            if ps > 0 and pe > 0 and cs > 0 and ce > 0 and cs >= ce:
                errors.append(_err(f"Invalid char range {cs}-{ce}"))
            
            ch = node.get("content_hash") or ""
            if ch != sha256_content(raw):
                errors.append(_err("content_hash mismatch"))
            if ch in content_hashes:
                errors.append(_err(f"Duplicate hash with {content_hashes[ch]}"))
            else:
                content_hashes[ch] = nid

            if contains_garbage_artifact(raw):
                errors.append(_err("Garbage artifact in content"))
            if is_paragraph_title(title) and not re.match(r"^\d+\.\d+", title):
                errors.append(_err(f"Paragraph used as title: {title[:60]}"))
            if len(title) > 85 and not re.match(r"^\d+\.\d+", title):
                errors.append(_err(f"Mega-title ({len(title)} chars)"))
            if not node.get("aliases"):
                errors.append(_err("Missing aliases"))
            if not node.get("keywords"):
                errors.append(_err("Missing keywords"))
            if not node.get("synonyms"):
                errors.append(_err("Missing synonyms"))

            compressed = (node.get("compressed_content") or "").strip()
            if not compressed:
                errors.append(_err("Missing compressed_content"))
            else:
                ratio = compression_ratio(raw, compressed)
                if len(raw) > 400:
                    if ratio < 0.18:
                        errors.append(_err(f"compressed_content too short ({ratio:.0%} of raw)"))
                    elif ratio > 0.92:
                        errors.append(_err(f"compressed_content not compressed ({ratio:.0%} of raw)"))
                elif len(compressed) < 15:
                    errors.append(_err("compressed_content too short for retrieval node"))
                if contains_garbage_artifact(compressed):
                    errors.append(_err("Garbage artifact in compressed_content"))

            if parent:
                pr = (parent.get("raw_content") or "").strip()
                if pr and len(raw) > 40 and (raw in pr or jaccard_similarity(pr, raw) >= 0.88):
                    errors.append(_err("Parent duplicates child content"))

            if parent and raw and ntype == "CONTENT":
                ppid = parent.get("node_id", "")
                ptype = normalize_type(parent.get("type", ""))
                if ptype not in ("FRONT_MATTER", "ROOT"):
                    sec_key = re.match(r"^(\d+(?:\.\d+)*)", title.strip())
                    for _, stitle, sraw in sibling_chunks:
                        prev_sec = re.match(r"^(\d+(?:\.\d+)*)", stitle.strip())
                        if not sec_key or not prev_sec:
                            continue
                        if sec_key.group(1) != prev_sec.group(1):
                            continue
                        if stitle.strip() == title.strip() or sha256_content(sraw) == sha256_content(raw):
                            continue
                        if sraw and jaccard_similarity(sraw, raw) >= overlap_adjacent_threshold:
                            errors.append(
                                f"Adjacent overlap > {overlap_adjacent_threshold:.0%}:\n"
                                f"Node A: {stitle}\n"
                                f"Node B: {title} [{nid}]"
                            )
                            break
                    sibling_chunks.append((ppid, title, raw))

        for child in children:
            walk(child, node, path_parts + [title])

    walk(root, None, [])

    for nid, pid in parent_map.items():
        if not pid:
            continue
        chain = {nid}
        cur = pid
        while cur:
            if cur in chain:
                errors.append(f"Cyclic parent chain involving {nid}")
                break
            chain.add(cur)
            cur = parent_map.get(cur)

    if not retrieval_nodes:
        errors.append("No retrieval candidate nodes in index")

    errors.extend(_global_overlap_duplicate_errors(retrieval_nodes, OVERLAP_DUPLICATE_THRESHOLD))

    if strict and errors:
        raise ValidationError(errors)
    return errors


import re
from typing import Any

from .schema import CONTAINER_TYPES, FRONT_MATTER_TYPES, RETRIEVAL_TYPES, normalize_type

# Error substrings mapped to readiness gates
_GATE_PATTERNS: dict[str, tuple[str, ...]] = {
    "hierarchy_valid": (
        "Invalid hierarchy", "parent_id mismatch", "children mismatch", 
        "Unknown type", "Cyclic parent", "Orphan", "orphan", "missing parent"
    ),
    "no_orphans": ("Orphan", "orphan", "missing parent"),
    "no_cycles": ("Cyclic parent",),
    "no_duplicate_chunks": ("Duplicate hash", "Overlap duplicate", "duplicate"),
    "no_parser_artifacts": ("Garbage artifact", "physical_index"),
    "no_malformed_titles": ("Paragraph used as title", "Mega-title", "Synthetic title"),
    "lexical_metadata_complete": (
        "Missing aliases", "Missing keywords",
        "Missing micro_summary", "content_hash mismatch", "Missing synonyms",
        "Forbidden field",
    ),
    "compressed_content_valid": (
        "Missing compressed_content", "compressed_content too short",
        "compressed_content not compressed", "Garbage artifact in compressed",
    ),
    "chunk_validation": ("Multi-topic", "raw_content too short", "Empty raw_content", "micro_summary must be"),
    "no_overlap_adjacent": ("Adjacent overlap",),
}


def _classify_errors(errors: list[str]) -> dict[str, bool]:
    gates = {k: True for k in _GATE_PATTERNS}
    gates["has_retrieval_nodes"] = True
    for err in errors:
        low = err.lower()
        for gate, patterns in _GATE_PATTERNS.items():
            if any(p.lower() in low for p in patterns):
                gates[gate] = False
        if "no retrieval" in low:
            gates["has_retrieval_nodes"] = False
            
    # Issue 2 fix: hierarchy_valid can ONLY be true if NO hierarchy warnings exist.
    # We mapped all hierarchy errors to "hierarchy_valid" above.
    return gates


def apply_retrieval_readiness(
    root: dict[str, Any],
    validation_errors: list[str],
    *,
    observability_initialized: bool = True,
    retrieval_tests_passed: bool | None = None,
    require_tests: bool = False,
) -> dict[str, Any]:
    """
    Set retrieval_ready on nodes and return document-level readiness report.
    retrieval_ready = true ONLY when every gate passes.
    """
    gates = _classify_errors(validation_errors)
    gates["observability_initialized"] = observability_initialized
    if require_tests:
        gates["retrieval_tests_passed"] = retrieval_tests_passed is True
    else:
        gates["retrieval_tests_passed"] = True

    all_pass = all(gates.values())
    ready_count = 0
    total_candidates = 0

    def _node_passes(node: dict) -> bool:
        raw = (node.get("raw_content") or "").strip()
        if len(raw) < 20 or contains_garbage_artifact(raw):
            return False
        compressed = (node.get("compressed_content") or "").strip()
        if not compressed or contains_garbage_artifact(compressed):
            return False
        if len(raw) > 400:
            ratio = compression_ratio(raw, compressed)
            if ratio < 0.18 or ratio > 0.92:
                return False
        if not (node.get("micro_summary") or "").strip():
            return False
        if not node.get("aliases") or not node.get("keywords") or not node.get("synonyms"):
            return False
        if is_paragraph_title(node.get("title", "")) or is_legal_section_reference_title(
            node.get("title", "")
        ):
            return False
        ch = node.get("content_hash") or ""
        if ch != sha256_content(raw):
            return False
        if not (node.get("path") or "").strip() or not node.get("parent_id"):
            return False
        return True

    def walk(node: dict) -> None:
        nonlocal ready_count, total_candidates
        ntype = normalize_type(node.get("type", ""))
        if ntype in FRONT_MATTER_TYPES or ntype == "TABLE_OF_CONTENTS":
            node["retrieval_ready"] = False
            node["is_retrieval_chunk"] = False
            node["is_front_matter"] = True
        elif ntype in RETRIEVAL_TYPES:
            if node.get("is_front_matter"):
                node["retrieval_ready"] = False
                node["is_retrieval_chunk"] = False
            else:
                total_candidates += 1
                node_ready = all_pass and _node_passes(node)
                node["retrieval_ready"] = node_ready
                node["is_retrieval_chunk"] = node_ready
                if node_ready:
                    ready_count += 1
        elif ntype in CONTAINER_TYPES:
            node["retrieval_ready"] = False
            node["is_retrieval_chunk"] = False
        for c in node.get("nodes") or []:
            walk(c)

    walk(root)

    doc_ready = all_pass and total_candidates > 0 and ready_count == total_candidates
    return {
        "retrieval_ready": doc_ready,
        "ready_node_count": ready_count,
        "candidate_node_count": total_candidates,
        "gates": gates,
        "validation_error_count": len(validation_errors),
        "blocking_errors": validation_errors[:20] if not all_pass else [],
    }


def clear_retrieval_ready(root: dict[str, Any]) -> None:
    """Force all nodes not ready (failed build)."""

    def walk(node: dict) -> None:
        node["retrieval_ready"] = False
        node["is_retrieval_chunk"] = False
        for c in node.get("nodes") or []:
            walk(c)

    walk(root)
