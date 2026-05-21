from __future__ import annotations
# --- validator.py ---
"""Production index validation — fail build on any integrity violation."""

import re
from typing import Any

from .processing import compression_ratio
from .schema import CONTAINER_TYPES, NODE_TYPES, RETRIEVAL_TYPES, VALID_PARENT, normalize_type
from .processing import contains_garbage_artifact
from .processing import is_paragraph_title, is_synthetic_title, jaccard_similarity, sha256_content

MULTI_TOPIC_PATTERNS = [
    re.compile(r"gstr[- ]?1\b", re.I),
    re.compile(r"gstr[- ]?2a", re.I),
    re.compile(r"gstr[- ]?2b", re.I),
    re.compile(r"gstr[- ]?3b", re.I),
    re.compile(r"reg[- ]?0?6", re.I),
    re.compile(r"drc[- ]?0?3", re.I),
]

OVERLAP_DUPLICATE_THRESHOLD = 0.85


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
            return f"{msg}\nNode: {n_path} [{n_type}] ({n_id})"

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

        for leg in ("text", "summary", "prefix_summary"):
            if (node.get(leg) or "").strip():
                if ntype in RETRIEVAL_TYPES and len(node.get(leg) or "") > 50:
                    errors.append(
                        _err(f"Legacy field '{leg}' present — use raw_content/compressed_content/micro_summary")
                    )

        if ntype in CONTAINER_TYPES:
            if len(raw) > 50:
                errors.append(_err(f"Container {ntype} must not store raw content"))
            if node.get("retrieval_ready") or node.get("is_retrieval_chunk"):
                errors.append(_err(f"Container marked retrieval_ready"))

        if ntype == "CONTENT" and raw:
            retrieval_nodes.append(node)

        if ntype in RETRIEVAL_TYPES and raw:
            ps, pe = int(node.get("page_start") or 0), int(node.get("page_end") or 0)
            cs, ce = int(node.get("char_start") or 0), int(node.get("char_end") or 0)
            if pe > 0 and ps > pe:
                errors.append(_err(f"Invalid pages {ps}-{pe}"))
            if pe > 0 and ce > 0 and cs >= ce:
                errors.append(_err(f"Invalid char range {cs}-{ce}"))

            comp = (node.get("compressed_content") or "").strip()
            if not comp:
                errors.append(_err("Missing compressed_content"))
            elif len(raw) > 400:
                ratio = compression_ratio(raw, comp)
                if ratio < 0.35 or ratio > 0.92:
                    errors.append(_err(f"Compression ratio {ratio:.2f} out of range"))
            if not (node.get("micro_summary") or "").strip():
                errors.append(_err("Missing micro_summary"))
            
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
            if not node.get("keywords") and ntype == "CONTENT" and not node.get("is_front_matter"):
                errors.append(_err("Missing keywords"))
            if not node.get("synonyms"):
                errors.append(_err("Missing synonyms"))

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

    for i, a in enumerate(retrieval_nodes):
        ra = a.get("raw_content") or ""
        for b in retrieval_nodes[i + 1 :]:
            rb = b.get("raw_content") or ""
            if ra and rb and jaccard_similarity(ra, rb) >= OVERLAP_DUPLICATE_THRESHOLD:
                errors.append(
                    f"Overlap duplicate:\n"
                    f"Node A: {a.get('path')} [{a.get('node_id')}]\n"
                    f"Node B: {b.get('path')} [{b.get('node_id')}]"
                )

    if strict and errors:
        raise ValidationError(errors)
    return errors


# --- readiness.py ---
"""Retrieval readiness gating — true only when all production checks pass."""

import re
from typing import Any

from .schema import CONTAINER_TYPES, RETRIEVAL_TYPES, normalize_type

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
        "Missing aliases", "Missing keywords", "Missing compressed", 
        "Missing micro_summary", "content_hash mismatch", "Missing synonyms"
    ),
    "chunk_validation": ("Multi-topic", "Compression ratio", "Empty raw_content"),
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

    def walk(node: dict) -> None:
        nonlocal ready_count, total_candidates
        ntype = normalize_type(node.get("type", ""))
        if ntype in RETRIEVAL_TYPES:
            total_candidates += 1
            node["retrieval_ready"] = all_pass
            node["is_retrieval_chunk"] = all_pass
            if all_pass:
                ready_count += 1
        elif ntype in CONTAINER_TYPES:
            node["retrieval_ready"] = False
            node["is_retrieval_chunk"] = False
        for c in node.get("nodes") or []:
            walk(c)

    walk(root)

    return {
        "retrieval_ready": all_pass and ready_count > 0,
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
