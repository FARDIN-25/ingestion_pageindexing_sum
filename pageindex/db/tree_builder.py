"""Build nested structure trees from document_nodes rows."""
from __future__ import annotations

from typing import Any

from pageindex.db.models import DocumentNode
from pageindex.db.node_order import sort_tree_nodes


def _node_payload(row: DocumentNode, *, include_body: bool = True) -> dict[str, Any]:
    meta = row.metadata_json if isinstance(row.metadata_json, dict) else {}
    payload: dict[str, Any] = {
        "seq_id": row.seq_id,
        "node_id": row.node_id,
        "parent_id": row.parent_id,
        "type": row.type,
        "title": row.title,
        "path": row.path,
        "level": row.level,
        "page_start": meta.get("page_start") or meta.get("page_index") or 0,
        "page_end": meta.get("page_end") or 0,
        "char_start": meta.get("char_start") or 0,
        "char_end": meta.get("char_end") or 0,
        "micro_summary": row.micro_summary or "",
        "content_hash": row.content_hash or "",
        "aliases": list(meta.get("aliases") or []),
        "keywords": list(meta.get("keywords") or []),
        "synonyms": list(meta.get("synonyms") or []),
        "retrieval_ready": bool(row.retrieval_ready),
        "is_retrieval_chunk": bool(row.retrieval_ready),
        "is_front_matter": bool(row.is_front_matter),
        "children": [],
        "children_ids": [],
        "nodes": [],
    }
    if include_body:
        payload["raw_content"] = row.raw_content or ""
        payload["token_count_raw"] = int(
            meta.get("token_count_raw") or len((row.raw_content or "").split())
        )
    return payload


def build_tree_from_nodes(
    rows: list[DocumentNode],
    *,
    include_body: bool = True,
) -> list[dict[str, Any]]:
    if not rows:
        return []

    node_map: dict[str, dict[str, Any]] = {}
    for row in rows:
        node_map[row.node_id] = _node_payload(row, include_body=include_body)

    roots: list[dict[str, Any]] = []
    for row in rows:
        payload = node_map[row.node_id]
        pid = row.parent_id
        if pid and pid in node_map:
            parent = node_map[pid]
            parent["nodes"].append(payload)
            parent["children"].append(row.node_id)
            parent["children_ids"].append(row.node_id)
        else:
            roots.append(payload)

    sort_tree_nodes(roots)
    return roots


def build_structure_vrag_root(rows: list[DocumentNode], *, include_body: bool = True) -> dict[str, Any]:
    """Return a ROOT wrapper matching export_node-style API responses."""
    children = build_tree_from_nodes(rows, include_body=include_body)
    root: dict[str, Any] = {
        "node_id": "root_0001",
        "parent_id": None,
        "type": "ROOT",
        "title": "ROOT",
        "path": "ROOT",
        "level": 0,
        "raw_content": "",
        "micro_summary": "",
        "children": [c["node_id"] for c in children],
        "children_ids": [c["node_id"] for c in children],
        "nodes": children,
        "retrieval_ready": False,
        "is_retrieval_chunk": False,
        "is_front_matter": False,
    }
    return root
