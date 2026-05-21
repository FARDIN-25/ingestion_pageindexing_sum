from __future__ import annotations
"""Canonical path builder: ROOT > SECTION_B > UNIT_6 > 6.2 GST REG-06"""

import re
from typing import Any

from .schema import normalize_type


def _segment(title: str, ntype: str) -> str:
    t = (title or "").strip()
    if ntype == "ROOT":
        return "ROOT"
    if ntype == "SECTION":
        m = re.search(r"SECTION\s+([A-Z])", t, re.I)
        return f"SECTION_{m.group(1).upper()}" if m else _slug(t)
    if ntype == "UNIT":
        m = re.search(r"UNIT\s+([IVXLC\d]+)", t, re.I)
        return f"UNIT_{m.group(1).upper()}" if m else _slug(t)
    if re.match(r"^\d+\.\d+", t):
        return _slug(t, 50)
    return _slug(t, 50)


def _slug(text: str, max_len: int = 40) -> str:
    s = re.sub(r"[^\w\s\-.]", "", text or "")
    s = re.sub(r"\s+", "_", s.strip())
    return s[:max_len] or "NODE"


def rebuild_paths(root: dict[str, Any]) -> None:
    def walk(node: dict, parent_path: str) -> None:
        ntype = normalize_type(node.get("type", ""))
        seg = _segment(node.get("title", ""), ntype)
        if parent_path == "ROOT" or not parent_path:
            path = seg if ntype == "ROOT" else f"ROOT > {seg}"
        else:
            path = f"{parent_path} > {seg}"
        node["path"] = path
        for c in node.get("nodes") or []:
            walk(c, path)

    walk(root, "")
