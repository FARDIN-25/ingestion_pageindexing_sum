"""Order flat node lists for stable seq_id / node_id assignment."""
from __future__ import annotations

from collections import defaultdict


def ordered_flat_nodes(flat_nodes: list[dict]) -> list[dict]:
    """
    Return nodes in tree order (pre-order DFS) so assigned node_id values
    are 0001, 0002, ... in ascending order while keeping parent links valid.
    """
    nodes = [n for n in flat_nodes if isinstance(n, dict)]
    if not nodes:
        return []

    def _nid(n: dict) -> str:
        return str(n.get("node_id") or "")

    idx = {_nid(n): n for n in nodes if _nid(n)}
    children: dict[str, list[str]] = defaultdict(list)
    roots: list[str] = []

    for n in nodes:
        nid = _nid(n)
        if not nid:
            continue
        pid = n.get("parent_id")
        pid_str = str(pid) if pid is not None else ""
        if pid is not None and pid_str in idx:
            children[pid_str].append(nid)
        else:
            roots.append(nid)

    order: list[dict] = []
    seen: set[str] = set()

    def dfs(nid: str) -> None:
        if nid in seen or nid not in idx:
            return
        seen.add(nid)
        order.append(idx[nid])
        for ch in sorted(children.get(nid, [])):
            dfs(ch)

    for r in sorted(roots):
        dfs(r)
    for nid in sorted(idx.keys()):
        if nid not in seen:
            dfs(nid)
    return order


def sort_tree_nodes(nodes: list[dict]) -> None:
    """Sort tree children by node_id ascending (in place)."""
    nodes.sort(key=lambda x: x.get("node_id") or "")
    for n in nodes:
        kids = n.get("nodes")
        if kids:
            sort_tree_nodes(kids)
