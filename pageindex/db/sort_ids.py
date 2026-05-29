"""Deterministic row ids that sort lexicographically by (seq_id, node_id)."""


def sortable_node_row_id(seq_id: int | None, node_id: str | None) -> str:
    """10-char id: 6-digit seq_id + 4-digit node_id (e.g. 0000010001)."""
    s = int(seq_id or 0)
    n = str(node_id or "0000").strip()
    if len(n) < 4:
        n = n.zfill(4)
    elif len(n) > 4:
        n = n[-4:]
    return f"{s:06d}{n}"
