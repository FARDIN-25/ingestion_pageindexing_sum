"""JSON persistence helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def export_build_result(result: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return path
