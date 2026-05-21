"""Terminal-safe logging (works with uvicorn --reload on Windows)."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
LOG_FILE = LOG_DIR / "app.log"

_configured = False


def setup_logging() -> logging.Logger:
    global _configured
    LOG_DIR.mkdir(exist_ok=True)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # stderr is always visible in uvicorn reload worker on Windows
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(fmt)

    file_h = logging.FileHandler(LOG_FILE, encoding="utf-8", mode="a")
    file_h.setFormatter(fmt)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    root.addHandler(console)
    root.addHandler(file_h)

    for name in (
        "pageindex-ui",
        "pageindex",
        "pageindex.vrag",
        "pageindex.page_index",
        "pageindex.utils",
        "pageindex.pageindex_api",
        "pageindex.cloud_index",
        "uvicorn",
        "uvicorn.access",
        "uvicorn.error",
        "urllib3",
        "requests",
    ):
        log = logging.getLogger(name)
        log.handlers.clear()
        log.setLevel(logging.INFO)
        log.propagate = True

    _configured = True
    return logging.getLogger("pageindex-ui")


def terminal_log(msg: str) -> None:
    """Always visible: stderr + log file."""
    line = msg if isinstance(msg, str) else str(msg)
    try:
        sys.stderr.write(line + "\n")
        sys.stderr.flush()
    except Exception:
        pass
    logging.getLogger("pageindex-ui").info(line)


def log_info(msg: str, *args) -> None:
    text = msg % args if args else msg
    terminal_log(text)


def log_error(msg: str, *args) -> None:
    text = msg % args if args else msg
    logging.getLogger("pageindex-ui").error(text)
    try:
        sys.stderr.write(f"[ERROR] {text}\n")
        sys.stderr.flush()
    except Exception:
        pass


def log_exception(msg: str, *args) -> None:
    text = msg % args if args else msg
    logging.getLogger("pageindex-ui").exception(text)
    try:
        sys.stderr.write(f"[ERROR] {text}\n")
        sys.stderr.flush()
    except Exception:
        pass
