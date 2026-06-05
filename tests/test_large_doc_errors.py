"""Large-document failure handling."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pageindex.api_errors import format_user_error
from pageindex.job_results import slim_job_results
from pageindex.vrag.validation import _global_overlap_duplicate_errors


def test_memory_error_message_not_empty():
    msg = format_user_error(MemoryError())
    assert msg
    assert "memory" in msg.lower() or "large" in msg.lower()


def test_slim_job_results_keeps_small_structure():
    big = {
        "pipeline": "pageindex",
        "schema_version": "2.5",
        "doc_id": "abc123",
        "structure": {"type": "ROOT", "nodes": []},
        "structure_vrag": {"type": "ROOT", "nodes": [{"node_id": "0001"}]},
        "page_count": 10,
        "validation": {"valid": True, "errors": ["a"], "error_count": 1},
    }
    slim = slim_job_results(big)
    assert slim["structure_vrag"]["type"] == "ROOT"
    assert slim["structure_url"] == "/api/ingestion/jobs/abc123"
    assert slim["page_count"] == 10


def test_overlap_check_bounded():
    nodes = [
        {"node_id": f"n{i}", "path": f"p{i}", "raw_content": f"GST topic {i} " + ("word " * 200)}
        for i in range(900)
    ]
    errors = _global_overlap_duplicate_errors(nodes, 0.85)
    assert isinstance(errors, list)
