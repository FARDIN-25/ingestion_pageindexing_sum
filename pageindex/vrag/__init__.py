"""Production vectorless RAG indexing and retrieval."""
from .pipeline import build_index, build_index_safe, LexicalRetriever, run_test_queries, search
from .validation import apply_retrieval_readiness, ValidationError, validate_index

build_vrag_index = build_index
validate_tree = validate_index
validate_or_raise = lambda s: validate_index(s, strict=True)
IndexValidationError = ValidationError

__all__ = [
    "build_vrag_index",
    "build_index",
    "build_index_safe",
    "search",
    "LexicalRetriever",
    "run_test_queries",
    "validate_tree",
    "validate_index",
    "validate_or_raise",
    "apply_retrieval_readiness",
    "ValidationError",
    "IndexValidationError",
]
