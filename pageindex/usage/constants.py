"""Credit usage operation identifiers."""
from __future__ import annotations

from enum import Enum


class Operation(str, Enum):
    PDF_UPLOAD = "pdf_upload"
    OCR_EXTRACTION = "ocr_extraction"
    DOCUMENT_PARSING = "document_parsing"
    PAGE_INDEXING = "page_indexing"
    STRUCTURE_DETECTION = "structure_detection"
    HEADING_DETECTION = "heading_detection"
    CHUNK_GENERATION = "chunk_generation"
    COMPRESSION = "compression_generation"
    MICRO_SUMMARY = "micro_summary_generation"
    ALIAS_GENERATION = "alias_generation"
    KEYWORD_GENERATION = "keyword_generation"
    DEDUPE_VALIDATION = "dedupe_validation"
    INDEXING = "indexing"
    RETRIEVAL_PREP = "retrieval_prep"
    TEST_VALIDATION = "test_validation"
    RETRY = "retry"
    FAILED_JOB = "failed_job"
    PARTIAL_PROCESSING = "partial_processing"
    REPROCESSING = "reprocessing"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    CLOUD_SUBMIT = "cloud_submit"
    CLOUD_POLL = "cloud_poll"
    CLOUD_METADATA = "cloud_metadata"
    CLOUD_TREE = "cloud_tree"
    CLOUD_RETRIEVAL = "cloud_retrieval"
    CLOUD_CHAT = "cloud_chat"


class Accuracy(str, Enum):
    EXACT = "EXACT"
    ESTIMATED = "ESTIMATED"
    LOCAL = "LOCAL"  # zero API credits; deterministic local compute only


class EventStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    SKIPPED = "skipped"
    RETRY = "retry"
