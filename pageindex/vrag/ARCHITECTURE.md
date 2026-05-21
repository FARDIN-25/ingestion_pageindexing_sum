# Vectorless RAG Pipeline (v2.2)

Production hierarchical index for **lexical / FTS / BM25** retrieval — no embeddings.

## Pipeline stages

1. **Extract** (`extractor.py`) — PyMuPDF line-level text + font metadata, OCR cleanup
2. **Sanitize** (`sanitizer.py`) — remove parser garbage, page markers, hyphen merge
3. **Headings** (`headings.py`) — structural heading detection only (no paragraph titles)
4. **Hierarchy** (`hierarchy.py`) — `ROOT → FRONT_MATTER | SECTION → UNIT → TOPIC → SUBTOPIC → CONTENT`
5. **Chunk** (`chunker.py`) — one concept per `CONTENT` node; split on numbering / forms / size
6. **Dedup** (`dedup.py`) — SHA256 exact hash + Jaccard overlap ≥ 0.85 reject
7. **Compress** (`compressor.py`) — 60–80% semantic compression (not summarization)
8. **Metadata** (`metadata.py`) — aliases, keywords, synonyms for lexical recall
9. **Validate** (`validator.py`) — fail build on any integrity violation
10. **Retrieve** (`retrieval.py`) — 9-stage explainable lexical search

## Node schema

Leaf retrieval nodes are type `CONTENT` with `retrieval_ready: true`. Containers never store `raw_content`.

## Build

```bash
python scripts/build_vrag_index.py --pdf uploads/your.pdf --test-retrieval
```

Set `pipeline: vrag` in `pageindex/config.yaml` for local builds (no PageIndex API).
