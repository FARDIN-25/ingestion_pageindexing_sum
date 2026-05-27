#!/usr/bin/env python3
"""CLI: build vectorless RAG index from PDF."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pageindex.config import ConfigLoader
from pageindex.vrag import build_index as build_vrag_index


def main():
    parser = argparse.ArgumentParser(description="Build VRAG index from PDF")
    parser.add_argument("--pdf", required=True, help="Path to PDF")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--test-retrieval", action="store_true")
    args = parser.parse_args()

    opt = ConfigLoader().load({"pipeline": "vrag", "run_retrieval_tests": "yes" if args.test_retrieval else "no"})
    result = build_vrag_index(args.pdf, opt=opt, run_retrieval_tests=args.test_retrieval)

    out = Path(args.output) if args.output else ROOT / "results" / f"{Path(args.pdf).stem}_structure.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Saved: {out}")
    print(f"Chunks: {result['validation']['chunk_count']}")
    print(f"Valid: {result['validation']['valid']}")
    if result.get("retrieval_report"):
        rr = result["retrieval_report"]
        print(f"Retrieval tests: {rr.get('passed', 0)}/{rr.get('total', 0)}")


if __name__ == "__main__":
    main()
