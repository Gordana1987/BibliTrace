"""
Build search indexes for one or more corpora: lemmatize → BM25 → Qwen (+ optional LaBSE).

Full-bible Qwen/LaBSE are for the legacy /api/analyze path.
Concept-search semantic uses Embedić (build_embedic_nt_embeddings.py), not this pipeline.

Run from backend/:
  python scripts/build_pipeline.py --corpus dk
  python scripts/build_pipeline.py --all-active
  python scripts/build_pipeline.py --all-active --skip-lemmatize
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ACTIVE_CORPORA

from build_bm25_index import build_index as build_bm25
from build_embeddings import build_labse_index, build_qwen_index
from lemmatize_bible import lemmatize_bible_csv


def build_corpus(
    corpus: str,
    *,
    skip_lemmatize: bool = False,
    labse: bool = True,
) -> None:
    print(f"\n=== Pipeline: {corpus} ===")
    if not skip_lemmatize:
        lemmatize_bible_csv(corpus=corpus)
    else:
        print("Skipping lemmatization (--skip-lemmatize).")
    build_bm25(corpus=corpus)
    build_qwen_index(corpus=corpus)
    if labse:
        build_labse_index(corpus=corpus)
    print(f"=== Done: {corpus} ===\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Lemmatize + BM25 + embeddings for Pouke corpora")
    parser.add_argument(
        "--corpus",
        action="append",
        dest="corpora",
        choices=ACTIVE_CORPORA,
        help=f"Corpus id (repeatable). Default active: {ACTIVE_CORPORA}",
    )
    parser.add_argument(
        "--all-active",
        action="store_true",
        help=f"Build all ACTIVE_CORPORA: {ACTIVE_CORPORA}",
    )
    parser.add_argument(
        "--skip-lemmatize",
        action="store_true",
        help="Reuse existing bible_lemmatized.csv (faster rebuild of indexes only).",
    )
    parser.add_argument(
        "--no-labse",
        action="store_true",
        help="Skip LaBSE embeddings (Qwen only).",
    )
    args = parser.parse_args()

    corpora = list(ACTIVE_CORPORA) if args.all_active else (args.corpora or ["dk"])
    for corpus in corpora:
        if corpus not in ACTIVE_CORPORA:
            raise SystemExit(f"Unknown or inactive corpus: {corpus!r}. Active: {ACTIVE_CORPORA}")
        build_corpus(
            corpus,
            skip_lemmatize=args.skip_lemmatize,
            labse=not args.no_labse,
        )


if __name__ == "__main__":
    main()
