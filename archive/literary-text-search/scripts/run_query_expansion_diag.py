"""
A/B: query synonym expansion ON vs OFF (BM25 pool + full NZ dense).

Default cases focus on реч↔слово; mechanism is generic (config.QUERY_SYNONYMS).

Run from backend/ (venv active):
  python ../archive/literary-text-search/scripts/run_query_expansion_diag.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ARCHIVE_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = Path(__file__).resolve().parents[3] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from config import DATA_DIR
from services.detection import (
    expand_query,
    get_bm25_ranked_pool,
    get_dense_ranked_pool,
)

RESULTS_DIR = ARCHIVE_DIR / "benchmark" / "results"

DEFAULT_CASES: list[dict] = [
    {
        "id": "ubime-prejaka-rec",
        "query": "Уби ме прејака реч",
        "expected": [{"book": "2. Коринћанима", "chapter": 3, "verse": 6}],
    },
    {
        "id": "ubime-prejako-slovo",
        "query": "Уби ме прејако слово",
        "expected": [{"book": "2. Коринћанима", "chapter": 3, "verse": 6}],
    },
    {
        "id": "pocetku-bese-slovo",
        "query": "У почетку бјеше Слово",
        "expected": [{"book": "Јован", "chapter": 1, "verse": 1}],
    },
]


def _ref_key(book: str, chapter: int, verse: int) -> tuple[str, int, int]:
    return (book.strip(), int(chapter), int(verse))


def _find_rank(pool: list[dict], expected: list[dict]) -> tuple[int | None, dict | None]:
    by_ref = {
        _ref_key(p["book"], p["chapter"], p["verse"]): p for p in pool
    }
    best: tuple[int | None, dict | None] = (None, None)
    for exp in expected:
        key = _ref_key(exp["book"], exp["chapter"], exp["verse"])
        found = by_ref.get(key)
        if found is None:
            continue
        rank = int(found["rank"])
        if best[0] is None or rank < best[0]:
            best = (rank, found)
    return best


def _eval_case(case: dict, corpus: str, expand: bool, bm25_k: int, dense_k: int) -> dict:
    q = case["query"]
    expanded = expand_query(q, enabled=expand)
    bm25_pool = get_bm25_ranked_pool(q, corpus=corpus, pool_size=bm25_k, expand=expand)
    dense_pool = get_dense_ranked_pool(
        q, corpus=corpus, pool_size=dense_k, expand=expand
    )
    bm25_rank, bm25_hit = _find_rank(bm25_pool, case["expected"])
    dense_rank, dense_hit = _find_rank(dense_pool, case["expected"])
    return {
        "id": case["id"],
        "query": q,
        "expand": expand,
        "expanded_query": expanded,
        "bm25_rank": bm25_rank,
        "bm25_in_pool": bm25_rank is not None,
        "dense_rank": dense_rank,
        "dense_in_pool": dense_rank is not None,
        "dense_cosine": dense_hit.get("cosine_score") if dense_hit else None,
        "bm25_top3": [
            f"{p['book']} {p['chapter']}:{p['verse']}" for p in bm25_pool[:3]
        ],
        "dense_top3": [
            f"{p['book']} {p['chapter']}:{p['verse']}" for p in dense_pool[:3]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Query expansion ON/OFF diagnostic")
    parser.add_argument("--corpus", default="dk")
    parser.add_argument("--bm25-k", type=int, default=200)
    parser.add_argument("--dense-k", type=int, default=500)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"query_expansion_diag_{stamp}.json"

    rows: list[dict] = []
    print(f"corpus={args.corpus} bm25_k={args.bm25_k} dense_k={args.dense_k}")
    print("-" * 78)

    for case in DEFAULT_CASES:
        print(f"\n[{case['id']}] {case['query']}")
        for expand in (False, True):
            row = _eval_case(case, args.corpus, expand, args.bm25_k, args.dense_k)
            rows.append(row)
            flag = "ON " if expand else "OFF"
            br = f"#{row['bm25_rank']}" if row["bm25_rank"] else "absent"
            dr = f"#{row['dense_rank']}" if row["dense_rank"] else "absent"
            print(f"  expand={flag}  bm25={br:8s}  dense={dr:8s}")
            if expand and row["expanded_query"] != row["query"]:
                print(f"           → {row['expanded_query']}")

    print("\n" + "=" * 78)
    print(f"{'case':28s}{'bm25 OFF':>10}{'bm25 ON':>10}{'dense OFF':>11}{'dense ON':>10}")
    by_id: dict[str, dict[bool, dict]] = {}
    for row in rows:
        by_id.setdefault(row["id"], {})[row["expand"]] = row
    for case_id, modes in by_id.items():
        off, on = modes[False], modes[True]
        def _r(v: int | None) -> str:
            return "—" if v is None else str(v)
        print(
            f"{case_id:28s}"
            f"{_r(off['bm25_rank']):>10}"
            f"{_r(on['bm25_rank']):>10}"
            f"{_r(off['dense_rank']):>11}"
            f"{_r(on['dense_rank']):>10}"
        )

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "corpus": args.corpus,
        "bm25_k": args.bm25_k,
        "dense_k": args.dense_k,
        "cases": rows,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
