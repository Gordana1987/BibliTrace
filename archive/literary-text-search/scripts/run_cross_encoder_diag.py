"""
A/B: embedding-only top-20 vs CE rerank of embedding shortlist (pool 50/100).

Expansion can be forced ON for the hard реч↔слово case (live default stays OFF).

Run from backend/ (venv active):
  python ../archive/literary-text-search/scripts/run_cross_encoder_diag.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ARCHIVE_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = Path(__file__).resolve().parents[3] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from config import CROSS_ENCODER_MODEL, DATA_DIR
from services.detection import (
    _build_semantic_pool,
    _get_bm25_candidates,
    _get_phrase_match_indices,
    expand_query,
    score_pool_with_cross_encoder,
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
        "id": "sinovi-groma",
        "query": "синови грома",
        "expected": [{"book": "Марко", "chapter": 3, "verse": 17}],
    },
    {
        "id": "pocetku-bese-rec",
        "query": "У почетку беше Реч",
        "expected": [{"book": "Јован", "chapter": 1, "verse": 1}],
    },
]


def _ref_key(book: str, chapter: int, verse: int) -> tuple[str, int, int]:
    return (book.strip(), int(chapter), int(verse))


def _find_rank(pool: list[dict], expected: list[dict]) -> tuple[int | None, dict | None]:
    by_ref = {_ref_key(p["book"], p["chapter"], p["verse"]): (i + 1, p) for i, p in enumerate(pool)}
    best: tuple[int | None, dict | None] = (None, None)
    for exp in expected:
        key = _ref_key(exp["book"], exp["chapter"], exp["verse"])
        found = by_ref.get(key)
        if found is None:
            continue
        rank, hit = found
        if best[0] is None or rank < best[0]:
            best = (rank, hit)
    return best


def _top_refs(pool: list[dict], n: int = 5) -> list[str]:
    return [f"{p['book']} {p['chapter']}:{p['verse']}" for p in pool[:n]]


def _eval_case(case: dict, corpus: str, expand: bool, pool_size: int) -> dict:
    q = case["query"]
    search_text = expand_query(q, enabled=expand)
    candidates = _get_bm25_candidates(search_text, corpus)
    phrases = set(_get_phrase_match_indices(q, corpus))
    pool = _build_semantic_pool(
        search_text,
        candidates,
        "qwen",
        corpus,
        phrases,
        pool_size=pool_size,
    )
    embed_rank, embed_hit = _find_rank(pool, case["expected"])

    t0 = time.perf_counter()
    ce_pool = score_pool_with_cross_encoder(search_text, pool)
    ce_ms = (time.perf_counter() - t0) * 1000
    ce_rank, ce_hit = _find_rank(ce_pool, case["expected"])

    return {
        "id": case["id"],
        "query": q,
        "expand": expand,
        "expanded_query": search_text,
        "pool_size": pool_size,
        "pool_filled": len(pool),
        "embed_rank_in_pool": embed_rank,
        "embed_in_pool": embed_rank is not None,
        "embed_score": embed_hit.get("embed_score") if embed_hit else None,
        "ce_rank": ce_rank,
        "ce_in_top20": ce_rank is not None and ce_rank <= 20,
        "ce_score": ce_hit.get("ce_score") if ce_hit else None,
        "ce_ms": round(ce_ms, 1),
        "embed_top5": _top_refs(pool, 5),
        "ce_top5": _top_refs(ce_pool, 5),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-encoder vs embedding shortlist diagnostic")
    parser.add_argument("--corpus", default="dk")
    parser.add_argument(
        "--pool-sizes",
        default="50,100",
        help="Comma-separated embedding shortlist sizes before CE",
    )
    parser.add_argument(
        "--expand",
        choices=("off", "on", "both"),
        default="both",
        help="Force query expansion (live default is OFF)",
    )
    args = parser.parse_args()

    pool_sizes = [int(x.strip()) for x in args.pool_sizes.split(",") if x.strip()]
    expand_modes = {"off": [False], "on": [True], "both": [False, True]}[args.expand]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"cross_encoder_diag_{stamp}.json"

    # Warm-load CE once so first-case timing is fairer.
    print(f"Loading CE model {CROSS_ENCODER_MODEL} …")
    score_pool_with_cross_encoder("warm", [{"verse_text": "warm"}])

    rows: list[dict] = []
    print(f"corpus={args.corpus} pool_sizes={pool_sizes} expand={args.expand}")
    print("-" * 88)

    for case in DEFAULT_CASES:
        print(f"\n[{case['id']}] {case['query']}")
        for expand in expand_modes:
            for pool_size in pool_sizes:
                row = _eval_case(case, args.corpus, expand, pool_size)
                rows.append(row)
                flag = "ON " if expand else "OFF"
                er = f"#{row['embed_rank_in_pool']}" if row["embed_rank_in_pool"] else "absent"
                cr = f"#{row['ce_rank']}" if row["ce_rank"] else "absent"
                top20 = "yes" if row["ce_in_top20"] else "no"
                print(
                    f"  expand={flag} pool={pool_size:3d}  "
                    f"embed_in_pool={er:8s}  ce={cr:8s}  top20={top20}  "
                    f"({row['ce_ms']:.0f} ms)"
                )
                if expand and row["expanded_query"] != row["query"]:
                    print(f"           → {row['expanded_query']}")

    print("\n" + "=" * 88)
    print(
        f"{'case':24s}{'exp':>4}{'pool':>6}"
        f"{'embed#':>8}{'ce#':>6}{'≤20':>5}"
    )
    for row in rows:
        er = "—" if row["embed_rank_in_pool"] is None else str(row["embed_rank_in_pool"])
        cr = "—" if row["ce_rank"] is None else str(row["ce_rank"])
        print(
            f"{row['id']:24s}"
            f"{('Y' if row['expand'] else 'N'):>4}"
            f"{row['pool_size']:>6}"
            f"{er:>8}"
            f"{cr:>6}"
            f"{('Y' if row['ce_in_top20'] else 'N'):>5}"
        )

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "corpus": args.corpus,
        "model": CROSS_ENCODER_MODEL,
        "pool_sizes": pool_sizes,
        "expand_modes": expand_modes,
        "cases": rows,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
