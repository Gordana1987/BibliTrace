"""
A/B diagnostic: Qwen query encode modes (query | doc | mean) on full NZ dense retrieval.

No BM25. Reports dense rank / hit@20 / hit@100 for expected anchors per mode.

Run from backend/ (venv active):
  python ../archive/literary-text-search/scripts/run_query_encode_diag.py
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

from config import DATA_DIR, QUERY_ENCODE_MODES
from services.detection import get_dense_ranked_pool

RESULTS_DIR = ARCHIVE_DIR / "benchmark" / "results"

# Focus cases for the реч↔слово motif + a few literary controls.
DEFAULT_CASES: list[dict] = [
    {
        "id": "ubime-prejaka-rec",
        "query": "Уби ме прејака реч",
        "expected": [{"book": "2. Коринћанима", "chapter": 3, "verse": 6}],
        "note": "Literary реч; target uses слово убија",
    },
    {
        "id": "ubime-prejako-slovo",
        "query": "Уби ме прејако слово",
        "expected": [{"book": "2. Коринћанима", "chapter": 3, "verse": 6}],
        "note": "Lexical bridge слово — control",
    },
    {
        "id": "pocetku-bese-slovo",
        "query": "У почетку бјеше Слово",
        "expected": [{"book": "Јован", "chapter": 1, "verse": 1}],
        "note": "Слово vs ријеч/Логос in John 1:1",
    },
    {
        "id": "pocetku-bese-rijec",
        "query": "У почетку бјеше ријеч",
        "expected": [{"book": "Јован", "chapter": 1, "verse": 1}],
        "note": "Surface match control for John 1:1",
    },
]


def _ref_key(book: str, chapter: int, verse: int) -> tuple[str, int, int]:
    return (book.strip(), int(chapter), int(verse))


def _evaluate(case: dict, corpus: str, mode: str, pool_size: int) -> dict:
    pool = get_dense_ranked_pool(
        case["query"],
        corpus=corpus,
        pool_size=pool_size,
        encode_mode=mode,
    )
    by_ref = {
        _ref_key(item["book"], item["chapter"], item["verse"]): item for item in pool
    }

    hits = []
    for exp in case.get("expected", []):
        key = _ref_key(exp["book"], exp["chapter"], exp["verse"])
        found = by_ref.get(key)
        hits.append(
            {
                "book": exp["book"],
                "chapter": exp["chapter"],
                "verse": exp["verse"],
                "in_pool": found is not None,
                "rank": found["rank"] if found else None,
                "cosine": found["cosine_score"] if found else None,
            }
        )

    ranks = [h["rank"] for h in hits if h["rank"] is not None]
    best_rank = min(ranks) if ranks else None
    return {
        "id": case["id"],
        "query": case["query"],
        "note": case.get("note", ""),
        "encode_mode": mode,
        "hits": hits,
        "best_rank": best_rank,
        "hit_at_20": best_rank is not None and best_rank <= 20,
        "hit_at_100": best_rank is not None and best_rank <= 100,
        "top5": [
            {
                "rank": p["rank"],
                "ref": f"{p['book']} {p['chapter']}:{p['verse']}",
                "cosine": round(p["cosine_score"], 4),
            }
            for p in pool[:5]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen query-encode mode A/B on NZ dense")
    parser.add_argument("--corpus", default="dk", help="Corpus id (default: dk)")
    parser.add_argument(
        "--pool-size",
        type=int,
        default=500,
        help="Dense pool size for rank lookup (default: 500)",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=list(QUERY_ENCODE_MODES),
        choices=list(QUERY_ENCODE_MODES),
        help="Encode modes to compare",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"query_encode_diag_{stamp}.json"

    rows: list[dict] = []
    print(f"Corpus={args.corpus}  pool_size={args.pool_size}  modes={args.modes}")
    print("-" * 72)

    for case in DEFAULT_CASES:
        print(f"\n[{case['id']}] {case['query']}")
        for mode in args.modes:
            row = _evaluate(case, args.corpus, mode, args.pool_size)
            rows.append(row)
            br = row["best_rank"]
            br_s = f"#{br}" if br is not None else "absent"
            flags = []
            if row["hit_at_20"]:
                flags.append("@20")
            elif row["hit_at_100"]:
                flags.append("@100")
            flag_s = ",".join(flags) if flags else "miss"
            print(f"  {mode:5s}  best={br_s:8s}  {flag_s:6s}  top1={row['top5'][0]['ref'] if row['top5'] else '—'}")

    # Summary table: best_rank by case × mode
    print("\n" + "=" * 72)
    print("Summary (best dense rank of expected anchor)")
    header = f"{'case':28s}" + "".join(f"{m:>10s}" for m in args.modes)
    print(header)
    by_case: dict[str, dict[str, int | None]] = {}
    for row in rows:
        by_case.setdefault(row["id"], {})[row["encode_mode"]] = row["best_rank"]
    for case_id, modes in by_case.items():
        cells = "".join(
            f"{('—' if modes.get(m) is None else modes[m]):>10}" for m in args.modes
        )
        print(f"{case_id:28s}{cells}")

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "corpus": args.corpus,
        "pool_size": args.pool_size,
        "modes": args.modes,
        "cases": rows,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
