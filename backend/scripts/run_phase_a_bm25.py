"""
Phase A benchmark: BM25 + phrase match only (no embedding / cross-encoder rerank).

Reports rank of each expected verse in the full candidate pool (default 200).
Verses not in the pool are marked pool_status=absent_from_pool (not rank=null alone).

Run from backend/:
  python scripts/run_phase_a_bm25.py
  python scripts/run_phase_a_bm25.py --golden data/benchmark/golden_set.json --corpus dk
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from config import DATA_DIR
from services.detection import _BM25_CANDIDATES, get_bm25_ranked_pool

GOLDEN_PATH = DATA_DIR / "benchmark" / "golden_set.json"
RESULTS_DIR = DATA_DIR / "benchmark" / "results"
BIBLE_CSV = DATA_DIR / "dk" / "bible.csv"


def _ref_key(book: str, chapter: int, verse: int) -> tuple[str, int, int]:
    return (book.strip(), int(chapter), int(verse))


def _load_golden(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _validate_expected_refs(cases: list[dict]) -> list[str]:
    df = pd.read_csv(BIBLE_CSV)
    refs = {_ref_key(row.book, row.chapter, row.verse) for _, row in df.iterrows()}
    errors: list[str] = []
    for case in cases:
        for exp in case.get("expected", []):
            key = _ref_key(exp["book"], exp["chapter"], exp["verse"])
            if key not in refs:
                errors.append(f"{case['id']}: missing in corpus {key[0]} {key[1]}:{key[2]}")
    return errors


def _rank_lookup(pool: list[dict]) -> dict[tuple[str, int, int], dict]:
    return {
        _ref_key(item["book"], item["chapter"], item["verse"]): item
        for item in pool
    }


def _evaluate_case(case: dict, corpus: str, pool_size: int) -> dict:
    pool = get_bm25_ranked_pool(case["query"], corpus=corpus, pool_size=pool_size)
    by_ref = _rank_lookup(pool)

    expected_hits = []
    for exp in case.get("expected", []):
        key = _ref_key(exp["book"], exp["chapter"], exp["verse"])
        hit = by_ref.get(key)
        if hit:
            expected_hits.append(
                {
                    "book": exp["book"],
                    "chapter": exp["chapter"],
                    "verse": exp["verse"],
                    "pool_status": "in_pool",
                    "rank_in_pool": hit["rank"],
                    "is_phrase": hit["is_phrase"],
                    "bm25_score": hit["bm25_score"],
                }
            )
        else:
            expected_hits.append(
                {
                    "book": exp["book"],
                    "chapter": exp["chapter"],
                    "verse": exp["verse"],
                    "pool_status": "absent_from_pool",
                    "rank_in_pool": None,
                    "is_phrase": None,
                    "bm25_score": None,
                }
            )

    ranks_in_pool = [h["rank_in_pool"] for h in expected_hits if h["pool_status"] == "in_pool"]
    best_rank_in_pool = min(ranks_in_pool) if ranks_in_pool else None
    all_in_pool = all(h["pool_status"] == "in_pool" for h in expected_hits)
    any_in_pool = any(h["pool_status"] == "in_pool" for h in expected_hits)
    match_mode = case.get("match", "all")
    if match_mode == "any":
        pool_recall_loose = any_in_pool
        pool_recall_strict = any_in_pool
    else:
        pool_recall_loose = any_in_pool
        pool_recall_strict = all_in_pool

    return {
        "id": case["id"],
        "layer": case.get("layer"),
        "probe": case.get("probe"),
        "diagnosis": case.get("diagnosis"),
        "derived_from": case.get("derived_from"),
        "match_mode": match_mode,
        "pool_size": len(pool),
        "pool_capacity": pool_size,
        "pool_recall_loose": pool_recall_loose,
        "pool_recall_strict": pool_recall_strict,
        "best_rank_in_pool": best_rank_in_pool,
        "expected_hits": expected_hits,
        "top_pool": [
            {
                "rank": item["rank"],
                "book": item["book"],
                "chapter": item["chapter"],
                "verse": item["verse"],
                "is_phrase": item["is_phrase"],
                "bm25_score": item["bm25_score"],
            }
            for item in pool[:20]
        ],
    }


def _aggregate(cases: list[dict]) -> dict:
    if not cases:
        return {"count": 0, "pool_recall_loose": 0.0, "pool_recall_strict": 0.0}
    n = len(cases)
    return {
        "count": n,
        "pool_recall_loose": round(sum(1 for c in cases if c["pool_recall_loose"]) / n, 4),
        "pool_recall_strict": round(sum(1 for c in cases if c["pool_recall_strict"]) / n, 4),
        "absent_from_pool": sum(
            1
            for c in cases
            if all(h["pool_status"] == "absent_from_pool" for h in c["expected_hits"])
        ),
        "in_pool_top20": sum(
            1 for c in cases if c["best_rank_in_pool"] is not None and c["best_rank_in_pool"] <= 20
        ),
    }


def _print_report(case_results: list[dict], case_by_id: dict[str, dict]) -> None:
    agg = _aggregate(case_results)
    print(
        f"Overall: in_pool_loose={agg['pool_recall_loose']:.0%}  "
        f"in_pool_strict={agg['pool_recall_strict']:.0%}  "
        f"all_absent={agg['absent_from_pool']}  "
        f"best_in_top20={agg['in_pool_top20']}"
    )
    print("\nPer case (expected verse rank in BM25 pool):")
    for c in case_results:
        golden = case_by_id.get(c["id"], {})
        if c["best_rank_in_pool"] is None:
            rank_str = "absent_from_pool"
        else:
            rank_str = f"rank={c['best_rank_in_pool']}"
        extras = []
        if golden.get("diagnosis"):
            extras.append(f"diagnosis={golden['diagnosis']}")
        if c.get("probe"):
            extras.append(f"probe={c['probe']}")
        extra_suffix = ("  " + "  ".join(extras)) if extras else ""
        hits_detail = []
        for h in c["expected_hits"]:
            if h["pool_status"] == "absent_from_pool":
                hits_detail.append(f"{h['book']} {h['chapter']}:{h['verse']}=absent")
            else:
                hits_detail.append(f"{h['book']} {h['chapter']}:{h['verse']}=#{h['rank_in_pool']}")
        print(
            f"  {c['id']:42}  {rank_str:18}  pool={c['pool_size']:3}  "
            f"layer={c['layer']}  {' | '.join(hits_detail)}{extra_suffix}"
        )


def run_eval(golden_path: Path, corpus: str, pool_size: int) -> Path:
    golden = _load_golden(golden_path)
    benchmark = golden.get("benchmark", {})
    set_id = benchmark.get("set_id", golden_path.stem.replace("golden_", "") or "main")
    skip_prefix = benchmark.get("skip_query_prefix", "TODO")
    all_cases = golden["cases"]

    print(f"\n{'#' * 60}")
    print(f"PHASE A (BM25 only): {golden_path.name}  set_id={set_id}  corpus={corpus}")
    print(f"{'#' * 60}")

    print("Validating expected refs against data/dk/bible.csv ...")
    errors = _validate_expected_refs(all_cases)
    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)
    print("All expected refs OK.")

    runnable = [c for c in all_cases if not c.get("query", "").strip().startswith(skip_prefix)]
    skipped = [c["id"] for c in all_cases if c.get("query", "").strip().startswith(skip_prefix)]
    case_by_id = {c["id"]: c for c in all_cases}
    print(f"Runnable cases: {len(runnable)}  Skipped (TODO): {skipped}")

    case_results = []
    for i, case in enumerate(runnable, start=1):
        print(f"  [{i}/{len(runnable)}] {case['id']} ...", flush=True)
        case_results.append(_evaluate_case(case, corpus, pool_size))

    _print_report(case_results, case_by_id)

    report = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "golden_file": str(golden_path.relative_to(BASE_DIR)),
        "set_id": set_id,
        "golden_version": golden.get("version"),
        "mode": "phase_a_bm25_only",
        "corpus": corpus,
        "pool_capacity": pool_size,
        "skipped": skipped,
        "aggregate": _aggregate(case_results),
        "cases": case_results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"phase_a_bm25_{set_id}_{stamp}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase A: BM25-only golden set eval.")
    parser.add_argument("--golden", type=Path, default=GOLDEN_PATH)
    parser.add_argument("--corpus", default="dk")
    parser.add_argument("--pool-size", type=int, default=_BM25_CANDIDATES)
    args = parser.parse_args()
    run_eval(args.golden.resolve(), args.corpus, args.pool_size)


if __name__ == "__main__":
    main()
