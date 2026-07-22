"""
HyDE dense full-corpus diagnostic.

PAUSED for live pipeline (2026-07): kept for future A/B only.
Encodes hyde_query through Qwen3 dense retrieval over the full embedding matrix.

Run from backend/ (venv active):
  python ../archive/literary-text-search/scripts/run_hyde_dense_diag.py
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
from services.detection import _BM25_CANDIDATES, get_dense_ranked_pool

BENCH_DIR = ARCHIVE_DIR / "benchmark"
HYDE_PATH = BENCH_DIR / "hyde_dense_cases.json"
BASELINE_PATH = BENCH_DIR / "results" / "dense_full_corpus_diag_20260714_175630.json"
PREV_HYDE_PATH = BENCH_DIR / "results" / "hyde_dense_diag_20260714_182529.json"
RESULTS_DIR = BENCH_DIR / "results"


def _ref_key(book: str, chapter: int, verse: int) -> tuple[str, int, int]:
    return (book.strip(), int(chapter), int(verse))


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _rank_lookup(pool: list[dict]) -> dict[tuple[str, int, int], dict]:
    return {
        _ref_key(item["book"], item["chapter"], item["verse"]): item
        for item in pool
    }


def _baseline_hits_by_id(baseline: dict) -> dict[str, dict[tuple[str, int, int], dict]]:
    out: dict[str, dict[tuple[str, int, int], dict]] = {}
    for case in baseline["cases"]:
        if case.get("group") != "phase_a_absent":
            continue
        out[case["id"]] = {
            _ref_key(h["book"], h["chapter"], h["verse"]): h
            for h in case["expected_hits"]
        }
    return out


def _prev_hyde_hits_by_id(prev: dict) -> dict[str, dict[tuple[str, int, int], dict]]:
    return {
        case["id"]: {
            _ref_key(h["book"], h["chapter"], h["verse"]): h
            for h in case["expected_hits"]
        }
        for case in prev.get("cases", [])
    }


def _evaluate_hyde_case(
    case: dict,
    corpus: str,
    pool_size: int,
    baseline_hits: dict[tuple[str, int, int], dict],
    prev_hyde_hits: dict[tuple[str, int, int], dict] | None = None,
) -> dict:
    pool = get_dense_ranked_pool(case["hyde_query"], corpus=corpus, pool_size=pool_size)
    by_ref = _rank_lookup(pool)

    expected_hits = []
    for exp in case["expected"]:
        key = _ref_key(exp["book"], exp["chapter"], exp["verse"])
        hit = by_ref.get(key)
        base = baseline_hits.get(key)
        prev = (prev_hyde_hits or {}).get(key)
        base_rank = base.get("rank_in_dense") if base else None
        prev_rank = prev.get("hyde_rank_in_dense") if prev else None
        hyde_rank = hit["rank"] if hit else None
        if base_rank is None and hyde_rank is None:
            delta = None
            outcome = "both_absent"
        elif base_rank is None and hyde_rank is not None:
            delta = "new_hit"
            outcome = "hyde_gain"
        elif base_rank is not None and hyde_rank is None:
            delta = "lost_hit"
            outcome = "hyde_regression"
        elif hyde_rank < base_rank:
            delta = base_rank - hyde_rank
            outcome = "hyde_better"
        elif hyde_rank > base_rank:
            delta = hyde_rank - base_rank
            outcome = "hyde_worse"
        else:
            delta = 0
            outcome = "same_rank"

        expected_hits.append(
            {
                "book": exp["book"],
                "chapter": exp["chapter"],
                "verse": exp["verse"],
                "baseline_rank_in_dense": base_rank,
                "baseline_dense_status": base.get("dense_status") if base else None,
                "baseline_cosine": base.get("cosine_score") if base else None,
                "prev_hyde_rank_in_dense": prev_rank,
                "prev_hyde_cosine": prev.get("hyde_cosine") if prev else None,
                "hyde_rank_in_dense": hyde_rank,
                "hyde_dense_status": "in_dense_top_k" if hit else "absent_from_dense_top_k",
                "hyde_cosine": hit["cosine_score"] if hit else None,
                "comparison": outcome,
                "rank_delta": delta,
            }
        )

    ranks = [h["hyde_rank_in_dense"] for h in expected_hits if h["hyde_rank_in_dense"] is not None]
    best_rank = min(ranks) if ranks else None
    any_hit = any(h["hyde_dense_status"] == "in_dense_top_k" for h in expected_hits)
    all_hit = all(h["hyde_dense_status"] == "in_dense_top_k" for h in expected_hits)
    match_mode = case.get("match_mode", "all")
    if match_mode == "any":
        hyde_recall_loose = hyde_recall_strict = any_hit
    else:
        hyde_recall_loose = any_hit
        hyde_recall_strict = all_hit

    gains = sum(1 for h in expected_hits if h["comparison"] in ("hyde_gain", "hyde_better"))
    return {
        "id": case["id"],
        "derived_from": case["derived_from"],
        "probe": case.get("probe"),
        "match_mode": match_mode,
        "original_query": case["original_query"],
        "hyde_query": case["hyde_query"],
        "note": case.get("note"),
        "hyde_recall_loose": hyde_recall_loose,
        "hyde_recall_strict": hyde_recall_strict,
        "best_rank_in_dense_hyde": best_rank,
        "anchors_improved": gains,
        "expected_hits": expected_hits,
        "top_dense_hyde": [
            {
                "rank": p["rank"],
                "book": p["book"],
                "chapter": p["chapter"],
                "verse": p["verse"],
                "cosine_score": round(p["cosine_score"], 4),
            }
            for p in pool[:10]
        ],
    }


def _print_report(case_results: list[dict]) -> None:
    print("\nPer anchor (raw baseline → HyDE v2):")
    for c in case_results:
        print(f"\n  {c['id']}  (from {c['derived_from']})")
        for h in c["expected_hits"]:
            base = h["baseline_rank_in_dense"]
            prev = h.get("prev_hyde_rank_in_dense")
            hyde = h["hyde_rank_in_dense"]
            base_s = "absent" if base is None else f"#{base}"
            prev_s = "—" if prev is None else f"#{prev}"
            hyde_s = "absent" if hyde is None else f"#{hyde}"
            print(
                f"    {h['book']} {h['chapter']}:{h['verse']}  "
                f"raw={base_s}  hyde_v1={prev_s}  hyde_v2={hyde_s}  [{h['comparison']}]"
            )

    gains = sum(c["anchors_improved"] for c in case_results)
    new_hits = sum(
        1 for c in case_results for h in c["expected_hits"] if h["comparison"] == "hyde_gain"
    )
    better = sum(
        1 for c in case_results for h in c["expected_hits"] if h["comparison"] == "hyde_better"
    )
    same = sum(
        1 for c in case_results for h in c["expected_hits"] if h["comparison"] == "same_rank"
    )
    worse = sum(
        1 for c in case_results for h in c["expected_hits"] if h["comparison"] == "hyde_worse"
    )
    both_absent = sum(
        1 for c in case_results for h in c["expected_hits"] if h["comparison"] == "both_absent"
    )
    print(
        f"\nSummary: new_hit={new_hits}  better={better}  same={same}  "
        f"worse={worse}  both_absent={both_absent}  anchors_improved={gains}"
    )


def run_diag(
    hyde_path: Path,
    baseline_path: Path,
    corpus: str,
    pool_size: int,
    prev_hyde_path: Path | None = None,
) -> Path:
    hyde_data = _load_json(hyde_path)
    baseline = _load_json(baseline_path)
    baseline_by_id = _baseline_hits_by_id(baseline)
    prev_by_id = {}
    if prev_hyde_path and prev_hyde_path.exists():
        prev_by_id = _prev_hyde_hits_by_id(_load_json(prev_hyde_path))
    cases = hyde_data["cases"]

    print(f"\n{'#' * 60}")
    print(f"HyDE DENSE DIAGNOSTIC v{hyde_data.get('version', '?')}  corpus={corpus}  pool={pool_size}")
    print(f"Baseline: {baseline_path.name}")
    if prev_hyde_path and prev_hyde_path.exists():
        print(f"Prev HyDE: {prev_hyde_path.name}")
    print(f"{'#' * 60}")

    case_results = []
    for i, case in enumerate(cases, start=1):
        derived = case["derived_from"]
        if derived not in baseline_by_id:
            raise SystemExit(f"No baseline absent-case entry for derived_from={derived}")
        print(f"  [{i}/{len(cases)}] {case['id']} ...", flush=True)
        case_results.append(
            _evaluate_hyde_case(
                case,
                corpus,
                pool_size,
                baseline_by_id[derived],
                prev_by_id.get(case["id"]),
            )
        )

    _print_report(case_results)

    report = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "mode": "hyde_dense_full_corpus_diag",
        "corpus": corpus,
        "pool_size": pool_size,
        "baseline_file": str(baseline_path.relative_to(BASE_DIR)),
        "hyde_file": str(hyde_path.relative_to(BASE_DIR)),
        "hyde_version": hyde_data.get("version"),
        "prev_hyde_file": str(prev_hyde_path.relative_to(BASE_DIR)) if prev_hyde_path and prev_hyde_path.exists() else None,
        "summary": {
            "cases": len(case_results),
            "anchors_new_hit": sum(
                1 for c in case_results for h in c["expected_hits"] if h["comparison"] == "hyde_gain"
            ),
            "anchors_better": sum(
                1 for c in case_results for h in c["expected_hits"] if h["comparison"] == "hyde_better"
            ),
            "anchors_same": sum(
                1 for c in case_results for h in c["expected_hits"] if h["comparison"] == "same_rank"
            ),
            "anchors_worse": sum(
                1 for c in case_results for h in c["expected_hits"] if h["comparison"] == "hyde_worse"
            ),
            "anchors_both_absent": sum(
                1 for c in case_results for h in c["expected_hits"] if h["comparison"] == "both_absent"
            ),
        },
        "cases": case_results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"hyde_dense_diag_{stamp}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="HyDE dense retrieval diagnostic.")
    parser.add_argument("--hyde", type=Path, default=HYDE_PATH)
    parser.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    parser.add_argument("--prev-hyde", type=Path, default=PREV_HYDE_PATH)
    parser.add_argument("--corpus", default="dk")
    parser.add_argument("--pool-size", type=int, default=_BM25_CANDIDATES)
    args = parser.parse_args()
    prev = args.prev_hyde.resolve() if args.prev_hyde else None
    run_diag(args.hyde.resolve(), args.baseline.resolve(), args.corpus, args.pool_size, prev)


if __name__ == "__main__":
    main()
