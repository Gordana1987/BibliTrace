"""
Dense full-corpus diagnostic (Klod hybrid test, step 1+2).

1. Measure Qwen3 query encode + full-matrix dot + argsort latency (ms).
2. Rank expected anchors in dense top-200 (no BM25 filter) for Phase-A-absent cases
   plus a small control set.

Run from backend/:
  python scripts/run_dense_full_corpus_diag.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from config import DATA_DIR
from services.detection import (
    _BM25_CANDIDATES,
    dense_retrieval_timings,
    get_dense_ranked_pool,
)

GOLDEN_PATH = DATA_DIR / "benchmark" / "golden_set.json"
PHASE_A_PATH = DATA_DIR / "benchmark" / "results" / "phase_a_bm25_main_20260714_171219.json"
RESULTS_DIR = DATA_DIR / "benchmark" / "results"

CONTROL_IDS = [
    "rastko-danseti-postanje1",
    "desanka-pomilovanje",
    "lalic-kanoni-more",
    "njegos-gorski-vijenac-getsemanija",
    "kostic-samson-i-delila-astarota-stub",
]

LATENCY_SAMPLE_ID = "jefimija-zmija-venac"


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


def _evaluate_case(
    case: dict,
    corpus: str,
    pool_size: int,
    *,
    group: str,
    phase_a_by_id: dict[str, dict] | None = None,
) -> dict:
    pool = get_dense_ranked_pool(case["query"], corpus=corpus, pool_size=pool_size)
    by_ref = _rank_lookup(pool)

    phase_a = (phase_a_by_id or {}).get(case["id"])
    phase_a_hits = {}
    if phase_a:
        for h in phase_a.get("expected_hits", []):
            key = _ref_key(h["book"], h["chapter"], h["verse"])
            phase_a_hits[key] = h

    expected_hits = []
    for exp in case.get("expected", []):
        key = _ref_key(exp["book"], exp["chapter"], exp["verse"])
        hit = by_ref.get(key)
        pa = phase_a_hits.get(key)
        expected_hits.append(
            {
                "book": exp["book"],
                "chapter": exp["chapter"],
                "verse": exp["verse"],
                "dense_status": "in_dense_top_k" if hit else "absent_from_dense_top_k",
                "rank_in_dense": hit["rank"] if hit else None,
                "cosine_score": hit["cosine_score"] if hit else None,
                "bm25_rank_in_pool": pa.get("rank_in_pool") if pa else None,
                "bm25_status": pa.get("pool_status") if pa else None,
            }
        )

    ranks = [h["rank_in_dense"] for h in expected_hits if h["rank_in_dense"] is not None]
    best_rank = min(ranks) if ranks else None
    any_hit = any(h["dense_status"] == "in_dense_top_k" for h in expected_hits)
    all_hit = all(h["dense_status"] == "in_dense_top_k" for h in expected_hits)
    match_mode = case.get("match", "all")
    if match_mode == "any":
        dense_recall_loose = dense_recall_strict = any_hit
    else:
        dense_recall_loose = any_hit
        dense_recall_strict = all_hit

    return {
        "id": case["id"],
        "group": group,
        "layer": case.get("layer"),
        "probe": case.get("probe"),
        "diagnosis": case.get("diagnosis"),
        "match_mode": match_mode,
        "dense_pool_size": len(pool),
        "dense_recall_loose": dense_recall_loose,
        "dense_recall_strict": dense_recall_strict,
        "best_rank_in_dense": best_rank,
        "phase_a_best_rank_in_bm25": phase_a.get("best_rank_in_pool") if phase_a else None,
        "expected_hits": expected_hits,
        "top_dense": [
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


def _print_latency(timings: dict) -> None:
    shape = timings["matrix_shape"]
    print(f"Matrix: {shape[0]} verses × {shape[1]} dims  (repeats={timings['repeats']})")
    for step in ("query_encode", "full_matrix_dot", "argsort_top_k"):
        s = timings[step]
        print(f"  {step:18}  mean={s['mean_ms']:.3f} ms  min={s['min_ms']:.3f}  max={s['max_ms']:.3f}")


def _print_cases(case_results: list[dict]) -> None:
    for c in case_results:
        if c["best_rank_in_dense"] is None:
            rank_str = "absent_from_dense"
        else:
            rank_str = f"dense_rank={c['best_rank_in_dense']}"
        anchors = []
        for h in c["expected_hits"]:
            if h["dense_status"] == "absent_from_dense_top_k":
                anchors.append(f"{h['book']} {h['chapter']}:{h['verse']}=absent")
            else:
                anchors.append(
                    f"{h['book']} {h['chapter']}:{h['verse']}=#{h['rank_in_dense']}"
                    f"(cos={h['cosine_score']:.3f})"
                )
        print(
            f"  [{c['group']}] {c['id']:42}  {rank_str:20}  "
            f"bm25_phase_a={c['phase_a_best_rank_in_bm25']}  {' | '.join(anchors)}"
        )


def run_diag(
    golden_path: Path,
    phase_a_path: Path,
    corpus: str,
    pool_size: int,
    repeats: int,
) -> Path:
    golden = _load_json(golden_path)
    phase_a = _load_json(phase_a_path)
    phase_a_by_id = {c["id"]: c for c in phase_a["cases"]}
    case_by_id = {c["id"]: c for c in golden["cases"]}

    absent_ids = [c["id"] for c in phase_a["cases"] if c["best_rank_in_pool"] is None]
    missing = [cid for cid in absent_ids if cid not in case_by_id]
    if missing:
        raise SystemExit(f"Golden missing Phase-A absent ids: {missing}")

    print(f"\n{'#' * 60}")
    print("DENSE FULL-CORPUS DIAGNOSTIC")
    print(f"{'#' * 60}")

    latency_case = case_by_id[LATENCY_SAMPLE_ID]
    print(f"\nLatency sample query: {LATENCY_SAMPLE_ID}")
    timings = dense_retrieval_timings(latency_case["query"], corpus=corpus, repeats=repeats)
    _print_latency(timings)

    absent_results = []
    print(f"\nPhase-A absent cases ({len(absent_ids)}):")
    for cid in absent_ids:
        print(f"  evaluating {cid} ...", flush=True)
        absent_results.append(
            _evaluate_case(
                case_by_id[cid],
                corpus,
                pool_size,
                group="phase_a_absent",
                phase_a_by_id=phase_a_by_id,
            )
        )
    _print_cases(absent_results)

    control_results = []
    print(f"\nControl cases ({len(CONTROL_IDS)}):")
    for cid in CONTROL_IDS:
        print(f"  evaluating {cid} ...", flush=True)
        control_results.append(
            _evaluate_case(
                case_by_id[cid],
                corpus,
                pool_size,
                group="control",
                phase_a_by_id=phase_a_by_id,
            )
        )
    _print_cases(control_results)

    absent_anchor_hits = sum(
        1
        for c in absent_results
        for h in c["expected_hits"]
        if h["dense_status"] == "in_dense_top_k"
    )
    absent_anchor_total = sum(len(c["expected_hits"]) for c in absent_results)
    absent_case_any = sum(1 for c in absent_results if c["dense_recall_loose"])

    print(f"\nSummary (absent group):")
    print(f"  anchors in dense top-{pool_size}: {absent_anchor_hits}/{absent_anchor_total}")
    print(f"  cases with any anchor hit: {absent_case_any}/{len(absent_results)}")

    report = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "mode": "dense_full_corpus_diag",
        "corpus": corpus,
        "pool_size": pool_size,
        "latency": timings,
        "summary": {
            "absent_cases": len(absent_results),
            "absent_anchors_in_dense": absent_anchor_hits,
            "absent_anchors_total": absent_anchor_total,
            "absent_cases_any_hit": absent_case_any,
        },
        "absent_case_ids": absent_ids,
        "control_case_ids": CONTROL_IDS,
        "cases": absent_results + control_results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"dense_full_corpus_diag_{stamp}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Dense full-corpus latency + absent-case diag.")
    parser.add_argument("--golden", type=Path, default=GOLDEN_PATH)
    parser.add_argument("--phase-a", type=Path, default=PHASE_A_PATH)
    parser.add_argument("--corpus", default="dk")
    parser.add_argument("--pool-size", type=int, default=_BM25_CANDIDATES)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    run_diag(
        args.golden.resolve(),
        args.phase_a.resolve(),
        args.corpus,
        args.pool_size,
        args.repeats,
    )


if __name__ == "__main__":
    main()
