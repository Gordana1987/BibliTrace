"""
Baseline eval over golden JSON files in data/benchmark/.

Stores full top-k matches per case (default 20) in results JSON.
Declarative fields (diagnosis, probe, derived_from) stay in golden files.

Run from backend/:
  python scripts/run_golden_set.py
  python scripts/run_golden_set.py --golden data/benchmark/golden_random.json
  python scripts/run_golden_set.py --top-k 20
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from config import DATA_DIR
from models.schemas import AnalyzeRequest
from services.detection import detect

GOLDEN_PATH = DATA_DIR / "benchmark" / "golden_set.json"
GOLDEN_RANDOM_PATH = DATA_DIR / "benchmark" / "golden_random.json"
RESULTS_DIR = DATA_DIR / "benchmark" / "results"
BIBLE_CSV = DATA_DIR / "dk" / "bible.csv"


def _ref_key(book: str, chapter: int, verse: int) -> tuple[str, int, int]:
    return (book.strip(), int(chapter), int(verse))


def _load_golden(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _validate_expected_refs(cases: list[dict]) -> list[str]:
    """Return list of validation errors (empty = OK)."""
    df = pd.read_csv(BIBLE_CSV)
    refs = {
        _ref_key(row.book, row.chapter, row.verse)
        for _, row in df.iterrows()
    }
    errors: list[str] = []
    for case in cases:
        for exp in case.get("expected", []):
            key = _ref_key(exp["book"], exp["chapter"], exp["verse"])
            if key not in refs:
                errors.append(
                    f"{case['id']}: missing in corpus {key[0]} {key[1]}:{key[2]}"
                )
    return errors


def _should_skip(case: dict, skip_prefix: str) -> bool:
    return case.get("query", "").strip().startswith(skip_prefix)


def _chapter_key(book: str, chapter: int) -> tuple[str, int]:
    return (book.strip(), int(chapter))


def _evaluate_case(
    case: dict,
    corpora: list[str],
    top_k: int,
) -> dict:
    request = AnalyzeRequest(text=case["query"], corpora=corpora)
    response = detect(request, compare_with_labse=False)
    matches = response.matches[:top_k]

    expected_keys = [
        _ref_key(e["book"], e["chapter"], e["verse"]) for e in case["expected"]
    ]
    found: dict[tuple[str, int, int], dict] = {}

    for rank, m in enumerate(matches, start=1):
        key = _ref_key(m.bible_ref.book, m.bible_ref.chapter, m.bible_ref.verse)
        if key in expected_keys and key not in found:
            found[key] = {
                "rank": rank,
                "confidence_type": m.confidence_type.value,
                "score": m.score,
                "match_corpus": m.corpus,
            }

    hits_detail = []
    for exp in case["expected"]:
        key = _ref_key(exp["book"], exp["chapter"], exp["verse"])
        hit = found.get(key)
        hits_detail.append(
            {
                "book": exp["book"],
                "chapter": exp["chapter"],
                "verse": exp["verse"],
                "hit": hit is not None,
                "rank": hit["rank"] if hit else None,
                "confidence_type": hit["confidence_type"] if hit else None,
                "score": hit["score"] if hit else None,
                "match_corpus": hit["match_corpus"] if hit else None,
            }
        )

    n_expected = len(expected_keys)
    n_hit = sum(1 for h in hits_detail if h["hit"])
    match_mode = case.get("match", "all")
    if match_mode == "any":
        recall_loose = n_hit >= 1
        recall_strict = n_hit >= 1
    else:
        recall_loose = n_hit >= 1
        recall_strict = n_hit == n_expected

    ranks_hit = [h["rank"] for h in hits_detail if h["rank"] is not None]
    best_rank = min(ranks_hit) if ranks_hit else None

    expected_chapters: list[tuple[str, int]] = []
    seen_chapters: set[tuple[str, int]] = set()
    for exp in case["expected"]:
        ck = _chapter_key(exp["book"], exp["chapter"])
        if ck not in seen_chapters:
            seen_chapters.add(ck)
            expected_chapters.append(ck)

    chapter_found: dict[tuple[str, int], dict] = {}
    for rank, m in enumerate(matches, start=1):
        ck = _chapter_key(m.bible_ref.book, m.bible_ref.chapter)
        if ck in seen_chapters and ck not in chapter_found:
            chapter_found[ck] = {
                "rank": rank,
                "verse": m.bible_ref.verse,
                "confidence_type": m.confidence_type.value,
                "score": m.score,
                "match_corpus": m.corpus,
            }

    chapter_hits_detail = []
    for book, chapter in expected_chapters:
        ck = (book, chapter)
        hit = chapter_found.get(ck)
        chapter_hits_detail.append(
            {
                "book": book,
                "chapter": chapter,
                "hit": hit is not None,
                "rank": hit["rank"] if hit else None,
                "verse": hit["verse"] if hit else None,
                "confidence_type": hit["confidence_type"] if hit else None,
                "score": hit["score"] if hit else None,
                "match_corpus": hit["match_corpus"] if hit else None,
            }
        )

    n_chapters = len(expected_chapters)
    n_chapter_hit = sum(1 for h in chapter_hits_detail if h["hit"])
    if match_mode == "any":
        recall_chapter_loose = n_chapter_hit >= 1
        recall_chapter_strict = n_chapter_hit >= 1
    else:
        recall_chapter_loose = n_chapter_hit >= 1
        recall_chapter_strict = n_chapter_hit == n_chapters

    chapter_ranks_hit = [h["rank"] for h in chapter_hits_detail if h["rank"] is not None]
    best_chapter_rank = min(chapter_ranks_hit) if chapter_ranks_hit else None

    return {
        "id": case["id"],
        "layer": case.get("layer"),
        "probe": case.get("probe"),
        "derived_from": case.get("derived_from"),
        "match_mode": match_mode,
        "recall_loose": recall_loose,
        "recall_strict": recall_strict,
        "recall_chapter_loose": recall_chapter_loose,
        "recall_chapter_strict": recall_chapter_strict,
        "best_rank": best_rank,
        "best_chapter_rank": best_chapter_rank,
        "result_count": len(matches),
        "message": response.message,
        "hits": hits_detail,
        "chapter_hits": chapter_hits_detail,
        "top_matches": [
            {
                "rank": i + 1,
                "book": m.bible_ref.book,
                "chapter": m.bible_ref.chapter,
                "verse": m.bible_ref.verse,
                "confidence_type": m.confidence_type.value,
                "score": m.score,
                "corpus": m.corpus,
            }
            for i, m in enumerate(matches)
        ],
    }


def _aggregate(cases: list[dict]) -> dict:
    if not cases:
        return {
            "count": 0,
            "recall_loose": 0.0,
            "recall_strict": 0.0,
            "recall_chapter_loose": 0.0,
            "recall_chapter_strict": 0.0,
        }
    n = len(cases)
    return {
        "count": n,
        "recall_loose": round(sum(1 for c in cases if c["recall_loose"]) / n, 4),
        "recall_strict": round(sum(1 for c in cases if c["recall_strict"]) / n, 4),
        "recall_chapter_loose": round(
            sum(1 for c in cases if c["recall_chapter_loose"]) / n, 4
        ),
        "recall_chapter_strict": round(
            sum(1 for c in cases if c["recall_chapter_strict"]) / n, 4
        ),
        "avg_result_count": round(sum(c["result_count"] for c in cases) / n, 2),
        "under_20_results": sum(1 for c in cases if c["result_count"] < 20),
    }


def _group_aggregate(case_results: list[dict], key_fn) -> dict:
    groups: dict[str, list] = defaultdict(list)
    for cr in case_results:
        groups[key_fn(cr) or "unknown"].append(cr)
    return {k: _aggregate(v) for k, v in sorted(groups.items())}


def _case_status(c: dict) -> str:
    if c["recall_strict"]:
        return "HIT"
    if c["recall_loose"]:
        return "PART"
    return "MISS"


def _chapter_status(c: dict) -> str:
    if c["recall_chapter_strict"]:
        return "CH-HIT"
    if c["recall_chapter_loose"]:
        return "CH-PART"
    return "CH-MISS"


def _print_pass_report(
    pass_name: str,
    pass_result: dict,
    case_by_id: dict[str, dict],
    *,
    set_id: str,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"PASS: {pass_name}  corpora={pass_result['corpora']}  set={set_id}")
    print(f"{'=' * 60}")
    agg = pass_result["aggregate"]
    print(
        f"Overall: verse_loose={agg['recall_loose']:.0%}  "
        f"verse_strict={agg['recall_strict']:.0%}  "
        f"chapter_loose={agg['recall_chapter_loose']:.0%}  "
        f"chapter_strict={agg['recall_chapter_strict']:.0%}  "
        f"avg_results={agg['avg_result_count']}  "
        f"under_20={agg['under_20_results']}"
    )
    for layer, layer_agg in pass_result.get("by_layer", {}).items():
        print(
            f"  [layer:{layer}] n={layer_agg['count']}  "
            f"v_loose={layer_agg['recall_loose']:.0%}  "
            f"v_strict={layer_agg['recall_strict']:.0%}  "
            f"ch_loose={layer_agg['recall_chapter_loose']:.0%}  "
            f"ch_strict={layer_agg['recall_chapter_strict']:.0%}"
        )
    for probe, probe_agg in pass_result.get("by_probe", {}).items():
        print(
            f"  [probe:{probe}] n={probe_agg['count']}  "
            f"v_loose={probe_agg['recall_loose']:.0%}  "
            f"v_strict={probe_agg['recall_strict']:.0%}  "
            f"ch_loose={probe_agg['recall_chapter_loose']:.0%}  "
            f"ch_strict={probe_agg['recall_chapter_strict']:.0%}"
        )

    print("\nPer case:")
    for c in pass_result["cases"]:
        golden = case_by_id.get(c["id"], {})
        status = _case_status(c)
        ch_status = _chapter_status(c)
        rank = f"rank={c['best_rank']}" if c["best_rank"] else "rank=-"
        ch_rank = (
            f"ch_rank={c['best_chapter_rank']}" if c["best_chapter_rank"] else "ch_rank=-"
        )
        types = ",".join(
            h["confidence_type"] or "-"
            for h in c["hits"]
            if h["hit"]
        ) or "-"
        extras = []
        if set_id == "main":
            diagnosis = golden.get("diagnosis")
            if diagnosis and not c["recall_strict"]:
                extras.append(f"diagnosis={diagnosis}")
        probe = golden.get("probe") or c.get("probe")
        if probe:
            extras.append(f"probe={probe}")
        derived = golden.get("derived_from") or c.get("derived_from")
        if derived:
            extras.append(f"from={derived}")
        extra_suffix = ("  " + "  ".join(extras)) if extras else ""
        print(
            f"  {status:4}  {ch_status:7}  {c['id']:40}  {rank:8}  {ch_rank:10}  "
            f"results={c['result_count']:2}  types={types}  "
            f"layer={c['layer']}  match={c.get('match_mode', 'all')}{extra_suffix}"
        )


def run_eval(golden_path: Path, top_k: int) -> Path:
    golden = _load_golden(golden_path)
    benchmark = golden.get("benchmark", {})
    set_id = benchmark.get("set_id", golden_path.stem.replace("golden_", "") or "main")
    skip_prefix = benchmark.get("skip_query_prefix", "TODO")
    corpus_passes = benchmark.get(
        "corpus_passes", [["dk_ekav"], ["dk"], ["dk", "dk_ekav"]]
    )
    all_cases = golden["cases"]

    print(f"\n{'#' * 60}")
    print(f"GOLDEN SET: {golden_path.name}  set_id={set_id}  version={golden.get('version')}")
    print(f"{'#' * 60}")

    print("Validating expected refs against data/dk/bible.csv ...")
    errors = _validate_expected_refs(all_cases)
    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)
    print("All expected refs OK.")

    runnable = [c for c in all_cases if not _should_skip(c, skip_prefix)]
    skipped = [c["id"] for c in all_cases if _should_skip(c, skip_prefix)]
    case_by_id = {c["id"]: c for c in all_cases}
    print(f"Runnable cases: {len(runnable)}  Skipped (TODO): {skipped}")

    report = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "golden_file": str(golden_path.relative_to(BASE_DIR)),
        "set_id": set_id,
        "golden_version": golden.get("version"),
        "mode": benchmark.get("mode", "baseline"),
        "top_k": top_k,
        "skipped": skipped,
        "passes": [],
    }

    pass_labels = ["ekav_only", "ijekav_only", "both"]

    for label, corpora in zip(pass_labels, corpus_passes):
        print(f"\nRunning pass {label} {corpora} ...")
        case_results = []
        for i, case in enumerate(runnable, start=1):
            print(f"  [{i}/{len(runnable)}] {case['id']} ...", flush=True)
            case_results.append(_evaluate_case(case, corpora, top_k))

        pass_result = {
            "name": label,
            "corpora": corpora,
            "aggregate": _aggregate(case_results),
            "by_layer": _group_aggregate(case_results, lambda c: c.get("layer")),
            "by_probe": _group_aggregate(
                case_results,
                lambda c: case_by_id.get(c["id"], {}).get("probe") or c.get("probe"),
            ),
            "cases": case_results,
        }
        report["passes"].append(pass_result)
        _print_pass_report(label, pass_result, case_by_id, set_id=set_id)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"baseline_{set_id}_{stamp}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run golden set baseline eval.")
    parser.add_argument("--top-k", type=int, default=20, help="Max ranks to consider.")
    parser.add_argument(
        "--golden",
        type=Path,
        default=None,
        help="Path to golden JSON (default: run main + random separately).",
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Run main golden_set.json then golden_random.json (default when --golden omitted).",
    )
    args = parser.parse_args()

    if args.golden is not None:
        run_eval(args.golden.resolve(), args.top_k)
        return

    paths = [GOLDEN_PATH, GOLDEN_RANDOM_PATH]
    for path in paths:
        if not path.exists():
            print(f"Missing {path}")
            raise SystemExit(1)
        run_eval(path, args.top_k)


if __name__ == "__main__":
    main()
