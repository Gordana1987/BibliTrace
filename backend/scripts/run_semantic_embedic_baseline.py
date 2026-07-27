"""
Golden-set measurement with Embedić (e5 query:/passage: prefixes).

Requires indexes from build_embedic_nt_embeddings.py.
Compares against semantic_baseline_v2_prefix.json (Qwen3 + Bible Instruct).

Run from backend/:
  python scripts/run_semantic_embedic_baseline.py --model both
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

GOLDEN = BASE_DIR / "data" / "concept" / "semantic_golden_v1.json"
V2 = BASE_DIR / "data" / "concept" / "semantic_baseline_v2_prefix.json"
OUT = BASE_DIR / "data" / "concept" / "semantic_baseline_v4_embedic.json"

MODEL_IDS = {
    "base": "djovak/embedic-base",
    "large": "djovak/embedic-large",
}
TOP_GOOD = 20  # "good rank" threshold


def index_path(corpus: str, size: str) -> Path:
    return BASE_DIR / "data" / corpus / f"embedic_{size}_nt_embeddings.joblib"


def rank_verse(verses, scores, book, chapter, verse):
    order = np.argsort(-scores)
    rank_of = {int(i): r + 1 for r, i in enumerate(order)}
    for i in range(len(verses)):
        row = verses.iloc[i]
        if (
            str(row["book"]).strip() == book
            and int(row["chapter"]) == chapter
            and int(row["verse"]) == verse
        ):
            return rank_of[i], float(scores[i])
    return None, None


def top1(verses, scores):
    i = int(np.argmax(scores))
    row = verses.iloc[i]
    return {
        "book": str(row["book"]).strip(),
        "chapter": int(row["chapter"]),
        "verse": int(row["verse"]),
        "score": round(float(scores[i]), 4),
        "text": str(row["text"])[:120],
    }


def measure_model(size: str, golden: dict) -> dict:
    from sentence_transformers import SentenceTransformer

    model_id = MODEL_IDS[size]
    print(f"\n=== Measuring {model_id} ===")
    model = SentenceTransformer(model_id, device="cpu")

    indexes = {}
    for corpus in ("dk", "spc"):
        path = index_path(corpus, size)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run: python scripts/build_embedic_nt_embeddings.py "
                f"--model {size} --all"
            )
        indexes[corpus] = joblib.load(path)

    out = {
        "model_id": model_id,
        "size": size,
        "encode": "query: / passage:",
        "query_prefix": "query: ",
        "doc_prefix": "passage: ",
        "concepts": [],
    }

    for concept in golden["concepts"]:
        query = concept["query"]
        # Critical: query: prefix; keep Cyrillic as typed.
        q = model.encode(
            [f"query: {query}"],
            normalize_embeddings=True,
        )
        q = np.asarray(q, dtype=np.float32).reshape(-1)
        entry = {"query": query, "by_corpus": {}}
        for corpus in ("dk", "spc"):
            embs = np.asarray(indexes[corpus]["embeddings"], dtype=np.float32)
            verses = indexes[corpus]["verses"]
            scores = embs @ q
            corp_out = {"top1": top1(verses, scores), "expected": []}
            for exp in concept["expected"]:
                if exp.get("corpus", "both") not in ("both", corpus):
                    continue
                rank, score = rank_verse(
                    verses, scores, exp["book"], exp["chapter"], exp["verse"]
                )
                corp_out["expected"].append(
                    {
                        "book": exp["book"],
                        "chapter": exp["chapter"],
                        "verse": exp["verse"],
                        "layer": exp["layer"],
                        "rank": rank,
                        "score": None if score is None else round(score, 4),
                        "known_baseline_rank": exp.get("known_baseline_rank"),
                    }
                )
            entry["by_corpus"][corpus] = corp_out
        out["concepts"].append(entry)
        print(f"  ✓ {query}")
    return out


def best_rank(concept_entry: dict) -> int | None:
    """Best (min) rank across corpora/expected for aggregate stats."""
    ranks = []
    for data in concept_entry["by_corpus"].values():
        for e in data["expected"]:
            if e["rank"] is not None:
                ranks.append(e["rank"])
    return min(ranks) if ranks else None


def aggregate(model_block: dict, v2: dict) -> dict:
    v2m = {c["query"]: c for c in v2["concepts"]}
    n_good = 0
    n_good_v2 = 0
    weird_top1 = 0
    weird_top1_v2 = 0
    ranks_all = []
    ranks_v2 = []
    per_query = []

    # Heuristic "weird top1": top1 not among any expected for that query+corpus
    for c in model_block["concepts"]:
        q = c["query"]
        best = best_rank(c)
        if best is not None:
            ranks_all.append(best)
            if best <= TOP_GOOD:
                n_good += 1
        v2c = v2m.get(q)
        if v2c:
            best_v2 = best_rank(v2c)
            if best_v2 is not None:
                ranks_v2.append(best_v2)
                if best_v2 <= TOP_GOOD:
                    n_good_v2 += 1

        for corpus, data in c["by_corpus"].items():
            t1 = data["top1"]
            expected_keys = {
                (e["book"], e["chapter"], e["verse"]) for e in data["expected"]
            }
            if (t1["book"], t1["chapter"], t1["verse"]) not in expected_keys:
                weird_top1 += 1
            if v2c:
                t1v = v2c["by_corpus"][corpus]["top1"]
                exp_v = {
                    (e["book"], e["chapter"], e["verse"])
                    for e in v2c["by_corpus"][corpus]["expected"]
                }
                if (t1v["book"], t1v["chapter"], t1v["verse"]) not in exp_v:
                    weird_top1_v2 += 1

        per_query.append(
            {
                "query": q,
                "best_rank": best,
                "v2_best_rank": None if not v2c else best_rank(v2c),
            }
        )

    return {
        "top_good_threshold": TOP_GOOD,
        "concepts_with_best_rank_le_20": n_good,
        "v2_concepts_with_best_rank_le_20": n_good_v2,
        "n_concepts": len(model_block["concepts"]),
        "weird_top1_count": weird_top1,
        "v2_weird_top1_count": weird_top1_v2,
        "mean_best_rank": None if not ranks_all else round(float(np.mean(ranks_all)), 1),
        "v2_mean_best_rank": None if not ranks_v2 else round(float(np.mean(ranks_v2)), 1),
        "median_best_rank": None if not ranks_all else round(float(np.median(ranks_all)), 1),
        "v2_median_best_rank": None if not ranks_v2 else round(float(np.median(ranks_v2)), 1),
        "per_query": per_query,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["base", "large", "both"], default="both")
    args = parser.parse_args()
    sizes = ["base", "large"] if args.model == "both" else [args.model]

    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    v2 = json.loads(V2.read_text(encoding="utf-8"))

    results = {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "mode": "semantic",
        "golden": str(GOLDEN.relative_to(BASE_DIR)),
        "compare_to": str(V2.relative_to(BASE_DIR)),
        "note": "Verses encoded with passage:; queries with query:; raw Cyrillic.",
        "by_model": {},
        "aggregate_vs_v2": {},
    }

    for size in sizes:
        block = measure_model(size, golden)
        results["by_model"][MODEL_IDS[size]] = block
        results["aggregate_vs_v2"][MODEL_IDS[size]] = aggregate(block, v2)

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT}")

    print("\n=== AGGREGATE vs v2 (Qwen3+Bible Instruct) ===")
    for mid, agg in results["aggregate_vs_v2"].items():
        print(f"\n{mid}")
        print(
            f"  concepts best≤20: {agg['concepts_with_best_rank_le_20']}/"
            f"{agg['n_concepts']}  (v2: {agg['v2_concepts_with_best_rank_le_20']})"
        )
        print(
            f"  weird top1 (not in expected): {agg['weird_top1_count']}  "
            f"(v2: {agg['v2_weird_top1_count']})"
        )
        print(
            f"  mean best-rank: {agg['mean_best_rank']}  (v2: {agg['v2_mean_best_rank']})"
        )
        print(
            f"  median best-rank: {agg['median_best_rank']}  "
            f"(v2: {agg['v2_median_best_rank']})"
        )


if __name__ == "__main__":
    main()
