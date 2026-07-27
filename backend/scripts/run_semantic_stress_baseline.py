"""
Stress golden (v2) measurement through production Embedić-large path.

Run from backend/:
  python scripts/run_semantic_stress_baseline.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from config import (  # noqa: E402
    SEARCH_EMBED_DOC_PREFIX,
    SEARCH_EMBED_INDEX_NAME,
    SEARCH_EMBED_MODEL_ID,
    SEARCH_EMBED_QUERY_PREFIX,
)
from services.search.semantic import encode_query, load_embed_nt_index  # noqa: E402

GOLDEN = BASE_DIR / "data" / "concept" / "semantic_golden_v2_stress.json"
OUT = BASE_DIR / "data" / "concept" / "semantic_baseline_v2_stress_embedic_large.json"
SCORE_DIST_CATEGORIES = {"D", "F"}


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
        "text": str(row["text"])[:160],
    }


def score_distribution(scores: np.ndarray) -> dict:
    order = np.argsort(-scores)
    top = scores[order[:10]]
    return {
        "max_score": round(float(top[0]), 4),
        "top10_mean": round(float(np.mean(top)), 4),
        "gap_top1_vs_top10": round(float(top[0] - top[9]), 4),
        "gap_top1_vs_top2": round(float(top[0] - top[1]), 4),
    }


def best_rank(expected_rows: list[dict]) -> int | None:
    ranks = [e["rank"] for e in expected_rows if e.get("rank") is not None]
    return min(ranks) if ranks else None


def main() -> None:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    indexes = {c: load_embed_nt_index(c) for c in ("dk", "spc")}
    emb = {c: np.asarray(indexes[c]["embeddings"], dtype=np.float32) for c in indexes}

    results = {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "mode": "semantic_stress",
        "model": SEARCH_EMBED_MODEL_ID,
        "index": SEARCH_EMBED_INDEX_NAME,
        "pipeline": "services.search.semantic.encode_query + load_embed_nt_index",
        "query_prefix": SEARCH_EMBED_QUERY_PREFIX,
        "doc_prefix": SEARCH_EMBED_DOC_PREFIX,
        "golden": str(GOLDEN.relative_to(BASE_DIR)),
        "categories": [],
        "summary": [],
    }

    print(f"Measuring {SEARCH_EMBED_MODEL_ID} on stress golden…")
    for cat in golden["categories"]:
        cat_out = {
            "id": cat["id"],
            "name": cat["name"],
            "priority": cat.get("priority"),
            "concepts": [],
        }
        for concept in cat["concepts"]:
            query = concept["query"]
            q = encode_query(query)
            entry = {
                "query": query,
                "hypothesis": concept.get("hypothesis"),
                "pair_with_v1": concept.get("pair_with_v1"),
                "by_corpus": {},
            }
            for corpus in ("dk", "spc"):
                scores = emb[corpus] @ q
                verses = indexes[corpus]["verses"]
                t1 = top1(verses, scores)
                expected_out = []
                for exp in concept.get("expected", []):
                    if exp.get("corpus", "both") not in ("both", corpus):
                        continue
                    rank, score = rank_verse(
                        verses, scores, exp["book"], exp["chapter"], exp["verse"]
                    )
                    expected_out.append(
                        {
                            **{
                                k: exp[k]
                                for k in ("book", "chapter", "verse", "layer", "corpus")
                                if k in exp
                            },
                            "rank": rank,
                            "score": None if score is None else round(score, 4),
                            "note": exp.get("note"),
                        }
                    )
                expected_keys = {
                    (e["book"], e["chapter"], e["verse"]) for e in expected_out
                }
                weird = (
                    (t1["book"], t1["chapter"], t1["verse"]) not in expected_keys
                    if expected_keys
                    else None
                )
                corp = {
                    "top1": t1,
                    "best_rank": best_rank(expected_out),
                    "weird_top1": weird,
                    "expected": expected_out,
                }
                if cat["id"] in SCORE_DIST_CATEGORIES:
                    corp["score_distribution"] = score_distribution(scores)
                entry["by_corpus"][corpus] = corp
            cat_out["concepts"].append(entry)
            br = {
                c: entry["by_corpus"][c]["best_rank"] for c in ("dk", "spc")
            }
            print(f"  [{cat['id']}] {query}: best_rank dk={br['dk']} spc={br['spc']}")
        results["categories"].append(cat_out)

    # Compact summary table
    for cat in results["categories"]:
        for c in cat["concepts"]:
            row = {
                "category": cat["id"],
                "query": c["query"],
                "dk_best_rank": c["by_corpus"]["dk"]["best_rank"],
                "spc_best_rank": c["by_corpus"]["spc"]["best_rank"],
                "dk_top1_score": c["by_corpus"]["dk"]["top1"]["score"],
                "spc_top1_score": c["by_corpus"]["spc"]["top1"]["score"],
                "dk_weird_top1": c["by_corpus"]["dk"]["weird_top1"],
                "spc_weird_top1": c["by_corpus"]["spc"]["weird_top1"],
            }
            if "score_distribution" in c["by_corpus"]["dk"]:
                row["dk_gap_t1_t10"] = c["by_corpus"]["dk"]["score_distribution"][
                    "gap_top1_vs_top10"
                ]
                row["spc_gap_t1_t10"] = c["by_corpus"]["spc"]["score_distribution"][
                    "gap_top1_vs_top10"
                ]
            results["summary"].append(row)

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT}")

    print("\n=== SUMMARY ===")
    for row in results["summary"]:
        extra = ""
        if "dk_gap_t1_t10" in row:
            extra = f"  gap10 dk/spc={row['dk_gap_t1_t10']}/{row['spc_gap_t1_t10']}"
        print(
            f"  {row['category']} {row['query']}: "
            f"rank {row['dk_best_rank']}/{row['spc_best_rank']}  "
            f"top1score {row['dk_top1_score']}/{row['spc_top1_score']}  "
            f"weird {row['dk_weird_top1']}/{row['spc_weird_top1']}"
            f"{extra}"
        )


if __name__ == "__main__":
    main()
