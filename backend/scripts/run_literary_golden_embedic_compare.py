"""
Literary golden_set.json through Embedić-large (NZ index) vs Qwen3 (full-bible).

Important scope note:
  - Live Embedić indexes are NT-only → OT expected anchors cannot hit.
  - Legacy Qwen qwen_embeddings.joblib is full Bible → fair peer for OT cases.

Run from backend/:
  python scripts/run_literary_golden_embedic_compare.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
REPO = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))

from config import DK_NT_BOOKS, SEARCH_EMBED_MODEL_ID, SEARCH_EMBED_QUERY_PREFIX  # noqa: E402
from services.search.semantic import encode_query, load_embed_nt_index  # noqa: E402

GOLDEN = REPO / "archive" / "literary-text-search" / "benchmark" / "golden_set.json"
OUT = (
    REPO
    / "archive"
    / "literary-text-search"
    / "benchmark"
    / "results"
    / "literary_golden_embedic_vs_qwen.json"
)
NT = frozenset(DK_NT_BOOKS)


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


def top_n(verses, scores, n=5):
    order = np.argsort(-scores)[:n]
    out = []
    for rank, i in enumerate(order, 1):
        row = verses.iloc[int(i)]
        out.append(
            {
                "rank": rank,
                "book": str(row["book"]).strip(),
                "chapter": int(row["chapter"]),
                "verse": int(row["verse"]),
                "score": round(float(scores[int(i)]), 4),
                "text": str(row["text"])[:120],
            }
        )
    return out


def evaluate_scores(case, verses, scores, *, index_scope: str):
    expected_out = []
    for exp in case["expected"]:
        book = exp["book"]
        in_index = book in NT if index_scope == "nt" else True
        if index_scope == "nt" and book not in NT:
            expected_out.append(
                {
                    "book": book,
                    "chapter": exp["chapter"],
                    "verse": exp["verse"],
                    "rank": None,
                    "score": None,
                    "in_index": False,
                }
            )
            continue
        rank, score = rank_verse(
            verses, scores, book, exp["chapter"], exp["verse"]
        )
        expected_out.append(
            {
                "book": book,
                "chapter": exp["chapter"],
                "verse": exp["verse"],
                "rank": rank,
                "score": None if score is None else round(score, 4),
                "in_index": in_index,
            }
        )
    ranks = [e["rank"] for e in expected_out if e["rank"] is not None]
    match_mode = case.get("match", "all")
    any_hit = bool(ranks)
    all_scorable = [e for e in expected_out if e["in_index"]]
    all_hit = bool(all_scorable) and all(e["rank"] is not None for e in all_scorable)
    # For reporting: treat "hit" as rank present (full argsort, not top-k truncate)
    if match_mode == "any":
        recall = any_hit
    else:
        # strict among anchors that exist in this index
        recall = all_hit if all_scorable else False
    return {
        "match_mode": match_mode,
        "best_rank": min(ranks) if ranks else None,
        "any_anchor_ranked": any_hit,
        "all_in_index_anchors_ranked": all_hit if all_scorable else None,
        "n_expected": len(expected_out),
        "n_in_index": sum(1 for e in expected_out if e["in_index"]),
        "expected": expected_out,
        "top5": top_n(verses, scores, 5),
    }


def encode_qwen(term: str) -> np.ndarray:
    from services.detection import encode_query_qwen

    return np.asarray(encode_query_qwen(term), dtype=np.float32).reshape(-1)


def load_qwen_full(corpus: str = "dk") -> dict:
    path = BASE_DIR / "data" / corpus / "qwen_embeddings.joblib"
    return joblib.load(path)


def scope_of_case(case: dict) -> str:
    books = {e["book"] for e in case["expected"]}
    if books <= NT:
        return "nt_only"
    if books.isdisjoint(NT):
        return "ot_only"
    return "mixed"


def agg(rows, key_best="best_rank"):
    bests = [r[key_best] for r in rows if r.get(key_best) is not None]
    return {
        "n": len(rows),
        "with_rank": len(bests),
        "best_le_10": sum(1 for b in bests if b <= 10),
        "best_le_20": sum(1 for b in bests if b <= 20),
        "best_le_50": sum(1 for b in bests if b <= 50),
        "best_le_200": sum(1 for b in bests if b <= 200),
        "mean_best": None if not bests else round(float(np.mean(bests)), 1),
        "median_best": None if not bests else round(float(np.median(bests)), 1),
    }


def main() -> None:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    cases = golden["cases"]

    print("Loading Embedić NT (dk)…")
    emb_idx = load_embed_nt_index("dk")
    emb_m = np.asarray(emb_idx["embeddings"], dtype=np.float32)
    emb_v = emb_idx["verses"]

    print("Loading Qwen full-bible (dk)…")
    qwen_idx = load_qwen_full("dk")
    qwen_m = np.asarray(qwen_idx["embeddings"], dtype=np.float32)
    qwen_v = qwen_idx["verses"]

    results = {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "golden": str(GOLDEN.relative_to(REPO)),
        "note": (
            "Literary golden (Andrić, Jefimija, …) through Embedić-large NZ index "
            "vs Qwen3-Embedding-0.6B full-bible. Embedić cannot retrieve OT anchors."
        ),
        "embedic": {
            "model": SEARCH_EMBED_MODEL_ID,
            "index": "embedic_large_nt_embeddings.joblib",
            "scope": "nt",
            "query_prefix": SEARCH_EMBED_QUERY_PREFIX,
            "n_verses": int(emb_m.shape[0]),
        },
        "qwen": {
            "model": "Qwen/Qwen3-Embedding-0.6B",
            "index": "qwen_embeddings.joblib",
            "scope": "full_bible",
            "n_verses": int(qwen_m.shape[0]),
        },
        "cases": [],
        "summary": {},
    }

    print(f"Measuring {len(cases)} cases…")
    for case in cases:
        q = case["query"]
        scope = scope_of_case(case)
        print(f"  [{scope}] {case['id']} ({len(q.split())} words)")

        e_scores = emb_m @ encode_query(q)
        q_scores = qwen_m @ encode_qwen(q)

        entry = {
            "id": case["id"],
            "layer": case.get("layer"),
            "scope": scope,
            "query_words": len(q.split()),
            "query_preview": q[:160],
            "embedic": evaluate_scores(case, emb_v, e_scores, index_scope="nt"),
            "qwen": evaluate_scores(case, qwen_v, q_scores, index_scope="full"),
        }
        results["cases"].append(entry)

    # Summaries
    def pick(model_key, scope_filter=None):
        rows = []
        for c in results["cases"]:
            if scope_filter and c["scope"] not in scope_filter:
                continue
            rows.append(
                {
                    "id": c["id"],
                    "best_rank": c[model_key]["best_rank"],
                    "scope": c["scope"],
                }
            )
        return rows

    results["summary"] = {
        "embedic_all": agg(pick("embedic")),
        "qwen_all": agg(pick("qwen")),
        "embedic_nt_and_mixed": agg(pick("embedic", {"nt_only", "mixed"})),
        "qwen_nt_and_mixed": agg(pick("qwen", {"nt_only", "mixed"})),
        "embedic_nt_only": agg(pick("embedic", {"nt_only"})),
        "qwen_nt_only": agg(pick("qwen", {"nt_only"})),
        "qwen_ot_only": agg(pick("qwen", {"ot_only"})),
        "embedic_ot_only_note": "OT anchors out of Embedić NZ index → best_rank null by design",
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT}")

    print("\n=== SUMMARY ===")
    for k, v in results["summary"].items():
        if isinstance(v, dict):
            print(
                f"  {k}: n={v['n']} with_rank={v['with_rank']} "
                f"≤10={v['best_le_10']} ≤20={v['best_le_20']} ≤50={v['best_le_50']} ≤200={v['best_le_200']} "
                f"mean={v['mean_best']} med={v['median_best']}"
            )
        else:
            print(f"  {k}: {v}")

    print("\n=== HIGHLIGHTS (Andrić / Jefimija / Desanka) ===")
    for c in results["cases"]:
        if any(x in c["id"] for x in ("andric", "jefimija", "desanka", "njegos", "rastko")):
            print(
                f"  {c['id']} [{c['scope']}]: "
                f"embedic best={c['embedic']['best_rank']}  "
                f"qwen best={c['qwen']['best_rank']}"
            )
            print(f"    emb top1: {c['embedic']['top5'][0]}")
            print(f"    qwen top1: {c['qwen']['top5'][0]}")


if __name__ == "__main__":
    main()
