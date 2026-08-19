"""
Narrow map-in-semantic check: the five v1 queries that hit a group, plus опростити.

Compares full-NZ cosine rank (goli encode_query) vs max-pool expansion.
Writes data/concept/semantic_map_narrow_delta.json

Run from backend/:
  python scripts/measure_semantic_map_narrow.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from config import SEARCH_SEMANTIC_POOL  # noqa: E402
from services.search.concept_map import (  # noqa: E402
    bridge_refs,
    expansion_queries,
    resolve_group,
)
from services.search.semantic import encode_query, load_embed_nt_index  # noqa: E402

GOLDEN_V1 = BASE_DIR / "data" / "concept" / "semantic_golden_v1.json"
BASELINE_V5 = BASE_DIR / "data" / "concept" / "semantic_baseline_v5_embedic_large_prod.json"
BASELINE_STRESS = BASE_DIR / "data" / "concept" / "semantic_baseline_v2_stress_embedic_large.json"
OUT = BASE_DIR / "data" / "concept" / "semantic_map_narrow_delta.json"

V1_QUERIES = ("опроштај", "покајање", "вера", "молитва", "васкрсење")
PAGE = 20


def verse_index(verses) -> dict[tuple[str, int, int], int]:
    out: dict[tuple[str, int, int], int] = {}
    books = verses["book"].astype(str).str.strip()
    for i in range(len(verses)):
        key = (str(books.iloc[i]), int(verses["chapter"].iloc[i]), int(verses["verse"].iloc[i]))
        out.setdefault(key, i)
    return out


def ranks_from(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores)
    rank_of = np.empty(len(scores), dtype=np.int32)
    rank_of[order] = np.arange(1, len(scores) + 1)
    return rank_of


def stored_v5_rank(v5: dict, query: str, corpus: str, book: str, ch: int, vs: int) -> int | None:
    for concept in v5["concepts"]:
        if concept["query"] != query:
            continue
        for exp in concept["by_corpus"][corpus]["expected"]:
            if exp["book"] == book and exp["chapter"] == ch and exp["verse"] == vs:
                return exp["rank"]
    return None


def stored_stress_oprosti(stress: dict, corpus: str, book: str, ch: int, vs: int) -> int | None:
    for cat in stress["categories"]:
        for concept in cat["concepts"]:
            if concept["query"] != "опростити":
                continue
            for exp in concept["by_corpus"][corpus]["expected"]:
                if exp["book"] == book and exp["chapter"] == ch and exp["verse"] == vs:
                    return exp["rank"]
    return None


def main() -> None:
    golden = json.loads(GOLDEN_V1.read_text(encoding="utf-8"))
    v5 = json.loads(BASELINE_V5.read_text(encoding="utf-8"))
    stress = json.loads(BASELINE_STRESS.read_text(encoding="utf-8"))

    expected_by_query: dict[str, list[dict]] = {}
    for concept in golden["concepts"]:
        if concept["query"] in V1_QUERIES:
            expected_by_query[concept["query"]] = concept["expected"]
    for cat in json.loads(
        (BASE_DIR / "data" / "concept" / "semantic_golden_v2_stress.json").read_text(
            encoding="utf-8"
        )
    )["categories"]:
        for concept in cat["concepts"]:
            if concept["query"] == "опростити":
                expected_by_query["опростити"] = concept["expected"]

    indexes = {c: load_embed_nt_index(c) for c in ("dk", "spc")}
    embs = {c: np.asarray(indexes[c]["embeddings"], dtype=np.float32) for c in indexes}
    lookups = {c: verse_index(indexes[c]["verses"]) for c in indexes}

    results: dict = {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "note": "Narrow brake: v1 queries that hit the map + stress опростити. "
        "old = goli encode_query (full NZ rank); new = max-pool over expansion. "
        "Pins do not change full-NZ cosine rank; recorded separately.",
        "pool": SEARCH_SEMANTIC_POOL,
        "page": PAGE,
        "queries": [],
    }

    print(
        f"{'query':<12} {'corp':<4} {'ref':<22} {'old':>6} {'new':>6} {'Δ':>6} "
        f"{'page1':<5} {'moved_by':<18}"
    )

    for query in (*V1_QUERIES, "опростити"):
        group = resolve_group(query)
        q_texts = expansion_queries(query)
        q_entry = {
            "query": query,
            "group_id": None if group is None else group["id"],
            "expansion": q_texts,
            "by_corpus": {},
        }
        for corpus in ("dk", "spc"):
            verses = indexes[corpus]["verses"]
            emb = embs[corpus]
            lookup = lookups[corpus]
            member_scores = {q: emb @ encode_query(q) for q in q_texts}
            old_scores = member_scores[q_texts[0]]
            stacked = np.stack([member_scores[q] for q in q_texts], axis=0)
            new_scores = np.max(stacked, axis=0)
            winner_idx = np.argmax(stacked, axis=0)
            old_ranks = ranks_from(old_scores)
            new_ranks = ranks_from(new_scores)
            bridges = set(bridge_refs(group, corpus)) if group else set()

            rows = []
            for exp in expected_by_query[query]:
                if exp.get("corpus", "both") not in ("both", corpus):
                    continue
                key = (exp["book"], int(exp["chapter"]), int(exp["verse"]))
                i = lookup.get(key)
                if i is None:
                    rows.append({"book": key[0], "chapter": key[1], "verse": key[2], "missing": True})
                    print(f"{query:<12} {corpus:<4} {key[0]} {key[1]}:{key[2]} MISSING")
                    continue
                old_r = int(old_ranks[i])
                new_r = int(new_ranks[i])
                stored = (
                    stored_stress_oprosti(stress, corpus, *key)
                    if query == "опростити"
                    else stored_v5_rank(v5, query, corpus, *key)
                )
                win_q = q_texts[int(winner_idx[i])]
                orig_s = float(old_scores[i])
                win_s = float(new_scores[i])
                if win_q == q_texts[0] or abs(win_s - orig_s) < 1e-6:
                    moved_by = "original"
                else:
                    moved_by = win_q
                pinned = key in bridges and new_r > SEARCH_SEMANTIC_POOL
                row = {
                    "book": key[0],
                    "chapter": key[1],
                    "verse": key[2],
                    "layer": exp.get("layer"),
                    "old_rank": old_r,
                    "new_rank": new_r,
                    "delta": old_r - new_r,
                    "stored_old_rank": stored,
                    "old_score": round(orig_s, 4),
                    "new_score": round(win_s, 4),
                    "moved_by": moved_by,
                    "in_pool": new_r <= SEARCH_SEMANTIC_POOL,
                    "on_page1": new_r <= PAGE,
                    "bridge_pin_would_apply": pinned,
                }
                rows.append(row)
                flag = ""
                if new_r > old_r:
                    flag = " REGRESS"
                elif new_r < old_r:
                    flag = " UP"
                print(
                    f"{query:<12} {corpus:<4} {key[0]} {key[1]}:{key[2]:<6} "
                    f"{old_r:>6} {new_r:>6} {old_r - new_r:>+6} "
                    f"{str(new_r <= PAGE):<5} {moved_by:<18}{flag}"
                )
            q_entry["by_corpus"][corpus] = {"expected": rows}
        results["queries"].append(q_entry)

    regressions = []
    lifts = []
    for q in results["queries"]:
        for corpus, block in q["by_corpus"].items():
            for row in block["expected"]:
                if row.get("missing"):
                    continue
                rec = {
                    "query": q["query"],
                    "corpus": corpus,
                    "ref": f"{row['book']} {row['chapter']}:{row['verse']}",
                    "old": row["old_rank"],
                    "new": row["new_rank"],
                    "delta": row["delta"],
                    "moved_by": row["moved_by"],
                    "on_page1": row["on_page1"],
                }
                if row["delta"] < 0:
                    regressions.append(rec)
                elif row["delta"] > 0:
                    lifts.append(rec)
    results["summary"] = {
        "n_expected": sum(
            len(b["expected"]) for q in results["queries"] for b in q["by_corpus"].values()
        ),
        "n_lift": len(lifts),
        "n_regress": len(regressions),
        "n_same": sum(
            1
            for q in results["queries"]
            for b in q["by_corpus"].values()
            for r in b["expected"]
            if not r.get("missing") and r["delta"] == 0
        ),
        "regressions": sorted(regressions, key=lambda x: x["delta"]),
        "big_lifts": sorted(lifts, key=lambda x: -x["delta"])[:12],
    }
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT}")
    print(
        f"lift={results['summary']['n_lift']} same={results['summary']['n_same']} "
        f"regress={results['summary']['n_regress']}"
    )


if __name__ == "__main__":
    main()
