"""
Measure dense + lemma RRF hybrid ranks on the golden set.

Uses live encode_query (Embedić query: prefix) + lemma hits fused with RRF (k=60).
Does not change production search — measurement only.

Run from backend/:
  python scripts/run_semantic_hybrid_rrf_baseline.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from config import SEARCH_EMBED_MODEL_ID, SEARCH_EMBED_QUERY_PREFIX  # noqa: E402
from services.search.lemma import search_lemma  # noqa: E402
from services.search.semantic import encode_query, load_embed_nt_index  # noqa: E402

GOLDEN = BASE_DIR / "data" / "concept" / "semantic_golden_v1.json"
V2 = BASE_DIR / "data" / "concept" / "semantic_baseline_v2_prefix.json"
OUT = BASE_DIR / "data" / "concept" / "semantic_baseline_v3_hybrid_rrf.json"

RRF_K = 60
MORPH_GROUP = ("опроштај", "грех", "молитва", "сведочанство", "стрпљење")


def dense_ranks(scores: np.ndarray) -> dict[int, int]:
    order = np.argsort(-scores)
    return {int(i): r + 1 for r, i in enumerate(order)}


def verse_key(book: str, chapter: int, verse: int) -> tuple[str, int, int]:
    return (str(book).strip(), int(chapter), int(verse))


def lemma_rank_map(corpus: str, term: str, verses_df) -> dict[tuple[str, int, int], int]:
    """Biblical-order ranks among lemma hits (1..N). Empty if no hits."""
    res = search_lemma(term, corpus, limit=10000)
    out: dict[tuple[str, int, int], int] = {}
    for rank, h in enumerate(res.hits, 1):
        out[verse_key(h.book, h.chapter, h.verse)] = rank
    return out


def index_key_to_pos(verses_df) -> dict[tuple[str, int, int], int]:
    m: dict[tuple[str, int, int], int] = {}
    for i in range(len(verses_df)):
        row = verses_df.iloc[i]
        m[verse_key(row["book"], row["chapter"], row["verse"])] = i
    return m


def rrf_scores(
    dense_rank: dict[int, int],
    lemma_by_key: dict[tuple[str, int, int], int],
    key_to_pos: dict[tuple[str, int, int], int],
    n_verses: int,
) -> np.ndarray:
    """RRF over all verses; lemma contributes only for keys it retrieved."""
    scores = np.zeros(n_verses, dtype=np.float64)
    for i, r in dense_rank.items():
        scores[i] += 1.0 / (RRF_K + r)
    for key, r in lemma_by_key.items():
        pos = key_to_pos.get(key)
        if pos is not None:
            scores[pos] += 1.0 / (RRF_K + r)
    return scores


def top1_from_scores(verses, scores: np.ndarray) -> dict:
    i = int(np.argmax(scores))
    row = verses.iloc[i]
    return {
        "book": str(row["book"]).strip(),
        "chapter": int(row["chapter"]),
        "verse": int(row["verse"]),
        "score": round(float(scores[i]), 6),
        "text": str(row["text"])[:120],
    }


def main() -> None:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    v2 = json.loads(V2.read_text(encoding="utf-8"))
    v2m = {c["query"]: c for c in v2["concepts"]}

    indexes = {c: load_embed_nt_index(c) for c in ("dk", "spc")}
    emb = {c: np.asarray(indexes[c]["embeddings"], dtype=np.float32) for c in indexes}
    key_maps = {c: index_key_to_pos(indexes[c]["verses"]) for c in indexes}

    results = {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "mode": "semantic_hybrid_rrf",
        "model": SEARCH_EMBED_MODEL_ID,
        "encode": "query_prefix",
        "query_prefix": SEARCH_EMBED_QUERY_PREFIX,
        "rrf_k": RRF_K,
        "lemma_channel": "search_lemma (same lemma only; no noun↔verb map)",
        "golden": str(GOLDEN.relative_to(BASE_DIR)),
        "compare_to": str(V2.relative_to(BASE_DIR)),
        "morph_group": list(MORPH_GROUP),
        "concepts": [],
        "morph_group_summary": [],
    }

    print("Encoding queries + RRF…")
    for concept in golden["concepts"]:
        query = concept["query"]
        q = encode_query(query)
        entry = {"query": query, "by_corpus": {}}
        for corpus in ("dk", "spc"):
            verses = indexes[corpus]["verses"]
            dense = emb[corpus] @ q
            d_rank = dense_ranks(dense)
            lem_map = lemma_rank_map(corpus, query, verses)
            fused = rrf_scores(d_rank, lem_map, key_maps[corpus], len(verses))
            fused_order = np.argsort(-fused)
            fused_rank_of = {int(i): r + 1 for r, i in enumerate(fused_order)}

            corp_out = {
                "top1": top1_from_scores(verses, fused),
                "lemma_hit_count": len(lem_map),
                "expected": [],
            }
            for exp in concept["expected"]:
                corp_filter = exp.get("corpus", "both")
                if corp_filter not in ("both", corpus):
                    continue
                key = verse_key(exp["book"], exp["chapter"], exp["verse"])
                pos = key_maps[corpus].get(key)
                if pos is None:
                    rank = score = None
                    in_lemma = False
                else:
                    rank = fused_rank_of[pos]
                    score = float(fused[pos])
                    in_lemma = key in lem_map
                # v2 dense rank for comparison
                v2_exp = None
                for e in v2m.get(query, {}).get("by_corpus", {}).get(corpus, {}).get("expected", []):
                    if (
                        e["book"] == exp["book"]
                        and e["chapter"] == exp["chapter"]
                        and e["verse"] == exp["verse"]
                    ):
                        v2_exp = e
                        break
                corp_out["expected"].append(
                    {
                        "book": exp["book"],
                        "chapter": exp["chapter"],
                        "verse": exp["verse"],
                        "layer": exp["layer"],
                        "rank": rank,
                        "score": None if score is None else round(score, 6),
                        "in_lemma_channel": in_lemma,
                        "v2_dense_rank": None if v2_exp is None else v2_exp["rank"],
                        "delta_vs_v2": None
                        if (rank is None or v2_exp is None or v2_exp["rank"] is None)
                        else (v2_exp["rank"] - rank),
                    }
                )
            entry["by_corpus"][corpus] = corp_out
        results["concepts"].append(entry)
        print(f"✓ {query}")

    # Morph-group summary table
    for query in MORPH_GROUP:
        entry = next(c for c in results["concepts"] if c["query"] == query)
        for corpus, data in entry["by_corpus"].items():
            for e in data["expected"]:
                results["morph_group_summary"].append(
                    {
                        "query": query,
                        "corpus": corpus,
                        "ref": f"{e['book']} {e['chapter']}:{e['verse']}",
                        "layer": e["layer"],
                        "in_lemma": e["in_lemma_channel"],
                        "v2_rank": e["v2_dense_rank"],
                        "rrf_rank": e["rank"],
                        "delta": e["delta_vs_v2"],
                        "lemma_hits_total": data["lemma_hit_count"],
                    }
                )

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT}")

    print("\n=== MORPH GROUP: v2 dense → RRF hybrid ===")
    print(f"{'query':14} {'corp':3} {'ref':28} {'lem?':4} {'v2':>6} {'rrf':>6} {'Δ':>6}")
    for row in results["morph_group_summary"]:
        d = row["delta"]
        ds = f"{d:+d}" if d is not None else "—"
        print(
            f"{row['query']:14} {row['corpus']:3} {row['ref']:28} "
            f"{'Y' if row['in_lemma'] else 'N':4} "
            f"{row['v2_rank'] or '—':>6} {row['rrf_rank'] or '—':>6} {ds:>6}"
        )


if __name__ == "__main__":
    main()
