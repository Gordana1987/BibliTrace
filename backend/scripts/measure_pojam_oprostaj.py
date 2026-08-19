"""
Measure pojam channel vs semantic on опроштај / опростити golden loci.

Run from backend/:
  python scripts/measure_pojam_oprostaj.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from services.search.concept_map import resolve_group  # noqa: E402
from services.search.semantic import encode_query, load_embed_nt_index, search_semantic  # noqa: E402

OUT = BASE_DIR / "data" / "concept" / "pojam_measure_oprostaj.json"

# Stress-golden infinitive hole + v1 noun control.
OPROSTITI_EXPECTED = [
    ("Матеј", 6, 14),
    ("Лука", 6, 37),
    ("Ефешанима", 4, 32),
]
OPROSTAJ_EXPECTED = [
    ("Матеј", 6, 14),
    ("Матеј", 6, 12),
    ("Лука", 6, 37),
    ("Лука", 15, 20),  # prodigal compassion — no forgive lemma; semantic-only
    ("Лука", 23, 34),
    ("Ефешанима", 4, 32),
]
SEMANTIC_BASELINE = {
    "опростити": {"dk": 621, "spc": 452, "source": "semantic_baseline_v2_stress_embedic_large"},
    "опроштај": {
        "dk": {"Матеј 6:14": 20, "Лука 6:37": 14},
        "source": "semantic_baseline_v5_embedic_large_prod",
    },
}


def pojam_presence(term: str, expected: list[tuple]) -> dict:
    group = resolve_group(term)
    out = {"group": None if group is None else group["id"], "by_corpus": {}}
    for corpus in ("dk", "spc"):
        block = search_semantic(term, corpus, offset=0, limit=10_000)
        ordered = [(h.book, h.chapter, h.verse) for h in block.hits]
        pos = {ref: i + 1 for i, ref in enumerate(ordered)}
        rows = []
        for book, ch, vs in expected:
            ref = (book, ch, vs)
            rows.append(
                {
                    "book": book,
                    "chapter": ch,
                    "verse": vs,
                    "present": ref in pos,
                    "biblical_position": pos.get(ref),
                }
            )
        out["by_corpus"][corpus] = {
            "total": block.total,
            "expected": rows,
            "n_present": sum(1 for r in rows if r["present"]),
        }
    return out


def semantic_ranks(term: str, expected: list[tuple]) -> dict:
    out = {}
    q = encode_query(term)
    for corpus in ("dk", "spc"):
        idx = load_embed_nt_index(corpus)
        embs = np.asarray(idx["embeddings"], dtype=np.float32)
        df = idx["verses"]
        scores = embs @ q
        order = np.argsort(-scores)
        rank_of = {int(i): r + 1 for r, i in enumerate(order)}
        rows = []
        best = None
        for book, ch, vs in expected:
            mask = (
                (df["book"].astype(str).str.strip() == book)
                & (df["chapter"].astype(int) == ch)
                & (df["verse"].astype(int) == vs)
            )
            hit = np.flatnonzero(mask.to_numpy())
            if len(hit) == 0:
                rows.append({"book": book, "chapter": ch, "verse": vs, "rank": None})
                continue
            rnk = rank_of[int(hit[0])]
            rows.append({"book": book, "chapter": ch, "verse": vs, "rank": rnk})
            best = rnk if best is None else min(best, rnk)
        out[corpus] = {"best_rank": best, "expected": rows}
    return out


def main() -> None:
    print("Resolving groups…")
    print("  опростити →", None if resolve_group("опростити") is None else resolve_group("опростити")["id"])
    print("  опроштај →", None if resolve_group("опроштај") is None else resolve_group("опроштај")["id"])

    print("Pojam channel…")
    pojam_inf = pojam_presence("опростити", OPROSTITI_EXPECTED)
    pojam_noun = pojam_presence("опроштај", OPROSTAJ_EXPECTED)

    print("Semantic control (embedic-large)…")
    sem_inf = semantic_ranks("опростити", OPROSTITI_EXPECTED)
    sem_noun = semantic_ranks("опроштај", OPROSTAJ_EXPECTED)

    payload = {
        "measured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "channel": "pojam = lemma union + bridges; semantic unchanged",
        "semantic_prior": SEMANTIC_BASELINE,
        "опростити": {"pojam": pojam_inf, "semantic": sem_inf},
        "опроштај": {"pojam": pojam_noun, "semantic": sem_noun},
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
