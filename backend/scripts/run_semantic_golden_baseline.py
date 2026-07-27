"""
Semantic golden-set measurement (ranks for expected verses).

Run from backend/:
  python scripts/run_semantic_golden_baseline.py
  python scripts/run_semantic_golden_baseline.py --out data/concept/semantic_baseline_v2_prefix.json \\
      --encode-label instruct_bible

Default writes semantic_baseline_v1.json (historical name). Prefer --out for new runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from config import (  # noqa: E402
    SEARCH_EMBED_MODEL_ID,
    SEARCH_EMBED_QUERY_PREFIX,
)
from services.search.semantic import encode_query, load_embed_nt_index  # noqa: E402

GOLDEN = BASE_DIR / "data" / "concept" / "semantic_golden_v1.json"


def rank_verse(verses, scores: np.ndarray, book: str, chapter: int, verse: int) -> tuple[int | None, float | None]:
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


def top1(verses, scores: np.ndarray) -> dict:
    i = int(np.argmax(scores))
    row = verses.iloc[i]
    return {
        "book": str(row["book"]).strip(),
        "chapter": int(row["chapter"]),
        "verse": int(row["verse"]),
        "score": round(float(scores[i]), 4),
        "text": str(row["text"])[:120],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure semantic ranks on golden concepts.")
    parser.add_argument(
        "--out",
        type=Path,
        default=BASE_DIR / "data" / "concept" / "semantic_baseline_v1.json",
        help="Output JSON path (default: semantic_baseline_v1.json).",
    )
    parser.add_argument(
        "--encode-label",
        default="query",
        help="Label stored in results.encode (e.g. query, instruct_bible).",
    )
    args = parser.parse_args()
    out_path = args.out if args.out.is_absolute() else BASE_DIR / args.out

    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    indexes = {c: load_embed_nt_index(c) for c in ("dk", "spc")}
    emb = {
        c: np.asarray(indexes[c]["embeddings"], dtype=np.float32) for c in indexes
    }

    results = {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "mode": "semantic",
        "model": SEARCH_EMBED_MODEL_ID,
        "encode": args.encode_label,
        "query_prefix": SEARCH_EMBED_QUERY_PREFIX,
        "golden": str(GOLDEN.relative_to(BASE_DIR)),
        "concepts": [],
    }

    for concept in golden["concepts"]:
        query = concept["query"]
        q = encode_query(query)
        entry = {"query": query, "by_corpus": {}}
        for corpus in ("dk", "spc"):
            scores = emb[corpus] @ q
            verses = indexes[corpus]["verses"]
            corp_out = {"top1": top1(verses, scores), "expected": []}
            for exp in concept["expected"]:
                corp_filter = exp.get("corpus", "both")
                if corp_filter not in ("both", corpus):
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
        results["concepts"].append(entry)
        print(f"✓ {query}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
