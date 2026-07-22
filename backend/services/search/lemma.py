"""
Lemma concept search: consecutive lemma-token sequence over the lemma index.

Requirements (locked):
- Query: Cyrillic→Latin → CLASSLA → consecutive lemma match
- Corpus: prebuilt lemma_index.joblib from bible_lemmatized.csv (Latin lemmas)
- Multi-token terms match as a *consecutive* lemma sequence (not bag-of-lemmas)
- Results in biblical NT order; no relevance score
- Pagination via offset/limit
"""

from __future__ import annotations

from models.schemas import CorpusSearchResult
from services.search.common import (
    consecutive_token_match,
    load_lemma_index,
    normalize_surface,
    page_hits,
    parse_term_lemmas,
    sort_hits_biblical,
)


def search_lemma(
    term: str,
    corpus: str,
    *,
    offset: int = 0,
    limit: int = 20,
) -> CorpusSearchResult:
    needle = parse_term_lemmas(term)
    idx = load_lemma_index(corpus)
    verses: list[dict] = idx["verses"]
    lemma_tokens: list[list[str]] = idx["lemma_tokens"]
    inverted: dict[str, list[int]] = idx.get("inverted") or {}

    # Candidate verses: intersection of inverted lists for each needle lemma.
    candidate_indices: list[int] | None = None
    for lem in needle:
        posting = inverted.get(lem, [])
        if candidate_indices is None:
            candidate_indices = list(posting)
        else:
            posting_set = set(posting)
            candidate_indices = [i for i in candidate_indices if i in posting_set]
        if not candidate_indices:
            break

    if not candidate_indices:
        return page_hits(
            [],
            corpus=corpus,
            offset=offset,
            limit=limit,
            ranking="biblical_order",
        )

    hits: list[dict] = []
    for i in candidate_indices:
        tokens = lemma_tokens[i]
        if not consecutive_token_match(tokens, needle):
            continue
        v = verses[i]
        hits.append(
            {
                "book": v["book"],
                "chapter": v["chapter"],
                "verse": v["verse"],
                "text": normalize_surface(str(v.get("text", ""))),
                "score": None,
            }
        )

    hits = sort_hits_biblical(hits)
    return page_hits(hits, corpus=corpus, offset=offset, limit=limit, ranking="biblical_order")
