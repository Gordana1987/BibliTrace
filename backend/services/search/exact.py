"""
Exact concept search: surface-form match only.

Requirements (locked):
- Case-insensitive
- Word-boundary safe (token match; \"реч\" must not match inside \"изрека\")
- Multi-token terms (2–3 words) match as a *consecutive* token sequence in the verse
- Optional wildcards per token: word* (prefix), *word (suffix), *word* (contains)
- Results in biblical NT order; no relevance score
- Pagination via offset/limit (UI: 20 + load more)
"""

from __future__ import annotations

from models.schemas import CorpusSearchResult
from services.search.common import (
    consecutive_pattern_match,
    filter_nt_rows,
    load_verses_df,
    normalize_surface,
    page_hits,
    parse_exact_patterns,
    sort_hits_biblical,
    tokenize_surface,
)


def search_exact(
    term: str,
    corpus: str,
    *,
    offset: int = 0,
    limit: int = 20,
) -> CorpusSearchResult:
    patterns = parse_exact_patterns(term)
    df = filter_nt_rows(load_verses_df(corpus))

    hits: list[dict] = []
    for row in df.itertuples(index=False):
        verse_text = normalize_surface(str(row.text))
        if not verse_text:
            continue
        tokens = tokenize_surface(verse_text)
        if not consecutive_pattern_match(tokens, patterns):
            continue
        hits.append(
            {
                "book": str(row.book).strip(),
                "chapter": int(row.chapter),
                "verse": int(row.verse),
                "text": verse_text,
                "score": None,
            }
        )

    hits = sort_hits_biblical(hits)
    return page_hits(hits, corpus=corpus, offset=offset, limit=limit, ranking="biblical_order")
