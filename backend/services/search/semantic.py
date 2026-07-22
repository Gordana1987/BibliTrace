"""
Semantic concept search: meaning / motif (dense retrieval).

v1 plan: pure Qwen dense over NZ (no CE, no LaBSE) — not wired yet.
"""

from __future__ import annotations

from models.schemas import CorpusSearchResult
from services.search.common import page_hits, parse_term_tokens


def search_semantic(
    term: str,
    corpus: str,
    *,
    offset: int = 0,
    limit: int = 20,
) -> CorpusSearchResult:
    parse_term_tokens(term)
    return page_hits(
        [],
        corpus=corpus,
        offset=offset,
        limit=limit,
        ranking="score",
    )
