"""LangChain tools wrapping live exact / lemma / semantic search."""

from __future__ import annotations

import json
from typing import Literal

from langchain_core.tools import tool

from config import AGENT_SEARCH_LIMIT
from services.search.common import resolve_books_filter
from services.search.exact import search_exact
from services.search.lemma import search_lemma
from services.search.semantic import search_semantic

CorpusId = Literal["dk", "spc"]


def _hit_dict(h) -> dict:
    if isinstance(h, dict):
        return {
            "book": h["book"],
            "chapter": h["chapter"],
            "verse": h["verse"],
            "text": h["text"],
            "score": h.get("score"),
        }
    return {
        "book": h.book,
        "chapter": h.chapter,
        "verse": h.verse,
        "text": h.text,
        "score": h.score,
    }


def _hits_payload(fn, term: str, corpus: CorpusId, limit: int, books: list[str] | None) -> dict:
    book_filter = resolve_books_filter(books)
    try:
        block = fn(term.strip(), corpus, offset=0, limit=limit, books=book_filter)
    except (ValueError, FileNotFoundError) as exc:
        return {"term": term, "corpus": corpus, "total": 0, "hits": [], "error": str(exc)}
    return {
        "term": term,
        "corpus": corpus,
        "total": block.total,
        "hits": [_hit_dict(h) for h in block.hits],
    }


def _as_json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


@tool
def semantic_search(
    term: str,
    corpus: CorpusId = "dk",
    limit: int = AGENT_SEARCH_LIMIT,
) -> str:
    """Семантичка претрага по значењу и мотиву (Embedić). term = ћирилица, кратак појам."""
    return _as_json(_hits_payload(search_semantic, term, corpus, limit, None))


@tool
def lemma_search(
    term: str,
    corpus: CorpusId = "dk",
    limit: int = AGENT_SEARCH_LIMIT,
) -> str:
    """Лема претрага — сви облици исте основе. term = ћирилица."""
    return _as_json(_hits_payload(search_lemma, term, corpus, limit, None))


@tool
def exact_search(
    term: str,
    corpus: CorpusId = "dk",
    limit: int = AGENT_SEARCH_LIMIT,
) -> str:
    """Егзактна претрага — тачан облик (* за wildcard). term = ћирилица."""
    return _as_json(_hits_payload(search_exact, term, corpus, limit, None))


@tool
def semantic_search_in_books(
    term: str,
    books: list[str],
    corpus: CorpusId = "dk",
    limit: int = AGENT_SEARCH_LIMIT,
) -> str:
    """Семантичка претрага ограничена на књиге (нpr. ['Марко', 'Матеј']). books = Pouke називи."""
    return _as_json(_hits_payload(search_semantic, term, corpus, limit, books))


SEARCH_TOOLS = [semantic_search, lemma_search, exact_search, semantic_search_in_books]
