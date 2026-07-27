"""Concept search dispatcher: exact | lemma | semantic (per corpus, no merge)."""

from __future__ import annotations

from models.schemas import SearchRequest, SearchResponse
from services.search.common import (
    active_corpora_from_request,
    default_limit,
    page_hits,
    resolve_books_filter,
)
from services.search.exact import search_exact
from services.search.lemma import search_lemma
from services.search.semantic import search_semantic

_MODE_FNS = {
    "exact": search_exact,
    "lemma": search_lemma,
    "semantic": search_semantic,
}


def search(request: SearchRequest) -> SearchResponse:
    term = request.term.strip()
    if not term:
        return SearchResponse(term="", mode=request.mode, message="Нема појма за претрагу.")

    try:
        books = resolve_books_filter(request.books)
    except ValueError as exc:
        return SearchResponse(term=term, mode=request.mode, message=str(exc))

    corpora = active_corpora_from_request(list(request.corpora))
    limit = default_limit(request.limit)
    offset = request.offset
    fn = _MODE_FNS[request.mode]
    ranking = "score" if request.mode == "semantic" else "biblical_order"

    results: dict = {}
    errors: list[str] = []
    for corpus in corpora:
        try:
            results[corpus] = fn(
                term, corpus, offset=offset, limit=limit, books=books
            )
        except FileNotFoundError as exc:
            results[corpus] = page_hits(
                [],
                corpus=corpus,
                offset=offset,
                limit=limit,
                ranking=ranking,
            )
            errors.append(str(exc))
        except ValueError as exc:
            return SearchResponse(term=term, mode=request.mode, message=str(exc))

    if request.mode == "lemma":
        total_any = sum(r.total for r in results.values())
        msg = (
            f"Лема претрага — {total_any} појава укупно."
            if total_any
            else "Нема погодака за овај појам (лема)."
        )
    elif request.mode == "semantic":
        total_any = sum(r.total for r in results.values())
        if errors and not total_any:
            msg = "Семантичка претрага није доступна."
        elif total_any:
            msg = f"Семантичка претрага — {total_any} резултата укупно (по сродности)."
        else:
            msg = "Нема семантичких погодака за овај појам."
    else:
        total_any = sum(r.total for r in results.values())
        msg = (
            f"Егзактна претрага — {total_any} појава укупно."
            if total_any
            else "Нема погодака за овај појам."
        )
    if errors:
        msg = f"{msg} ({'; '.join(errors)})"

    return SearchResponse(
        term=term,
        mode=request.mode,
        results_by_corpus=results,
        message=msg,
    )
