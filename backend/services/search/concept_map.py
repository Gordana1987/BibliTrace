"""
Concept map: resolve a query to a Mark-pilot group for semantic query expansion.

Used by semantic search (not a separate UI mode).
"""

from __future__ import annotations

import json
from functools import lru_cache

from config import CONCEPT_MAP_PATH
from services.search.common import lemmatize_to_tokens, parse_term_tokens, tokenize_surface
from services.transliterate import cyrillic_to_latin, latin_to_cyrillic

_DEFAULT_BRIDGE_BOOK = "Марко"


@lru_cache(maxsize=1)
def load_concept_map() -> dict:
    if not CONCEPT_MAP_PATH.exists():
        return {"groups": []}
    return json.loads(CONCEPT_MAP_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _member_index() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for group in load_concept_map().get("groups", []):
        for raw in list(group.get("lemmas") or []) + list(group.get("query_aliases") or []):
            key = str(raw).casefold().strip()
            if key and key not in out:
                out[key] = group
    return out


def resolve_group(term: str) -> dict | None:
    """Single-token query → group if the surface or lemma is a member/alias."""
    parse_term_tokens(term, max_tokens=None)
    index = _member_index()
    surface = tokenize_surface(cyrillic_to_latin(term))
    if len(surface) == 1 and surface[0] in index:
        return index[surface[0]]
    lemmas = lemmatize_to_tokens(term)
    if len(lemmas) == 1 and lemmas[0] in index:
        return index[lemmas[0]]
    return None


def expansion_queries(term: str) -> list[str]:
    """Original Cyrillic query plus group members as Cyrillic (deduped, order preserved)."""
    seen: set[str] = set()
    out: list[str] = []
    original = " ".join(term.strip().split())
    if original:
        seen.add(original.casefold())
        out.append(original)
    group = resolve_group(term)
    if group is None:
        return out
    for raw in list(group.get("lemmas") or []) + list(group.get("query_aliases") or []):
        cyr = latin_to_cyrillic(str(raw).strip())
        if not cyr:
            continue
        key = cyr.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(cyr)
    return out


def bridge_refs(group: dict | None, corpus: str) -> list[tuple[str, int, int]]:
    if not group:
        return []
    spec = (group.get("bridges") or {}).get(corpus) or []
    default_book = str(load_concept_map().get("book") or _DEFAULT_BRIDGE_BOOK)
    refs: list[tuple[str, int, int]] = []
    for item in spec:
        book = str(item.get("book") or default_book).strip()
        refs.append((book, int(item["chapter"]), int(item["verse"])))
    return refs
