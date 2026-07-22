"""Shared helpers for concept-search modes (exact / lemma / semantic)."""

from __future__ import annotations

import re
from functools import lru_cache

import pandas as pd

from config import (
    ACTIVE_CORPORA,
    DATA_DIR,
    DK_NT_BOOK_ORDER,
    DK_NT_BOOKS,
    RESTRICT_TO_NEW_TESTAMENT,
    SEARCH_MAX_TERM_TOKENS,
    SEARCH_PAGE_SIZE,
)
from models.schemas import CorpusSearchResult, SearchHit, SearchRanking

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
# Liturgical / typographic marks in verse text (not used for query wildcard parsing).
_LITURGICAL_RE = re.compile(r"[*†„""«»]")
# Exact-mode pattern token: optional leading/trailing * around a word.
_EXACT_PATTERN_RE = re.compile(r"^(\*)?(\w+)(\*)?$", re.UNICODE)


def normalize_surface(text: str) -> str:
    """Strip liturgical/typographic marks; collapse whitespace."""
    if not isinstance(text, str):
        return ""
    cleaned = _LITURGICAL_RE.sub("", text)
    return " ".join(cleaned.split())


def tokenize_surface(text: str) -> list[str]:
    """Case-insensitive word tokens (Cyrillic/Latin). Each token is a word boundary unit."""
    return [t.casefold() for t in _TOKEN_RE.findall(normalize_surface(text))]


def parse_term_tokens(term: str) -> list[str]:
    """
    Parse user term into 1..SEARCH_MAX_TERM_TOKENS surface tokens.

    Raises ValueError if empty or too many tokens.
    """
    tokens = tokenize_surface(term)
    if not tokens:
        raise ValueError("Потребна је бар једна реч.")
    if len(tokens) > SEARCH_MAX_TERM_TOKENS:
        raise ValueError(
            f"Појам може имати највише {SEARCH_MAX_TERM_TOKENS} речи "
            f"(унето {len(tokens)})."
        )
    return tokens


def parse_exact_patterns(term: str) -> list[str]:
    """
    Parse exact-mode query into 1..SEARCH_MAX_TERM_TOKENS patterns.

    Patterns (casefolded):
      word      — exact token
      word*     — prefix (token startswith word)
      *word     — suffix (token endswith word)
      *word*    — contains (token contains word)

    Raises ValueError on empty / too many tokens / invalid pattern (e.g. lone *).
    """
    if not isinstance(term, str) or not term.strip():
        raise ValueError("Потребна је бар једна реч.")

    # Keep * for wildcards; strip other typographic quotes if present.
    cleaned = term.replace("„", "").replace("“", "").replace("”", "").replace("«", "").replace("»", "")
    cleaned = cleaned.replace("†", "")
    parts = cleaned.strip().split()
    if not parts:
        raise ValueError("Потребна је бар једна реч.")
    if len(parts) > SEARCH_MAX_TERM_TOKENS:
        raise ValueError(
            f"Појам може имати највише {SEARCH_MAX_TERM_TOKENS} речи "
            f"(унето {len(parts)})."
        )

    patterns: list[str] = []
    for part in parts:
        p = part.casefold()
        if p == "*" or p == "**":
            raise ValueError("Звездица * мора ићи уз реч (нпр. опрост* или *ост).")
        m = _EXACT_PATTERN_RE.match(p)
        if not m:
            raise ValueError(
                f"Неисправан образац {part!r}. "
                "Користите реч, реч*, *реч или *реч*."
            )
        patterns.append(p)
    return patterns


def token_matches_pattern(token: str, pattern: str) -> bool:
    """Match one verse token against an exact-mode pattern (see parse_exact_patterns)."""
    if not token or not pattern:
        return False
    leading = pattern.startswith("*")
    trailing = pattern.endswith("*")
    core = pattern.strip("*")
    if not core:
        return False
    if leading and trailing:
        return core in token
    if trailing:
        return token.startswith(core)
    if leading:
        return token.endswith(core)
    return token == core


def consecutive_token_match(haystack_tokens: list[str], needle_tokens: list[str]) -> bool:
    """True if needle appears as a contiguous token sequence in haystack (word-boundary safe)."""
    if not needle_tokens or not haystack_tokens:
        return False
    n = len(needle_tokens)
    if n > len(haystack_tokens):
        return False
    for i in range(len(haystack_tokens) - n + 1):
        if haystack_tokens[i : i + n] == needle_tokens:
            return True
    return False


def consecutive_pattern_match(haystack_tokens: list[str], patterns: list[str]) -> bool:
    """Like consecutive_token_match, but each needle slot may be a wildcard pattern."""
    if not patterns or not haystack_tokens:
        return False
    n = len(patterns)
    if n > len(haystack_tokens):
        return False
    for i in range(len(haystack_tokens) - n + 1):
        if all(token_matches_pattern(haystack_tokens[i + j], patterns[j]) for j in range(n)):
            return True
    return False


_classla_pipeline = None


def _get_classla_pipeline():
    global _classla_pipeline
    if _classla_pipeline is None:
        import classla

        classla.download("sr")
        _classla_pipeline = classla.Pipeline(
            "sr", processors="tokenize,pos,lemma", use_gpu=False
        )
    return _classla_pipeline


def lemmatize_to_tokens(text: str) -> list[str]:
    """
    CLASSLA lemmatize → casefolded word tokens (same tokenizer as corpus index).

    Cyrillic is transliterated to Latin first (CLASSLA sr is Latin-keyed).
    Empty / non-string → []. Used for lemma-mode queries.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    from services.transliterate import cyrillic_to_latin

    lat = cyrillic_to_latin(text)
    doc = _get_classla_pipeline()(lat)
    lemmas = [
        (w.lemma or w.text).strip()
        for s in doc.sentences
        for w in s.words
        if (w.lemma or w.text).strip()
    ]
    return tokenize_surface(" ".join(lemmas))


def parse_term_lemmas(term: str) -> list[str]:
    """Validate term length on surface tokens, then return query lemma token sequence."""
    parse_term_tokens(term)  # enforces 1..MAX surface tokens
    lemmas = lemmatize_to_tokens(term)
    if not lemmas:
        raise ValueError("Није могуће лематизовати појам.")
    if len(lemmas) > SEARCH_MAX_TERM_TOKENS:
        raise ValueError(
            f"Појам може имати највише {SEARCH_MAX_TERM_TOKENS} речи "
            f"(унето {len(lemmas)} лема)."
        )
    return lemmas


@lru_cache(maxsize=8)
def load_lemma_index(corpus: str) -> dict:
    """Load data/<corpus>/lemma_index.joblib (build with scripts/build_lemma_index.py)."""
    path = DATA_DIR / corpus / "lemma_index.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Lemma index not found at {path}. "
            f"Run: python scripts/build_lemma_index.py --corpus {corpus}"
        )
    import joblib

    return joblib.load(path)


@lru_cache(maxsize=8)
def load_verses_df(corpus: str) -> pd.DataFrame:
    """Load bible.csv for a corpus (raw surface text)."""
    path = DATA_DIR / corpus / "bible.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing corpus CSV: {path}")
    df = pd.read_csv(path)
    for col in ("book", "chapter", "verse", "text"):
        if col not in df.columns:
            raise ValueError(f"{path} missing column {col}")
    return df


def is_nt_book(book: str) -> bool:
    return isinstance(book, str) and book.strip() in DK_NT_BOOKS


def filter_nt_rows(df: pd.DataFrame) -> pd.DataFrame:
    if not RESTRICT_TO_NEW_TESTAMENT:
        return df
    return df[df["book"].astype(str).str.strip().isin(DK_NT_BOOKS)].copy()


_BOOK_RANK = {name: i for i, name in enumerate(DK_NT_BOOK_ORDER)}


def biblical_sort_key(row: dict) -> tuple[int, int, int]:
    book = str(row.get("book", "")).strip()
    try:
        chapter = int(row["chapter"])
        verse = int(row["verse"])
    except (TypeError, ValueError):
        chapter = verse = 0
    return (_BOOK_RANK.get(book, 10_000), chapter, verse)


def sort_hits_biblical(hits: list[dict]) -> list[dict]:
    return sorted(hits, key=biblical_sort_key)


def active_corpora_from_request(corpora: list[str] | None) -> list[str]:
    if not corpora:
        return list(ACTIVE_CORPORA[:1]) or ["dk"]
    seen: list[str] = []
    for c in corpora:
        if c in ACTIVE_CORPORA and c not in seen:
            seen.append(c)
    return seen if seen else (list(ACTIVE_CORPORA[:1]) or ["dk"])


def page_hits(
    all_hits: list[dict],
    *,
    corpus: str,
    offset: int,
    limit: int,
    ranking: SearchRanking,
) -> CorpusSearchResult:
    total = len(all_hits)
    page = all_hits[offset : offset + limit]
    hits = [
        SearchHit(
            book=str(h["book"]).strip(),
            chapter=int(h["chapter"]),
            verse=int(h["verse"]),
            text=str(h["text"]),
            corpus=corpus,
            score=h.get("score"),
        )
        for h in page
    ]
    return CorpusSearchResult(
        corpus=corpus,
        total=total,
        offset=offset,
        limit=limit,
        returned=len(hits),
        ranking=ranking,
        hits=hits,
    )


def default_limit(limit: int | None) -> int:
    return SEARCH_PAGE_SIZE if limit is None else limit
