"""Detection pipeline: BM25 (lexical) + LaBSE (semantic later). Uses CLASSLA for lemmatization."""

from __future__ import annotations

import re

import joblib

from config import DATA_DIR
from models.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BibleRef,
    ConfidenceType,
    MatchFragment,
    OTNTSummary,
)

# Lazy-loaded state
_bm25_index: dict | None = None
_classla_pipeline = None

# Word tokens (Cyrillic + Latin); must match build_bm25_index.py
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# New Testament book name substrings (Serbian corpus)
_NT_MARKERS = (
    "Јеванђеље",
    "Jevanđelje",
    "Јеванђеље од",
    "Посланица",
    "Дјела ",
    "Дјела Светих",
    "Откривење",
    "Отк –",
)


def _tokenize(text: str) -> list[str]:
    """Return list of word tokens (same as in build_bm25_index.py)."""
    if not isinstance(text, str) or not text.strip():
        return []
    return _TOKEN_RE.findall(text)


def _get_bm25_index() -> dict:
    """Load and cache the BM25 index."""
    global _bm25_index
    if _bm25_index is None:
        path = DATA_DIR / "bible" / "bm25_index.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {path}. Run: python scripts/build_bm25_index.py"
            )
        _bm25_index = joblib.load(path)
    return _bm25_index


def _get_classla_pipeline():
    """Load and cache the CLASSLA pipeline for Serbian lemmatization."""
    global _classla_pipeline
    if _classla_pipeline is None:
        import classla
        classla.download("sr")
        _classla_pipeline = classla.Pipeline("sr", processors="tokenize,pos,lemma", use_gpu=False)
    return _classla_pipeline


def _lemmatize_text(text: str) -> str:
    """Return space-separated lemmas for the given text."""
    if not isinstance(text, str) or not text.strip():
        return ""
    pipeline = _get_classla_pipeline()
    doc = pipeline(text)
    lemmas = []
    for sent in doc.sentences:
        for word in sent.words:
            lemma = (word.lemma or word.text).strip()
            if lemma:
                lemmas.append(lemma)
    return " ".join(lemmas)


def _is_new_testament(book: str) -> bool:
    """Heuristic: True if book name looks like NT (Gospels, Acts, Epistles, Revelation)."""
    if not isinstance(book, str):
        return False
    b = book.strip()
    return any(marker in b for marker in _NT_MARKERS)


def run_lexical_search(text: str, top_k: int = 20) -> tuple[list[MatchFragment], OTNTSummary]:
    """
    Lemmatize input, run BM25 retrieval, return match fragments and OT/NT counts.
    """
    index = _get_bm25_index()
    bm25 = index["bm25"]
    verses_df = index["verses"]

    query_lemma = _lemmatize_text(text)
    query_tokens = _tokenize(query_lemma)
    if not query_tokens:
        return [], OTNTSummary()

    scores = bm25.get_scores(query_tokens)
    top_indices = scores.argsort()[-top_k:][::-1]

    # Normalize scores to [0, 1] for API (BM25 raw scores are unbounded)
    max_score = float(scores.max()) if len(scores) else 0.0
    scale = max_score if max_score > 0 else 1.0

    snippet_end = min(300, len(text))
    input_snippet = (text[:snippet_end] + ("..." if len(text) > snippet_end else "")).strip()

    matches = []
    ot_count = 0
    nt_count = 0

    for idx in top_indices:
        raw = float(scores[idx])
        if raw <= 0:
            continue
        score = round(min(1.0, raw / scale), 4)

        row = verses_df.iloc[idx]
        book = str(row["book"])
        chapter = int(row["chapter"])
        verse = int(row["verse"])
        verse_text = str(row["text"])

        if _is_new_testament(book):
            nt_count += 1
        else:
            ot_count += 1

        matches.append(
            MatchFragment(
                start=0,
                end=snippet_end,
                input_snippet=input_snippet,
                bible_ref=BibleRef(book=book, chapter=chapter, verse=verse, text=verse_text),
                confidence_type=ConfidenceType.LEXICAL,
                score=score,
            )
        )

    summary = OTNTSummary(old_testament=ot_count, new_testament=nt_count)
    return matches, summary


def detect(request: AnalyzeRequest) -> AnalyzeResponse:
    """Run lexical (BM25) detection on the input text and return matches + summary."""
    matches, summary = run_lexical_search(request.text, top_k=20)
    message = "Lexical (BM25) matches." if matches else "No lexical matches above threshold."
    return AnalyzeResponse(matches=matches, summary=summary, message=message)
