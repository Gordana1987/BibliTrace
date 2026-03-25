"""Detection pipeline: BM25 (lexical) + Qwen3 (semantic, default) + LaBSE (semantic, on-demand compare)."""

from __future__ import annotations

import re

import joblib
import numpy as np

from config import DATA_DIR
from models.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BibleRef,
    ConfidenceType,
    MatchFragment,
    OTNTSummary,
)

# Lazy-loaded state — indexes keyed by corpus name, models shared across corpora
_bm25_indexes: dict[str, dict] = {}
_classla_pipeline = None
_qwen_indexes: dict[str, dict] = {}
_qwen_model = None
_labse_indexes: dict[str, dict] = {}
_labse_model = None

# Maps version/corpus name to data directory folder name
_CORPUS_DIR = {
    "dk": "bible",
    "bakotic": "bakotic",
}

# Word tokens (Cyrillic + Latin); must match build_bm25_index.py
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# New Testament book name substrings (covers both DK Ijekavian and Bakotić Ekavian forms)
_NT_MARKERS = (
    "Јеванђеље",
    "Jevanđelje",
    "Посланица",
    "посланица",
    "Дјела ",       # DK: Дјела Светих апостола
    "Дела ",        # Bakotić: Дела Апостолска
    "Откривење",
    "Отк –",
)

# Hybrid retrieval
_BM25_CANDIDATES = 200
_SEMANTIC_TOP_K = 20


def _tokenize(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    return _TOKEN_RE.findall(text)


def _get_bm25_index(corpus: str = "dk") -> dict:
    global _bm25_indexes
    if corpus not in _bm25_indexes:
        folder = _CORPUS_DIR.get(corpus, corpus)
        path = DATA_DIR / folder / "bm25_index.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {path}. Run: python scripts/build_bm25_index.py --corpus {corpus}"
            )
        _bm25_indexes[corpus] = joblib.load(path)
    return _bm25_indexes[corpus]


def _get_classla_pipeline():
    global _classla_pipeline
    if _classla_pipeline is None:
        import classla
        classla.download("sr")
        _classla_pipeline = classla.Pipeline("sr", processors="tokenize,pos,lemma", use_gpu=False)
    return _classla_pipeline


def _lemmatize_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    doc = _get_classla_pipeline()(text)
    return " ".join(
        (w.lemma or w.text).strip()
        for s in doc.sentences
        for w in s.words
        if (w.lemma or w.text).strip()
    )


def _is_new_testament(book: str) -> bool:
    if not isinstance(book, str):
        return False
    b = book.strip()
    return any(m in b for m in _NT_MARKERS)


def _get_phrase_match_indices(text: str, corpus: str = "dk") -> list[int]:
    """Return verse indices where the verse text contains the exact query phrase.
    Normalizes verse text before matching to strip liturgical markers and typographic quotes
    so e.g. „синови грома" in Bakotić matches the query 'синови грома'."""
    if not text or not text.strip():
        return []
    idx = _get_bm25_index(corpus)
    verses = idx["verses"]
    phrase = " ".join(text.strip().split())
    try:
        normalized = verses["text"].astype(str).str.replace(r'[*†„""]', '', regex=True)
        mask = normalized.str.contains(re.escape(phrase), case=False, na=False)
    except Exception:
        return []
    return mask[mask].index.tolist()


def _get_bm25_candidates(text: str, corpus: str = "dk", top_k: int = _BM25_CANDIDATES) -> list[int]:
    """Return top BM25 candidate indices using both lemmatized and raw tokens."""
    idx = _get_bm25_index(corpus)
    lemma = _lemmatize_text(text)
    tokens_lemma = _tokenize(lemma)
    tokens_raw = _tokenize(text)
    seen = set()
    tokens = []
    for t in tokens_lemma + tokens_raw:
        if t and t not in seen:
            seen.add(t)
            tokens.append(t)
    if not tokens:
        return []
    scores = idx["bm25"].get_scores(tokens)
    bm25_list = scores.argsort()[-top_k:][::-1].tolist()
    phrase_indices = _get_phrase_match_indices(text, corpus)
    seen_idx = set(phrase_indices)
    merged = phrase_indices + [i for i in bm25_list if i not in seen_idx]
    return merged[:top_k]


def _get_qwen_index(corpus: str = "dk") -> dict:
    global _qwen_indexes
    if corpus not in _qwen_indexes:
        folder = _CORPUS_DIR.get(corpus, corpus)
        path = DATA_DIR / folder / "qwen_embeddings.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"Qwen embeddings not found at {path}. Run: python scripts/build_embeddings.py qwen --corpus {corpus}"
            )
        _qwen_indexes[corpus] = joblib.load(path)
    return _qwen_indexes[corpus]


def _get_qwen_model():
    global _qwen_model
    if _qwen_model is None:
        from sentence_transformers import SentenceTransformer
        _qwen_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cpu")
    return _qwen_model


def _get_labse_index(corpus: str = "dk") -> dict:
    global _labse_indexes
    if corpus not in _labse_indexes:
        folder = _CORPUS_DIR.get(corpus, corpus)
        path = DATA_DIR / folder / "labse_embeddings.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"LaBSE embeddings not found at {path}. Run: python scripts/build_embeddings.py labse --corpus {corpus}"
            )
        _labse_indexes[corpus] = joblib.load(path)
    return _labse_indexes[corpus]


def _get_labse_model():
    global _labse_model
    if _labse_model is None:
        from sentence_transformers import SentenceTransformer
        _labse_model = SentenceTransformer("sentence-transformers/LaBSE", device="cpu")
    return _labse_model


def _run_semantic_rerank(
    text: str,
    candidate_indices: list[int],
    model_name: str,
    corpus: str = "dk",
    phrase_match_indices: set[int] | None = None,
) -> tuple[list[MatchFragment], OTNTSummary]:
    """Rerank BM25 candidates with Qwen or LaBSE; return top _SEMANTIC_TOP_K.
    Verses in phrase_match_indices (exact phrase in verse) are forced to the top."""
    if not candidate_indices:
        return [], OTNTSummary()
    phrase_match_indices = phrase_match_indices or set()

    if model_name == "qwen":
        idx = _get_qwen_index(corpus)
        mdl = _get_qwen_model()
        q_emb = mdl.encode([text], prompt_name="query", normalize_embeddings=True)
    else:
        idx = _get_labse_index(corpus)
        mdl = _get_labse_model()
        q_emb = mdl.encode([text], normalize_embeddings=True)

    embs = idx["embeddings"]
    verses_df = idx["verses"]
    dup_mask = verses_df.duplicated(subset=["book", "chapter", "verse"], keep="first")
    cand_embs = embs[candidate_indices]
    scores = np.dot(cand_embs, q_emb.ravel())
    # Build order: phrase matches first (by score among themselves), then rest by score
    phrase_in_cand = [i for i in range(len(candidate_indices)) if candidate_indices[i] in phrase_match_indices]
    other = [i for i in range(len(candidate_indices)) if candidate_indices[i] not in phrase_match_indices]
    phrase_order = sorted(phrase_in_cand, key=lambda i: float(scores[i]), reverse=True)
    other_order = sorted(other, key=lambda i: float(scores[i]), reverse=True)[:_SEMANTIC_TOP_K]
    order = (phrase_order + other_order)[:_SEMANTIC_TOP_K]

    snippet_end = min(300, len(text))
    input_snippet = (text[:snippet_end] + ("..." if len(text) > snippet_end else "")).strip()

    matches = []
    ot_count = nt_count = 0
    for pos in order:
        raw = float(scores[pos])
        if raw <= 0:
            continue
        idx_row = candidate_indices[pos]
        # Skip duplicate reference rows (likely section headlines).
        if 0 <= idx_row < len(dup_mask) and bool(dup_mask.iloc[idx_row]):
            continue
        row = verses_df.iloc[idx_row]
        book = str(row.get("book", "")).strip()
        if not book:
            continue
        try:
            chapter = int(row["chapter"])
            verse = int(row["verse"])
        except (TypeError, ValueError):
            # Bad or missing chapter/verse (e.g. stray "Pages:" rows) – skip.
            continue
        verse_text_raw = str(row["text"])
        # Clean display text: strip liturgical markers and pagination artefacts
        verse_text = re.sub(r"[*†]+", "", verse_text_raw).strip()
        if not verse_text:
            continue
        # Skip stray pagination blocks like "Pages:" / "1", "2", ...
        lower_text = verse_text.lower()
        if lower_text.startswith("pages:") or verse_text.isdigit():
            continue
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
                confidence_type=ConfidenceType.SEMANTIC,
                score=raw,  # raw cosine score — caller normalizes after merging
                corpus=corpus,
            )
        )
    return matches, OTNTSummary(old_testament=ot_count, new_testament=nt_count)


def _normalize_scores(matches: list[MatchFragment]) -> None:
    """Normalize scores in-place so the top result = 1.0.
    Called once after all corpora are merged so cross-corpus scores are comparable."""
    if not matches:
        return
    max_score = max(m.score for m in matches)
    scale = max_score if max_score > 0 else 1.0
    for m in matches:
        m.score = round(min(1.0, m.score / scale), 4)


def _detect_corpus(
    text: str,
    corpus: str,
    compare_with_labse: bool,
) -> tuple[list[MatchFragment], OTNTSummary, list[MatchFragment] | None]:
    """Run full detection pipeline for a single corpus."""
    candidates = _get_bm25_candidates(text, corpus)
    if not candidates:
        return [], OTNTSummary(), None
    phrase_matches = set(_get_phrase_match_indices(text, corpus))
    matches_qwen, summary = _run_semantic_rerank(text, candidates, "qwen", corpus, phrase_matches)
    labse_matches = None
    if compare_with_labse:
        labse_matches, _ = _run_semantic_rerank(text, candidates, "labse", corpus, phrase_matches)
    return matches_qwen, summary, labse_matches


def detect(request: AnalyzeRequest, compare_with_labse: bool = False) -> AnalyzeResponse:
    """
    BM25 candidates → Qwen3 rerank (primary). Optionally also LaBSE rerank for comparison.
    Supports version='dk', 'bakotic', or 'both'.
    """
    text = request.text.strip()
    if not text:
        return AnalyzeResponse(message="No text provided.")

    version = getattr(request, "version", "dk")
    corpora = ["dk", "bakotic"] if version == "both" else [version]

    all_qwen: list[MatchFragment] = []
    all_labse: list[MatchFragment] = []
    ot_total = nt_total = 0

    for corpus in corpora:
        q_matches, summary, l_matches = _detect_corpus(text, corpus, compare_with_labse)
        all_qwen.extend(q_matches)
        ot_total += summary.old_testament
        nt_total += summary.new_testament
        if l_matches:
            all_labse.extend(l_matches)

    # Sort and trim merged results, then normalize scores across the full pool
    all_qwen.sort(key=lambda m: m.score, reverse=True)
    all_qwen = all_qwen[:_SEMANTIC_TOP_K]
    _normalize_scores(all_qwen)

    if all_labse:
        all_labse.sort(key=lambda m: m.score, reverse=True)
        all_labse = all_labse[:_SEMANTIC_TOP_K]
        _normalize_scores(all_labse)

    if not all_qwen:
        return AnalyzeResponse(message="No lexical candidates. Enter Cyrillic text.")

    msg = "Qwen3 semantic matches." if all_qwen else "No semantic matches above threshold."
    return AnalyzeResponse(
        matches=all_qwen,
        summary=OTNTSummary(old_testament=ot_total, new_testament=nt_total),
        message=msg,
        labse_matches=all_labse if all_labse else None,
    )
