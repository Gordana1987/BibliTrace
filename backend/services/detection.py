"""Detection pipeline: BM25 → Qwen3 semantic rerank (LaBSE on-demand). NZ-only; per-corpus results."""

from __future__ import annotations

import re

import joblib
import numpy as np

from config import (
    ACTIVE_CORPORA,
    CORPUS_LABELS,
    CROSS_ENCODER_ENABLED,
    CROSS_ENCODER_MODEL,
    CROSS_ENCODER_POOL,
    DATA_DIR,
    DK_NT_BOOKS,
    INACTIVE_CORPORA,
    QUERY_ENCODE_MODE,
    QUERY_ENCODE_MODES,
    QUERY_EXPANSION_ENABLED,
    QUERY_SYNONYMS,
    RESTRICT_TO_NEW_TESTAMENT,
)
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
_cross_encoder_model = None

# Maps API corpus key to data directory folder name (matches data/<id>/)
_CORPUS_DIR = {c: c for c in ACTIVE_CORPORA}
_CORPUS_BUILD_ARG = dict(_CORPUS_DIR)

# Word tokens (Cyrillic + Latin); must match build_bm25_index.py
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Legacy substring markers (SPC / older naming). Prefer DK_NT_BOOKS for DK.
_NT_MARKERS = (
    "Јеванђеље",
    "Jevanđelje",
    "Посланица",
    "посланица",
    "Дјела ",
    "Дела ",
    "Откривење",
    "Отк –",
)

# Hybrid retrieval
_BM25_CANDIDATES = 200
_SEMANTIC_TOP_K = 20
# Floor for phrase matches when embedding cosine is ≤ 0 (still returned as lexical hits).
_PHRASE_SCORE_FLOOR = 0.001


def _tokenize(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    return _TOKEN_RE.findall(text)


def expand_query(text: str, *, enabled: bool | None = None) -> str:
    """
    Append surface synonyms for known tokens (e.g. реч → слово).

    Used before BM25 and dense encode when QUERY_EXPANSION_ENABLED (or enabled=True).
    Phrase matching should keep the original user text.
    """
    if enabled is None:
        enabled = QUERY_EXPANSION_ENABLED
    if not enabled or not isinstance(text, str) or not text.strip():
        return text

    tokens = _TOKEN_RE.findall(text)
    if not tokens:
        return text

    original_keys = {t.casefold() for t in tokens}
    extras: list[str] = []
    seen_extra: set[str] = set()
    for tok in tokens:
        for syn in QUERY_SYNONYMS.get(tok.casefold(), ()):
            key = syn.casefold()
            if key in original_keys or key in seen_extra:
                continue
            extras.append(syn)
            seen_extra.add(key)

    if not extras:
        return text
    return f"{text.rstrip()} {' '.join(extras)}"


def _get_bm25_index(corpus: str = "dk") -> dict:
    global _bm25_indexes
    if corpus not in _bm25_indexes:
        folder = _CORPUS_DIR.get(corpus, corpus)
        path = DATA_DIR / folder / "bm25_index.joblib"
        if not path.exists():
            cli = _CORPUS_BUILD_ARG.get(corpus, corpus)
            raise FileNotFoundError(
                f"BM25 index not found at {path}. Run: python scripts/build_bm25_index.py --corpus {cli}"
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
    """Lemmatize via CLASSLA after Cyrillic→Latin (sr models are Latin-keyed)."""
    if not isinstance(text, str) or not text.strip():
        return ""
    from services.transliterate import cyrillic_to_latin

    lat = cyrillic_to_latin(text)
    doc = _get_classla_pipeline()(lat)
    return " ".join(
        (w.lemma or w.text).strip()
        for s in doc.sentences
        for w in s.words
        if (w.lemma or w.text).strip()
    )


def _is_new_testament(book: str) -> bool:
    """True if book is New Testament (DK exact names, else legacy markers)."""
    if not isinstance(book, str):
        return False
    b = book.strip()
    if b in DK_NT_BOOKS:
        return True
    return any(m in b for m in _NT_MARKERS)


def _book_allowed(book: str) -> bool:
    """Whether a verse book may appear in live results (NZ-only when configured)."""
    if not RESTRICT_TO_NEW_TESTAMENT:
        return True
    return _is_new_testament(book)


def _filter_indices_to_allowed(indices: list[int], corpus: str) -> list[int]:
    if not RESTRICT_TO_NEW_TESTAMENT or not indices:
        return indices
    verses = _get_bm25_index(corpus)["verses"]
    out: list[int] = []
    for i in indices:
        book = str(verses.iloc[i].get("book", "")).strip()
        if _book_allowed(book):
            out.append(i)
    return out


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
    return _filter_indices_to_allowed(mask[mask].index.tolist(), corpus)


def _get_bm25_candidates(text: str, corpus: str = "dk", top_k: int = _BM25_CANDIDATES) -> list[int]:
    """Return top BM25 candidate indices using both lemmatized and raw tokens.

    When RESTRICT_TO_NEW_TESTAMENT is set, walks the full BM25 ranking until top_k
    New Testament verses are collected (not merely the global top_k then filtered).
    """
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
    # Full descending order so NZ filter can still fill top_k from deeper ranks.
    bm25_order = scores.argsort()[::-1].tolist()
    phrase_indices = _get_phrase_match_indices(text, corpus)
    seen_idx = set(phrase_indices)
    merged = phrase_indices + [i for i in bm25_order if i not in seen_idx]
    if RESTRICT_TO_NEW_TESTAMENT:
        merged = _filter_indices_to_allowed(merged, corpus)
    return merged[:top_k]


def get_bm25_ranked_pool(
    text: str,
    corpus: str = "dk",
    pool_size: int = _BM25_CANDIDATES,
    expand: bool | None = None,
) -> list[dict]:
    """BM25 + phrase-match candidate pool with 1-based ranks (no semantic rerank).

    Order matches _get_bm25_candidates: phrase hits first, then BM25 by score.
    """
    search_text = expand_query(text, enabled=expand)
    candidate_indices = _get_bm25_candidates(search_text, corpus, top_k=pool_size)
    if not candidate_indices:
        return []

    idx = _get_bm25_index(corpus)
    verses_df = idx["verses"]
    # Phrase match stays on the original user string.
    phrase_indices = set(_get_phrase_match_indices(text, corpus))

    lemma = _lemmatize_text(search_text)
    tokens_lemma = _tokenize(lemma)
    tokens_raw = _tokenize(search_text)
    seen_tok: set[str] = set()
    tokens: list[str] = []
    for t in tokens_lemma + tokens_raw:
        if t and t not in seen_tok:
            seen_tok.add(t)
            tokens.append(t)
    all_scores = idx["bm25"].get_scores(tokens) if tokens else None

    pool: list[dict] = []
    for rank, idx_row in enumerate(candidate_indices, start=1):
        row = verses_df.iloc[idx_row]
        book = str(row.get("book", "")).strip()
        try:
            chapter = int(row["chapter"])
            verse = int(row["verse"])
        except (TypeError, ValueError):
            chapter = verse = 0
        is_phrase = idx_row in phrase_indices
        pool.append(
            {
                "rank": rank,
                "book": book,
                "chapter": chapter,
                "verse": verse,
                "verse_text": str(row.get("text", "")),
                "bm25_score": float(all_scores[idx_row]) if all_scores is not None and not is_phrase else None,
                "is_phrase": is_phrase,
                "corpus": corpus,
                "verse_index": idx_row,
            }
        )
    return pool


def _get_qwen_index(corpus: str = "dk") -> dict:
    global _qwen_indexes
    if corpus not in _qwen_indexes:
        folder = _CORPUS_DIR.get(corpus, corpus)
        path = DATA_DIR / folder / "qwen_embeddings.joblib"
        if not path.exists():
            cli = _CORPUS_BUILD_ARG.get(corpus, corpus)
            raise FileNotFoundError(
                f"Qwen embeddings not found at {path}. Run: python scripts/build_embeddings.py qwen --corpus {cli}"
            )
        _qwen_indexes[corpus] = joblib.load(path)
    return _qwen_indexes[corpus]


def _get_qwen_model():
    global _qwen_model
    if _qwen_model is None:
        from sentence_transformers import SentenceTransformer
        _qwen_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cpu")
    return _qwen_model


def encode_query_qwen(text: str, mode: str | None = None) -> np.ndarray:
    """
    Encode a search query for Qwen dense retrieval.

    Modes (see config.QUERY_ENCODE_MODES):
      query — retrieval prompt (default, current live behaviour)
      doc   — same path as verse documents (no prompt)
      mean  — L2-normalized average of query + doc encodings
    Returns shape (1, dim), float32, L2-normalized.
    """
    mode = (mode or QUERY_ENCODE_MODE).strip().lower()
    if mode not in QUERY_ENCODE_MODES:
        raise ValueError(f"Unknown query encode mode {mode!r}; expected one of {QUERY_ENCODE_MODES}")
    text = (text or "").strip()
    if not text:
        raise ValueError("Cannot encode empty query")

    mdl = _get_qwen_model()
    if mode == "query":
        v = mdl.encode([text], prompt_name="query", normalize_embeddings=True)
    elif mode == "doc":
        v = mdl.encode([text], normalize_embeddings=True)
    else:  # mean
        vq = mdl.encode([text], prompt_name="query", normalize_embeddings=True)
        vd = mdl.encode([text], normalize_embeddings=True)
        v = np.asarray(vq, dtype=np.float32) + np.asarray(vd, dtype=np.float32)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        v = v / norms
        return v.astype(np.float32)
    return np.asarray(v, dtype=np.float32)


def _get_labse_index(corpus: str = "dk") -> dict:
    global _labse_indexes
    if corpus not in _labse_indexes:
        folder = _CORPUS_DIR.get(corpus, corpus)
        path = DATA_DIR / folder / "labse_embeddings.joblib"
        if not path.exists():
            cli = _CORPUS_BUILD_ARG.get(corpus, corpus)
            raise FileNotFoundError(
                f"LaBSE embeddings not found at {path}. Run: python scripts/build_embeddings.py labse --corpus {cli}"
            )
        _labse_indexes[corpus] = joblib.load(path)
    return _labse_indexes[corpus]


def _get_labse_model():
    global _labse_model
    if _labse_model is None:
        from sentence_transformers import SentenceTransformer
        _labse_model = SentenceTransformer("sentence-transformers/LaBSE", device="cpu")
    return _labse_model


def _get_cross_encoder_model():
    global _cross_encoder_model
    if _cross_encoder_model is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder_model = CrossEncoder(CROSS_ENCODER_MODEL, device="cpu")
    return _cross_encoder_model


def _verse_row_to_text(row) -> str | None:
    verse_text_raw = str(row["text"])
    verse_text = re.sub(r"[*†]+", "", verse_text_raw).strip()
    if not verse_text:
        return None
    lower_text = verse_text.lower()
    if lower_text.startswith("pages:") or verse_text.isdigit():
        return None
    return verse_text


# --- HyDE / full-corpus dense diagnostics (paused for live path; scripts may still import) ---
def get_dense_ranked_pool(
    text: str,
    corpus: str = "dk",
    pool_size: int = _BM25_CANDIDATES,
    encode_mode: str | None = None,
    expand: bool | None = None,
) -> list[dict]:
    """Qwen dense retrieval over the full precomputed embedding matrix (no BM25 filter).

    Diagnostic helper (HyDE / dense-only / query-encode A/B). Not used by live detect().
    """
    if not text or not text.strip():
        return []

    search_text = expand_query(text, enabled=expand)
    idx = _get_qwen_index(corpus)
    q_emb = encode_query_qwen(search_text, mode=encode_mode)

    embs = idx["embeddings"]
    verses_df = idx["verses"]
    dup_mask = verses_df.duplicated(subset=["book", "chapter", "verse"], keep="first")
    scores = np.dot(embs, q_emb.ravel())
    order = scores.argsort()[::-1]

    pool: list[dict] = []
    seen_refs: set[tuple[str, int, int]] = set()
    for idx_row in order:
        if len(pool) >= pool_size:
            break
        if 0 <= idx_row < len(dup_mask) and bool(dup_mask.iloc[idx_row]):
            continue
        row = verses_df.iloc[idx_row]
        book = str(row.get("book", "")).strip()
        if not book or not _book_allowed(book):
            continue
        try:
            chapter = int(row["chapter"])
            verse = int(row["verse"])
        except (TypeError, ValueError):
            continue
        ref_key = (book, chapter, verse)
        if ref_key in seen_refs:
            continue
        verse_text = _verse_row_to_text(row)
        if verse_text is None:
            continue
        seen_refs.add(ref_key)
        pool.append(
            {
                "rank": len(pool) + 1,
                "book": book,
                "chapter": chapter,
                "verse": verse,
                "verse_text": verse_text,
                "cosine_score": float(scores[idx_row]),
                "corpus": corpus,
                "verse_index": int(idx_row),
            }
        )
    return pool


def dense_retrieval_timings(
    text: str,
    corpus: str = "dk",
    repeats: int = 5,
    encode_mode: str | None = None,
) -> dict:
    """Benchmark query encode, full-matrix dot product, and top-k argsort (ms)."""
    import time

    idx = _get_qwen_index(corpus)
    embs = idx["embeddings"]
    n, dim = embs.shape
    top_k = _BM25_CANDIDATES
    mode = encode_mode or QUERY_ENCODE_MODE

    # Warm-up
    q_emb = encode_query_qwen(text, mode=mode)
    scores = np.dot(embs, q_emb.ravel())
    _ = scores.argsort()[::-1][:top_k]

    encode_ms: list[float] = []
    dot_ms: list[float] = []
    sort_ms: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        q_emb = encode_query_qwen(text, mode=mode)
        t1 = time.perf_counter()
        scores = np.dot(embs, q_emb.ravel())
        t2 = time.perf_counter()
        _ = scores.argsort()[::-1][:top_k]
        t3 = time.perf_counter()
        encode_ms.append((t1 - t0) * 1000)
        dot_ms.append((t2 - t1) * 1000)
        sort_ms.append((t3 - t2) * 1000)

    def _stats(vals: list[float]) -> dict:
        return {
            "min_ms": round(min(vals), 3),
            "max_ms": round(max(vals), 3),
            "mean_ms": round(sum(vals) / len(vals), 3),
        }

    return {
        "corpus": corpus,
        "encode_mode": mode,
        "matrix_shape": [int(n), int(dim)],
        "repeats": repeats,
        "query_encode": _stats(encode_ms),
        "full_matrix_dot": _stats(dot_ms),
        "argsort_top_k": _stats(sort_ms),
    }


def _build_semantic_pool(
    text: str,
    candidate_indices: list[int],
    model_name: str,
    corpus: str = "dk",
    phrase_match_indices: set[int] | None = None,
    pool_size: int = _SEMANTIC_TOP_K,
) -> list[dict]:
    """Score BM25 candidates with embeddings; return top pool for final ranking."""
    if not candidate_indices:
        return []
    phrase_match_indices = phrase_match_indices or set()

    if model_name == "qwen":
        idx = _get_qwen_index(corpus)
        q_emb = encode_query_qwen(text)
    else:
        idx = _get_labse_index(corpus)
        mdl = _get_labse_model()
        q_emb = mdl.encode([text], normalize_embeddings=True)

    embs = idx["embeddings"]
    verses_df = idx["verses"]
    dup_mask = verses_df.duplicated(subset=["book", "chapter", "verse"], keep="first")
    cand_embs = embs[candidate_indices]
    scores = np.dot(cand_embs, q_emb.ravel())

    phrase_in_cand = [i for i in range(len(candidate_indices)) if candidate_indices[i] in phrase_match_indices]
    other = [i for i in range(len(candidate_indices)) if candidate_indices[i] not in phrase_match_indices]
    phrase_order = sorted(phrase_in_cand, key=lambda i: float(scores[i]), reverse=True)
    other_order = sorted(other, key=lambda i: float(scores[i]), reverse=True)
    full_order = phrase_order + other_order

    pool: list[dict] = []
    seen_refs: set[tuple[str, int, int]] = set()
    for pos in full_order:
        if len(pool) >= pool_size:
            break
        raw = float(scores[pos])
        is_phrase = candidate_indices[pos] in phrase_match_indices
        if raw <= 0 and not is_phrase:
            continue
        idx_row = candidate_indices[pos]
        if 0 <= idx_row < len(dup_mask) and bool(dup_mask.iloc[idx_row]):
            continue
        row = verses_df.iloc[idx_row]
        book = str(row.get("book", "")).strip()
        if not book or not _book_allowed(book):
            continue
        try:
            chapter = int(row["chapter"])
            verse = int(row["verse"])
        except (TypeError, ValueError):
            continue
        ref_key = (book, chapter, verse)
        if ref_key in seen_refs:
            continue
        verse_text = _verse_row_to_text(row)
        if verse_text is None:
            continue
        seen_refs.add(ref_key)
        pool.append(
            {
                "book": book,
                "chapter": chapter,
                "verse": verse,
                "verse_text": verse_text,
                "is_phrase": is_phrase,
                "embed_score": raw if raw > 0 else _PHRASE_SCORE_FLOOR,
            }
        )
    return pool


def _pool_to_matches(
    text: str,
    pool: list[dict],
    corpus: str = "dk",
    *,
    score_key: str = "embed_score",
    top_k: int | None = None,
) -> tuple[list[MatchFragment], OTNTSummary]:
    """Turn ranked pool into MatchFragment list (embedding or CE scores)."""
    if not pool:
        return [], OTNTSummary()

    limit = top_k if top_k is not None else len(pool)
    snippet_end = min(300, len(text))
    input_snippet = (text[:snippet_end] + ("..." if len(text) > snippet_end else "")).strip()

    matches: list[MatchFragment] = []
    ot_count = nt_count = 0
    for item in pool[:limit]:
        book = item["book"]
        if _is_new_testament(book):
            nt_count += 1
        else:
            ot_count += 1
        matches.append(
            MatchFragment(
                start=0,
                end=snippet_end,
                input_snippet=input_snippet,
                bible_ref=BibleRef(
                    book=book,
                    chapter=item["chapter"],
                    verse=item["verse"],
                    text=item["verse_text"],
                ),
                confidence_type=ConfidenceType.LEXICAL if item["is_phrase"] else ConfidenceType.SEMANTIC,
                score=float(item[score_key]),
                corpus=corpus,
            )
        )
    return matches, OTNTSummary(old_testament=ot_count, new_testament=nt_count)


def _run_cross_encoder_rerank(
    text: str,
    pool: list[dict],
    corpus: str = "dk",
    top_k: int = _SEMANTIC_TOP_K,
) -> tuple[list[MatchFragment], OTNTSummary]:
    """Rerank embedding pool with a cross-encoder; return top_k matches."""
    if not pool:
        return [], OTNTSummary()

    model = _get_cross_encoder_model()
    pairs = [(text, item["verse_text"]) for item in pool]
    ce_scores = model.predict(pairs, batch_size=16, show_progress_bar=False)
    order = sorted(range(len(pool)), key=lambda i: float(ce_scores[i]), reverse=True)

    ranked: list[dict] = []
    for i in order:
        item = dict(pool[i])
        item["ce_score"] = float(ce_scores[i])
        ranked.append(item)
    return _pool_to_matches(text, ranked, corpus, score_key="ce_score", top_k=top_k)


def score_pool_with_cross_encoder(text: str, pool: list[dict]) -> list[dict]:
    """Return a copy of pool sorted by CE score (diagnostic helper)."""
    if not pool:
        return []
    model = _get_cross_encoder_model()
    pairs = [(text, item["verse_text"]) for item in pool]
    ce_scores = model.predict(pairs, batch_size=16, show_progress_bar=False)
    ranked: list[dict] = []
    for i, item in enumerate(pool):
        row = dict(item)
        row["ce_score"] = float(ce_scores[i])
        ranked.append(row)
    ranked.sort(key=lambda r: r["ce_score"], reverse=True)
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


def _run_semantic_rerank(
    text: str,
    candidate_indices: list[int],
    model_name: str,
    corpus: str = "dk",
    phrase_match_indices: set[int] | None = None,
) -> tuple[list[MatchFragment], OTNTSummary]:
    """BM25 candidates → embedding shortlist → optional CE → top_k."""
    use_ce = CROSS_ENCODER_ENABLED
    pool_size = CROSS_ENCODER_POOL if use_ce else _SEMANTIC_TOP_K
    pool = _build_semantic_pool(
        text,
        candidate_indices,
        model_name,
        corpus,
        phrase_match_indices,
        pool_size=pool_size,
    )
    if use_ce:
        return _run_cross_encoder_rerank(text, pool, corpus, top_k=_SEMANTIC_TOP_K)
    return _pool_to_matches(text, pool, corpus, top_k=_SEMANTIC_TOP_K)


def _normalize_scores(matches: list[MatchFragment]) -> None:
    """Normalize scores in-place so the top result = 1.0 (per corpus)."""
    if not matches:
        return
    max_score = max(m.score for m in matches)
    scale = max_score if max_score > 0 else 1.0
    for m in matches:
        m.score = round(min(1.0, m.score / scale), 4)


def _rank_corpus_matches(matches: list[MatchFragment], top_k: int = _SEMANTIC_TOP_K) -> list[MatchFragment]:
    """Lexical (phrase) hits first, then semantic by score — within one corpus only."""
    lexical = sorted(
        (m for m in matches if m.confidence_type == ConfidenceType.LEXICAL),
        key=lambda m: m.score,
        reverse=True,
    )
    semantic = sorted(
        (m for m in matches if m.confidence_type != ConfidenceType.LEXICAL),
        key=lambda m: m.score,
        reverse=True,
    )
    return (lexical + semantic)[:top_k]


def _detect_corpus(
    text: str,
    corpus: str,
    compare_with_labse: bool,
) -> tuple[list[MatchFragment], OTNTSummary, list[MatchFragment] | None]:
    """Run full detection pipeline for a single corpus."""
    search_text = expand_query(text)
    candidates = _get_bm25_candidates(search_text, corpus)
    if not candidates:
        return [], OTNTSummary(), None
    # Exact phrase boost uses the unmodified user string.
    phrase_matches = set(_get_phrase_match_indices(text, corpus))
    matches_qwen, summary = _run_semantic_rerank(
        search_text, candidates, "qwen", corpus, phrase_matches
    )
    matches_qwen = _rank_corpus_matches(matches_qwen, _SEMANTIC_TOP_K)
    _normalize_scores(matches_qwen)
    labse_matches = None
    if compare_with_labse:
        labse_matches, _ = _run_semantic_rerank(
            search_text, candidates, "labse", corpus, phrase_matches
        )
        labse_matches = _rank_corpus_matches(labse_matches, _SEMANTIC_TOP_K)
        _normalize_scores(labse_matches)
    return matches_qwen, summary, labse_matches


def _active_only(corpora: list[str]) -> list[str]:
    """Drop inactive corpora; default to dk if nothing left."""
    active = [c for c in corpora if c in ACTIVE_CORPORA]
    return active if active else ["dk"]


def _corpora_from_request(request: AnalyzeRequest) -> list[str]:
    """Resolve corpus list from corpora[] or legacy version field."""
    if request.version is not None:
        if request.version in ("both", "all"):
            return list(ACTIVE_CORPORA)
        return _active_only([request.version])
    if request.corpora:
        seen: list[str] = []
        for c in request.corpora:
            if c not in seen:
                seen.append(c)
        return _active_only(seen if seen else ["dk"])
    return ["dk"]


def detect(request: AnalyzeRequest, compare_with_labse: bool = False) -> AnalyzeResponse:
    """
    BM25 → Qwen3 shortlist → optional CE top-k. Optional LaBSE for comparison.
    Live scope: New Testament only; each selected corpus is searched separately (no merge).
    """
    text = request.text.strip()
    if not text:
        return AnalyzeResponse(message="No text provided.")

    corpora = _corpora_from_request(request)

    matches_by_corpus: dict[str, list[MatchFragment]] = {}
    labse_by_corpus: dict[str, list[MatchFragment]] = {}
    ot_total = nt_total = 0
    any_hits = False

    for corpus in corpora:
        try:
            q_matches, summary, l_matches = _detect_corpus(text, corpus, compare_with_labse)
        except FileNotFoundError:
            matches_by_corpus[corpus] = []
            continue
        matches_by_corpus[corpus] = q_matches
        ot_total += summary.old_testament
        nt_total += summary.new_testament
        if q_matches:
            any_hits = True
        if l_matches:
            labse_by_corpus[corpus] = l_matches

    if not any_hits:
        return AnalyzeResponse(
            message="No lexical candidates. Enter Cyrillic text.",
            matches_by_corpus=matches_by_corpus,
        )

    labels = [CORPUS_LABELS.get(c, c) for c in corpora]
    ce_note = " + CE" if CROSS_ENCODER_ENABLED else ""
    msg = f"Qwen3{ce_note} semantic matches (NZ) — {', '.join(labels)}."
    single = len(corpora) == 1
    return AnalyzeResponse(
        matches=matches_by_corpus[corpora[0]] if single else [],
        matches_by_corpus=matches_by_corpus,
        summary=OTNTSummary(old_testament=ot_total, new_testament=nt_total),
        message=msg,
        labse_matches=(labse_by_corpus.get(corpora[0]) if single and labse_by_corpus else None),
        labse_matches_by_corpus=labse_by_corpus if labse_by_corpus else None,
    )
