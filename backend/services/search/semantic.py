"""
Semantic concept search: meaning / motif via pure dense retrieval.

Live encoder: Embedić-large (e5-style).
  - queries:  "query: " + term   (full Cyrillic)
  - verses:   prebuilt with "passage: " prefix (embedic_large_nt_embeddings.joblib)
Per corpus, no merge; ranked by cosine; pool capped at SEARCH_SEMANTIC_POOL.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from config import (
    DATA_DIR,
    SEARCH_EMBED_INDEX_NAME,
    SEARCH_EMBED_MODEL_ID,
    SEARCH_EMBED_QUERY_PREFIX,
    SEARCH_SEMANTIC_POOL,
)
from models.schemas import CorpusSearchResult
from services.search.common import (
    normalize_surface,
    page_hits,
    parse_term_tokens,
)

_embed_model = None


@lru_cache(maxsize=8)
def load_embed_nt_index(corpus: str) -> dict:
    """Load data/<corpus>/embedic_*_nt_embeddings.joblib."""
    path = DATA_DIR / corpus / SEARCH_EMBED_INDEX_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"Embedić NZ embeddings not found at {path}. "
            f"Run: python scripts/build_embedic_nt_embeddings.py "
            f"--model large --corpus {corpus}"
        )
    import joblib

    return joblib.load(path)


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer

        _embed_model = SentenceTransformer(SEARCH_EMBED_MODEL_ID, device="cpu")
    return _embed_model


@lru_cache(maxsize=64)
def encode_query(term: str) -> np.ndarray:
    """L2-normalized query embedding with e5 `query:` prefix, shape (dim,)."""
    mdl = _get_embed_model()
    text = f"{SEARCH_EMBED_QUERY_PREFIX}{term}"
    v = mdl.encode([text], normalize_embeddings=True)
    return np.asarray(v, dtype=np.float32).reshape(-1)


def search_semantic(
    term: str,
    corpus: str,
    *,
    offset: int = 0,
    limit: int = 20,
    books: frozenset[str] | None = None,
) -> CorpusSearchResult:
    parse_term_tokens(term, max_tokens=None)
    idx = load_embed_nt_index(corpus)
    embs: np.ndarray = np.asarray(idx["embeddings"], dtype=np.float32)
    verses = idx["verses"]

    q = encode_query(term.strip())
    scores = embs @ q  # (N,)

    if books is not None:
        book_col = verses["book"].astype(str).str.strip()
        mask = book_col.isin(books).to_numpy()
        scores = np.where(mask, scores, -np.inf)

    valid = np.isfinite(scores)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return page_hits(
            [],
            corpus=corpus,
            offset=offset,
            limit=limit,
            ranking="score",
        )

    pool = min(SEARCH_SEMANTIC_POOL, n_valid)
    cand = np.flatnonzero(valid)
    cand_scores = scores[cand]
    if len(cand) > pool:
        part = np.argpartition(-cand_scores, pool - 1)[:pool]
        cand = cand[part]
        cand_scores = scores[cand]
    order = np.argsort(-cand_scores)
    top_idx = cand[order]

    hits: list[dict] = []
    for i in top_idx:
        row = verses.iloc[int(i)]
        hits.append(
            {
                "book": str(row["book"]).strip(),
                "chapter": int(row["chapter"]),
                "verse": int(row["verse"]),
                "text": normalize_surface(str(row["text"])),
                "score": round(float(scores[int(i)]), 4),
            }
        )

    return page_hits(hits, corpus=corpus, offset=offset, limit=limit, ranking="score")
