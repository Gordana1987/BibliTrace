"""
Build embedding indexes for Qwen3-Embedding-0.6B and LaBSE.

Loads bible_lemmatized.csv (uses raw 'text' column for semantics), embeds all verses,
saves to qwen_embeddings.joblib and labse_embeddings.joblib.
Verse order must match BM25 index for hybrid retrieval.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
BIBLE_DIR = DATA_DIR / "bible"
INPUT_CSV = BIBLE_DIR / "bible_lemmatized.csv"
BATCH_SIZE = 8


def build_qwen_index() -> Path:
    """Embed verses with Qwen3-Embedding-0.6B. Saves to qwen_embeddings.joblib."""
    from sentence_transformers import SentenceTransformer

    path = BIBLE_DIR / "qwen_embeddings.joblib"
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Not found: {INPUT_CSV}. Run lemmatize_bible.py first.")

    print("Loading Qwen3-Embedding-0.6B (first run downloads ~1.2GB)...")
    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

    df = pd.read_csv(INPUT_CSV)
    verses = df[["book", "chapter", "verse", "text"]].copy()
    texts = df["text"].fillna("").astype(str).tolist()

    print("Embedding verses (documents, no prompt)...")
    embs = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embs = np.array(embs, dtype=np.float32)

    joblib.dump({"embeddings": embs, "verses": verses}, path)
    print(f"Saved to {path}")
    return path


def build_labse_index() -> Path:
    """Embed verses with LaBSE. Saves to labse_embeddings.joblib."""
    from sentence_transformers import SentenceTransformer

    path = BIBLE_DIR / "labse_embeddings.joblib"
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Not found: {INPUT_CSV}. Run lemmatize_bible.py first.")

    print("Loading LaBSE (first run downloads ~470MB)...")
    model = SentenceTransformer("sentence-transformers/LaBSE")

    df = pd.read_csv(INPUT_CSV)
    verses = df[["book", "chapter", "verse", "text"]].copy()
    texts = df["text"].fillna("").astype(str).tolist()

    print("Embedding verses...")
    embs = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embs = np.array(embs, dtype=np.float32)

    joblib.dump({"embeddings": embs, "verses": verses}, path)
    print(f"Saved to {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build embedding indexes for Qwen3 or LaBSE.")
    parser.add_argument(
        "model",
        choices=["qwen", "labse", "both"],
        help="Which model(s) to build.",
    )
    args = parser.parse_args()
    if args.model in ("qwen", "both"):
        build_qwen_index()
    if args.model in ("labse", "both"):
        build_labse_index()


if __name__ == "__main__":
    main()
