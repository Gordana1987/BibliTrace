"""
Build a BM25 index over the lemmatized Bible corpus.

Loads bible_lemmatized.csv, tokenizes the 'lemmatized' column (word tokens),
builds BM25Okapi index, saves it with verse metadata for fast lexical retrieval.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import joblib
import pandas as pd
from rank_bm25 import BM25Okapi


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
BIBLE_DIR = DATA_DIR / "bible"

INPUT_CSV = BIBLE_DIR / "bible_lemmatized.csv"
INDEX_PATH = BIBLE_DIR / "bm25_index.joblib"

# Word tokens (Cyrillic + Latin); must match detection.py
TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Return list of word tokens. Used for both index build and query."""
    if not isinstance(text, str) or not text.strip():
        return []
    return TOKEN_PATTERN.findall(text)


def build_index(k1: float = 1.5, b: float = 0.75) -> Path:
    """
    Load bible_lemmatized.csv, tokenize, build BM25Okapi, save to bm25_index.joblib.

    Saved dict:
      - bm25: BM25Okapi instance
      - verses: DataFrame with columns book, chapter, verse, text
    """
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Lemmatized Bible not found: {INPUT_CSV}. Run lemmatize_bible.py first.")

    print(f"Loading {INPUT_CSV} ...")
    df = pd.read_csv(INPUT_CSV)

    if "lemmatized" not in df.columns:
        raise ValueError("Expected a 'lemmatized' column. Run lemmatize_bible.py first.")

    texts = df["lemmatized"].fillna("").astype(str)
    print("Tokenizing corpus ...")
    corpus = [tokenize(t) for t in texts]

    print("Building BM25 index ...")
    bm25 = BM25Okapi(corpus, k1=k1, b=b)

    verses = df[["book", "chapter", "verse", "text"]].copy()

    index_data = {
        "bm25": bm25,
        "verses": verses,
    }

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing index to {INDEX_PATH} ...")
    joblib.dump(index_data, INDEX_PATH)
    print("Done.")
    return INDEX_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Build BM25 index over lemmatized Bible.")
    parser.add_argument("--k1", type=float, default=1.5, help="BM25 k1 (term frequency saturation).")
    parser.add_argument("--b", type=float, default=0.75, help="BM25 b (length normalization).")
    args = parser.parse_args()
    build_index(k1=args.k1, b=args.b)


if __name__ == "__main__":
    main()
