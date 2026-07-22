"""
Build a lemma index for concept-search (lemma mode).

Loads bible_lemmatized.csv, tokenizes the lemmatized column, optionally restricts
to NT books, builds an inverted index lemma → verse positions, saves joblib.

Output: data/<corpus>/lemma_index.joblib

  {
    "verses": list[dict]  # book, chapter, verse, text (surface)
    "lemma_tokens": list[list[str]]  # parallel, casefolded word tokens
    "inverted": dict[str, list[int]]  # lemma -> verse indices
  }

Note: CLASSLA sr expects Latin — lemmatize_bible.py transliterates Cyrillic→Latin
before tagging so lemmas are real bases (e.g. praštajte/prašta → praštati).
Rebuild this index after re-running lemmatize_bible.py.

Run from backend/:
  python scripts/build_lemma_index.py
  python scripts/build_lemma_index.py --corpus spc
  python scripts/build_lemma_index.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from config import ACTIVE_CORPORA, DATA_DIR, DK_NT_BOOKS, RESTRICT_TO_NEW_TESTAMENT
from services.search.common import tokenize_surface


def build_lemma_index(corpus: str = "dk") -> Path:
    corpus_dir = DATA_DIR / corpus
    input_csv = corpus_dir / "bible_lemmatized.csv"
    out_path = corpus_dir / "lemma_index.joblib"

    if not input_csv.exists():
        raise FileNotFoundError(
            f"Lemmatized Bible not found: {input_csv}. Run lemmatize_bible.py first."
        )

    print(f"Loading {input_csv} ...")
    df = pd.read_csv(input_csv)
    if "lemmatized" not in df.columns:
        raise ValueError("Expected 'lemmatized' column.")

    if RESTRICT_TO_NEW_TESTAMENT:
        before = len(df)
        df = df[df["book"].astype(str).str.strip().isin(DK_NT_BOOKS)].copy()
        print(f"NT filter: {before} → {len(df)} verses")

    verses: list[dict] = []
    lemma_tokens: list[list[str]] = []
    inverted: dict[str, list[int]] = {}

    for row in df.itertuples(index=False):
        book = str(row.book).strip()
        try:
            chapter = int(row.chapter)
            verse = int(row.verse)
        except (TypeError, ValueError):
            continue
        text = str(row.text) if pd.notna(row.text) else ""
        lemmas = tokenize_surface(str(row.lemmatized) if pd.notna(row.lemmatized) else "")
        idx = len(verses)
        verses.append({"book": book, "chapter": chapter, "verse": verse, "text": text})
        lemma_tokens.append(lemmas)
        for lem in set(lemmas):
            inverted.setdefault(lem, []).append(idx)

    payload = {
        "corpus": corpus,
        "verses": verses,
        "lemma_tokens": lemma_tokens,
        "inverted": inverted,
        "nt_only": bool(RESTRICT_TO_NEW_TESTAMENT),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, out_path)
    print(
        f"Wrote {out_path}  verses={len(verses)}  "
        f"unique_lemmas={len(inverted)}"
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build lemma index for concept search")
    parser.add_argument("--corpus", default="dk", help="Corpus id under data/")
    parser.add_argument(
        "--all",
        action="store_true",
        help=f"Build for all ACTIVE_CORPORA ({ACTIVE_CORPORA})",
    )
    args = parser.parse_args()
    corpora = list(ACTIVE_CORPORA) if args.all else [args.corpus]
    for c in corpora:
        build_lemma_index(c)


if __name__ == "__main__":
    main()
