from __future__ import annotations

import argparse
from pathlib import Path

import classla
import pandas as pd
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parents[1]  # .../backend
DATA_DIR = BASE_DIR / "data"

import sys

sys.path.insert(0, str(BASE_DIR))
from config import ACTIVE_CORPORA


def get_bible_paths(corpus: str = "dk") -> tuple[Path, Path]:
    """Return input and output CSV paths for the given corpus."""
    corpus_dir = DATA_DIR / corpus
    input_path = corpus_dir / "bible.csv"
    output_path = corpus_dir / "bible_lemmatized.csv"
    return input_path, output_path


def ensure_classla_model(lang: str = "sr") -> None:
    """
    Ensure the CLASSLA model for Serbian is available.

    First run will download ~500MB of data.
    """
    # classla.download is safe to call multiple times; it skips if already present.
    classla.download(lang)


def build_pipeline(lang: str = "sr", use_gpu: bool | None = None) -> classla.Pipeline:
    """
    Build a CLASSLA pipeline for lemmatization.

    We keep processors minimal to speed things up.
    """
    if use_gpu is None:
        # Let CLASSLA decide based on environment; user can override with CLI flag.
        use_gpu = False

    return classla.Pipeline(
        lang,
        processors="tokenize,pos,lemma",
        use_gpu=use_gpu,
    )


def lemmatize_text(pipeline: classla.Pipeline, text: str) -> str:
    """
    Return a space-separated lemma string for the given text.

    CLASSLA sr expects Latin script: Cyrillic input is transliterated first.
    Lemmas are kept in Latin so they match the query-side lemmatizer.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    from services.transliterate import cyrillic_to_latin

    lat = cyrillic_to_latin(text)
    doc = pipeline(lat)
    lemmas: list[str] = []
    for sent in doc.sentences:
        for word in sent.words:
            lemma = (word.lemma or word.text).strip()
            if lemma:
                lemmas.append(lemma)
    return " ".join(lemmas)


def lemmatize_bible_csv(use_gpu: bool = False, overwrite: bool = False, corpus: str = "dk") -> Path:
    """
    Load bible.csv, lemmatize verse text, and write bible_lemmatized.csv.

    - Input:  DATA_DIR / <corpus> / "bible.csv"
    - Output: DATA_DIR / <corpus> / "bible_lemmatized.csv" (or overwrite input if requested)
    """
    input_path, output_path = get_bible_paths(corpus)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    print(f"Loading Bible CSV from {input_path} ...")
    df = pd.read_csv(input_path)

    if "text" not in df.columns:
        raise ValueError("Expected a 'text' column in the Bible CSV.")

    # Download model if needed and build pipeline once.
    print("Ensuring CLASSLA Serbian model is available (this may take a while on first run)...")
    ensure_classla_model("sr")
    print("Initializing CLASSLA pipeline...")
    pipeline = build_pipeline("sr", use_gpu=use_gpu)

    print("Lemmatizing verses with progress bar (tqdm)...")
    lemmas: list[str] = []
    for verse_text in tqdm(df["text"], total=len(df)):
        lemmas.append(lemmatize_text(pipeline, verse_text))

    df["lemmatized"] = lemmas

    target_path = input_path if overwrite else output_path
    print(f"Writing lemmatized Bible to {target_path} ...")
    df.to_csv(target_path, index=False)
    print("Done.")
    return target_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lemmatize Bible verses using CLASSLA and save to CSV."
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU for CLASSLA if available.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing bible.csv instead of writing bible_lemmatized.csv.",
    )
    parser.add_argument(
        "--corpus",
        default="dk",
        help=f"Corpus folder under data/ (default dk). Use --all for {ACTIVE_CORPORA}.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=f"Lemmatize all ACTIVE_CORPORA ({ACTIVE_CORPORA}).",
    )
    args = parser.parse_args()

    corpora = list(ACTIVE_CORPORA) if args.all else [args.corpus]
    for corpus in corpora:
        if corpus not in ACTIVE_CORPORA and corpus not in ("dk_ekav", "bakotic"):
            # Allow inactive corpora on disk for maintenance; warn otherwise.
            print(f"Warning: {corpus!r} is not in ACTIVE_CORPORA; continuing anyway.")
        lemmatize_bible_csv(use_gpu=args.use_gpu, overwrite=args.overwrite, corpus=corpus)


if __name__ == "__main__":
    main()

