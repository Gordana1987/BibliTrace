"""
Build embedding indexes for Qwen3-Embedding-0.6B and LaBSE.

  qwen / labse     → data/<corpus>/qwen_embeddings.joblib / labse_embeddings.joblib
                     (full CSV; legacy /api/analyze hybrid)
  qwen --nt-only   → archive/qwen-nt-embeddings/<corpus>/qwen_nt_embeddings.joblib
                     (historical NZ concept-search index; live semantic is Embedić)

Live concept semantic: scripts/build_embedic_nt_embeddings.py

Run from backend/:
  python scripts/build_embeddings.py qwen --corpus dk
  python scripts/build_embeddings.py qwen --all
  python scripts/build_embeddings.py qwen --all --nt-only   # archive rebuild only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data"
ARCHIVE_QWEN_NT = REPO_ROOT / "archive" / "qwen-nt-embeddings"
QWEN_NT_INDEX_NAME = "qwen_nt_embeddings.joblib"
BATCH_SIZE = 8

sys.path.insert(0, str(BASE_DIR))
from config import ACTIVE_CORPORA, DK_NT_BOOKS  # noqa: E402


def _load_verse_frame(corpus: str, *, nt_only: bool) -> pd.DataFrame:
    corpus_dir = DATA_DIR / corpus
    input_csv = corpus_dir / "bible_lemmatized.csv"
    if not input_csv.exists():
        raise FileNotFoundError(f"Not found: {input_csv}. Run lemmatize_bible.py first.")
    df = pd.read_csv(input_csv)
    if "text" not in df.columns:
        raise ValueError(f"{input_csv} missing 'text' column")
    before = len(df)
    if nt_only:
        df = df[df["book"].astype(str).str.strip().isin(DK_NT_BOOKS)].copy()
        print(f"NT filter: {before} → {len(df)} verses")
    return df.reset_index(drop=True)


def build_qwen_index(corpus: str = "dk", *, nt_only: bool = False) -> Path:
    """Embed verses with Qwen3-Embedding-0.6B."""
    from sentence_transformers import SentenceTransformer

    if nt_only:
        out_dir = ARCHIVE_QWEN_NT / corpus
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / QWEN_NT_INDEX_NAME
    else:
        path = DATA_DIR / corpus / "qwen_embeddings.joblib"

    print("Loading Qwen3-Embedding-0.6B (first run downloads ~1.2GB)...")
    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cpu")

    df = _load_verse_frame(corpus, nt_only=nt_only)
    verses = df[["book", "chapter", "verse", "text"]].copy()
    texts = df["text"].fillna("").astype(str).tolist()

    print(f"Embedding {len(texts)} verses (documents, no prompt)...")
    embs = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embs = np.asarray(embs, dtype=np.float32)

    joblib.dump({"embeddings": embs, "verses": verses}, path)
    print(f"Saved to {path}  shape={embs.shape}")
    return path


def build_labse_index(corpus: str = "dk", *, nt_only: bool = False) -> Path:
    """Embed verses with LaBSE (legacy compare path; no concept-search use)."""
    from sentence_transformers import SentenceTransformer

    if nt_only:
        raise ValueError(
            "LaBSE build does not use --nt-only "
            "(live concept semantic is Embedić; see build_embedic_nt_embeddings.py)."
        )

    path = DATA_DIR / corpus / "labse_embeddings.joblib"

    print("Loading LaBSE (first run downloads ~470MB)...")
    model = SentenceTransformer("sentence-transformers/LaBSE", device="cpu")

    df = _load_verse_frame(corpus, nt_only=False)
    verses = df[["book", "chapter", "verse", "text"]].copy()
    texts = df["text"].fillna("").astype(str).tolist()

    print(f"Embedding {len(texts)} verses...")
    embs = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embs = np.asarray(embs, dtype=np.float32)

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
    parser.add_argument(
        "--corpus",
        default="dk",
        help=f"Corpus folder under data/ (default dk). Active: {ACTIVE_CORPORA}.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=f"Build for all ACTIVE_CORPORA ({ACTIVE_CORPORA}).",
    )
    parser.add_argument(
        "--nt-only",
        action="store_true",
        help=(
            f"Restrict to NZ books; write archive/.../{QWEN_NT_INDEX_NAME} "
            "(historical; live semantic uses Embedić)."
        ),
    )
    args = parser.parse_args()
    corpora = list(ACTIVE_CORPORA) if args.all else [args.corpus]
    for corpus in corpora:
        if args.model in ("qwen", "both"):
            build_qwen_index(corpus=corpus, nt_only=args.nt_only)
        if args.model in ("labse", "both"):
            build_labse_index(corpus=corpus, nt_only=args.nt_only)


if __name__ == "__main__":
    main()
