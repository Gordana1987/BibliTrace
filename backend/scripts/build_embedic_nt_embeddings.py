"""
Build NZ Embedić indexes (e5-style query:/passage: prefixes).

Documents (verses) are encoded with "passage: " + raw Cyrillic text from bible.csv.
Queries are NOT encoded here — see run_semantic_embedic_baseline.py (query: prefix).

Run from backend/:
  python scripts/build_embedic_nt_embeddings.py --model base --all
  python scripts/build_embedic_nt_embeddings.py --model large --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
BATCH_SIZE = 32

sys.path.insert(0, str(BASE_DIR))
from config import ACTIVE_CORPORA, DK_NT_BOOKS  # noqa: E402

MODEL_IDS = {
    "base": "djovak/embedic-base",
    "large": "djovak/embedic-large",
}


def index_name(size: str) -> str:
    return f"embedic_{size}_nt_embeddings.joblib"


def load_nt_verses(corpus: str) -> pd.DataFrame:
    """Raw Cyrillic surface text from bible.csv, NZ only — no Latin strip."""
    path = DATA_DIR / corpus / "bible.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    before = len(df)
    df = df[df["book"].astype(str).str.strip().isin(DK_NT_BOOKS)].copy()
    print(f"{corpus}: NT filter {before} → {len(df)}")
    return df.reset_index(drop=True)


def build(corpus: str, size: str) -> Path:
    from sentence_transformers import SentenceTransformer

    model_id = MODEL_IDS[size]
    out = DATA_DIR / corpus / index_name(size)
    print(f"Loading {model_id}…")
    try:
        model = SentenceTransformer(model_id, device="cpu")
    except Exception as exc:
        msg = str(exc)
        if "gated" in msg.lower() or "401" in msg or "restricted" in msg.lower():
            raise SystemExit(
                f"\nCannot download {model_id} (gated Hugging Face repo).\n"
                "1) Open https://huggingface.co/djovak/embedic-base (and …/embedic-large)\n"
                "   and accept access while logged in.\n"
                "2) Create a token at https://huggingface.co/settings/tokens\n"
                "3) Export it, then re-run:\n"
                "   export HF_TOKEN=hf_...\n"
                "   python scripts/build_embedic_nt_embeddings.py --model both --all\n"
            ) from exc
        raise

    df = load_nt_verses(corpus)
    verses = df[["book", "chapter", "verse", "text"]].copy()
    # Critical: passage: prefix; keep full Cyrillic orthography.
    texts = ["passage: " + str(t) for t in df["text"].fillna("").astype(str).tolist()]

    print(f"Embedding {len(texts)} passages…")
    embs = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embs = np.asarray(embs, dtype=np.float32)
    joblib.dump(
        {
            "embeddings": embs,
            "verses": verses,
            "model_id": model_id,
            "doc_prefix": "passage: ",
        },
        out,
    )
    print(f"Saved {out} shape={embs.shape}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["base", "large", "both"], default="base")
    parser.add_argument("--corpus", default="dk")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    corpora = list(ACTIVE_CORPORA) if args.all else [args.corpus]
    sizes = ["base", "large"] if args.model == "both" else [args.model]
    for size in sizes:
        for corpus in corpora:
            build(corpus, size)


if __name__ == "__main__":
    main()
