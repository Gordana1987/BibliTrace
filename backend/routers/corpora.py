"""Corpus metadata for the UI."""

from __future__ import annotations

from fastapi import APIRouter

from config import ACTIVE_CORPORA, CORPUS_LABELS, INACTIVE_CORPORA

router = APIRouter(prefix="/api", tags=["corpora"])


@router.get("/corpora")
async def get_corpora() -> dict:
    """Active corpus ids, labels, and inactive list for the frontend."""
    return {
        "active": ACTIVE_CORPORA,
        "labels": CORPUS_LABELS,
        "inactive": INACTIVE_CORPORA,
    }
