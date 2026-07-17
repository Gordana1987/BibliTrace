"""Request/response models for the API."""
from typing import Literal

from pydantic import BaseModel, Field
from enum import Enum


class ConfidenceType(str, Enum):
    LEXICAL = "lexical"   # phrase/substring match
    SEMANTIC = "semantic" # embedding similarity (Qwen / LaBSE)


class BibleRef(BaseModel):
    book: str
    chapter: int
    verse: int
    text: str = ""


class MatchFragment(BaseModel):
    """One detected link: a span in the input text mapped to a Bible reference."""
    start: int
    end: int
    input_snippet: str
    bible_ref: BibleRef
    confidence_type: ConfidenceType
    score: float = Field(ge=0, le=1)
    corpus: str = Field(
        default="dk",
        description="Source corpus id (dk, spc, …)",
    )


class OTNTSummary(BaseModel):
    old_testament: int = 0
    new_testament: int = 0


CorpusId = Literal["dk", "dk_ekav", "bakotic", "spc"]


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Serbian literary text to analyze")
    compare_with_labse: bool = Field(default=False, description="Also run LaBSE for side-by-side comparison")
    corpora: list[CorpusId] = Field(
        default=["dk"],
        min_length=1,
        description="Corpora to search separately (e.g. dk, spc). Results are not merged.",
    )
    version: Literal["dk", "dk_ekav", "bakotic", "spc", "both", "all"] | None = Field(
        default=None,
        description="Deprecated; use corpora. both/all → all ACTIVE_CORPORA.",
    )


class AnalyzeResponse(BaseModel):
    matches: list[MatchFragment] = Field(
        default_factory=list,
        description="Single-corpus convenience: same as matches_by_corpus[that corpus]. Empty when multiple corpora.",
    )
    matches_by_corpus: dict[str, list[MatchFragment]] = Field(
        default_factory=dict,
        description="Top matches per corpus (no cross-corpus merge).",
    )
    summary: OTNTSummary = OTNTSummary()
    message: str = ""
    labse_matches: list[MatchFragment] | None = Field(
        default=None,
        description="Single-corpus LaBSE results when compare_with_labse=True",
    )
    labse_matches_by_corpus: dict[str, list[MatchFragment]] | None = Field(
        default=None,
        description="LaBSE results per corpus when compare_with_labse=True",
    )
