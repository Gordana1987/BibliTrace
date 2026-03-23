"""Request/response models for the API."""
from pydantic import BaseModel, Field
from enum import Enum


class ConfidenceType(str, Enum):
    LEXICAL = "lexical"   # direct TF-IDF match
    SEMANTIC = "semantic" # LaBSE embedding similarity


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
    corpus: str = Field(default="dk", description="Source corpus: 'dk' or 'bakotic'")


class OTNTSummary(BaseModel):
    old_testament: int = 0
    new_testament: int = 0


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Serbian literary text to analyze")
    compare_with_labse: bool = Field(default=False, description="Also run LaBSE for side-by-side comparison")
    version: str = Field(default="dk", description="Corpus to search: 'dk', 'bakotic', or 'both'")


class AnalyzeResponse(BaseModel):
    matches: list[MatchFragment] = []
    summary: OTNTSummary = OTNTSummary()
    message: str = ""
    labse_matches: list[MatchFragment] | None = Field(default=None, description="LaBSE results when compare_with_labse=True")
