"""Request/response models for the API."""
from typing import Literal

from pydantic import BaseModel, Field
from enum import Enum


class ConfidenceType(str, Enum):
    LEXICAL = "lexical"   # phrase/substring match
    SEMANTIC = "semantic"  # embedding similarity (Embedić-large)


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


# --- Concept search (pojmovna pretraga) ---

SearchMode = Literal["exact", "lemma", "semantic"]
SearchRanking = Literal["biblical_order", "score"]


class SearchRequest(BaseModel):
    term: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Concept / short phrase, Cyrillic (max ~one verse of text).",
    )
    mode: SearchMode = Field(
        default="semantic",
        description="exact = surface form; lemma = all forms of the lemma; semantic = meaning (concept map expands the query).",
    )
    corpora: list[CorpusId] = Field(
        default=["dk"],
        min_length=1,
        description="Corpora to search separately (no merge).",
    )
    books: list[str] | None = Field(
        default=None,
        description=(
            "Optional NT book filter (Pouke names). "
            "Omit or all 27 books = search whole NZ; empty list is rejected."
        ),
    )
    offset: int = Field(default=0, ge=0, description="0-based offset into the ranked hit list.")
    limit: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Page size (UI default 20).",
    )


class SearchHit(BaseModel):
    book: str
    chapter: int
    verse: int
    text: str
    corpus: str
    score: float | None = Field(
        default=None,
        description="Present for semantic mode; null for exact/lemma (biblical order only).",
    )


class CorpusSearchResult(BaseModel):
    corpus: str
    total: int = Field(description="Total matching verses in this corpus.")
    offset: int
    limit: int
    returned: int = Field(description="len(hits) on this page.")
    ranking: SearchRanking = Field(
        description="biblical_order for exact/lemma; score for semantic.",
    )
    hits: list[SearchHit] = Field(default_factory=list)


class SearchResponse(BaseModel):
    term: str
    mode: SearchMode
    results_by_corpus: dict[str, CorpusSearchResult] = Field(default_factory=dict)
    message: str = ""


# --- AI ask-agent (4. način — pitanje, ne pojam) ---


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Pitanje o NZ-u, ćirilica (npr. gde se govori o oproštenju u Marku).",
    )


class AskCitation(BaseModel):
    book: str
    chapter: int
    verse: int
    text: str
    corpus: str


class AskStep(BaseModel):
    tool: str
    input: dict = Field(default_factory=dict)
    summary: str = ""


class AskResponse(BaseModel):
    question: str
    answer: str = ""
    citations: list[AskCitation] = Field(default_factory=list)
    steps: list[AskStep] = Field(default_factory=list)
    message: str = ""
