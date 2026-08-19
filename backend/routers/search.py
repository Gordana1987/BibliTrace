from fastapi import APIRouter

from models.schemas import SearchRequest, SearchResponse
from services.search import search

router = APIRouter(prefix="/api", tags=["search"])


@router.post("/search", response_model=SearchResponse)
def concept_search(request: SearchRequest):
    """
    Concept / term search over NZ corpora.

    Modes: exact (surface), lemma (lemma index), semantic (meaning; concept-map query expansion).
    Each corpus is searched separately (no merge). Paginate with offset/limit.
    """
    return search(request)
