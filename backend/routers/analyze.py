from fastapi import APIRouter

from models.schemas import AnalyzeRequest, AnalyzeResponse
from services.detection import detect

router = APIRouter(prefix="/api", tags=["analyze"])


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    """Analyze Serbian text for Biblical intertextuality. Uses TF-IDF over CLASSLA-lemmatized text."""
    return detect(request)
