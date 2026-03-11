from fastapi import APIRouter

from models.schemas import AnalyzeRequest, AnalyzeResponse
from services.detection import detect

router = APIRouter(prefix="/api", tags=["analyze"])


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    """Analyze Serbian text. BM25 + Qwen3 (default). Set compare_with_labse=True for LaBSE comparison."""
    return detect(request, compare_with_labse=request.compare_with_labse)
