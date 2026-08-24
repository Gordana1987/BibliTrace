from fastapi import APIRouter, HTTPException

from config import AGENT_MODEL
from models.schemas import AskRequest, AskResponse
from services.agent.graph import AgentNotConfiguredError, agent_configured, ask_agent

router = APIRouter(prefix="/api", tags=["ask"])


@router.get("/ask/status")
def ask_status():
    """Whether the AI agent can run (ANTHROPIC_API_KEY set)."""
    return {"configured": agent_configured(), "model": AGENT_MODEL}


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """
    AI pretraga: prirodno pitanje → agent bira alate (exact/lemma/semantic) → odgovor sa citatima.
    Zahteva ANTHROPIC_API_KEY u backend/.env.
    """
    if not agent_configured():
        raise HTTPException(
            status_code=503,
            detail="AI agent nije podešen. Postavi ANTHROPIC_API_KEY u backend/.env.",
        )
    try:
        return ask_agent(request.question)
    except AgentNotConfiguredError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Agent greška: {exc}") from exc
