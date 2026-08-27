"""Ask-agent over NZ search tools (LangChain create_agent / LangGraph runtime)."""

from __future__ import annotations

import json
import os
from functools import lru_cache

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_anthropic import ChatAnthropic

from config import AGENT_MODEL
from models.schemas import AskCitation, AskResponse, AskStep
from services.agent.prompts import SYSTEM_PROMPT
from services.agent.refs import filter_answer_to_allowed_refs, format_removed_refs_message
from services.agent.tools import SEARCH_TOOLS


class AgentNotConfiguredError(RuntimeError):
    pass


def _anthropic_api_key() -> str:
    return os.getenv("ANTHROPIC_API_KEY", "").strip()


def agent_configured() -> bool:
    key = _anthropic_api_key()
    return bool(key) and key.startswith("sk-ant")


@lru_cache(maxsize=1)
def _agent():
    if not agent_configured():
        raise AgentNotConfiguredError(
            "ANTHROPIC_API_KEY није постављен. Копирај backend/.env.example у backend/.env."
        )
    llm = ChatAnthropic(model=AGENT_MODEL, temperature=0, api_key=_anthropic_api_key())
    return create_agent(llm, SEARCH_TOOLS, system_prompt=SYSTEM_PROMPT)


def _extract_citations(messages) -> list[AskCitation]:
    seen: set[tuple] = set()
    out: list[AskCitation] = []
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        try:
            payload = json.loads(str(msg.content))
        except (json.JSONDecodeError, TypeError):
            continue
        corpus = payload.get("corpus", "dk")
        for hit in payload.get("hits") or []:
            key = (corpus, hit["book"], hit["chapter"], hit["verse"])
            if key in seen:
                continue
            seen.add(key)
            out.append(
                AskCitation(
                    book=hit["book"],
                    chapter=int(hit["chapter"]),
                    verse=int(hit["verse"]),
                    text=hit.get("text") or "",
                    corpus=corpus,
                )
            )
    return out


def _extract_steps(messages) -> list[AskStep]:
    tool_inputs: dict[str, dict] = {}
    steps: list[AskStep] = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for call in msg.tool_calls:
                tool_inputs[call["id"]] = {
                    "tool": call["name"],
                    "input": call.get("args") or {},
                }
        if isinstance(msg, ToolMessage):
            meta = tool_inputs.get(msg.tool_call_id, {"tool": "unknown", "input": {}})
            summary = str(msg.content)
            if len(summary) > 400:
                summary = summary[:400] + "…"
            steps.append(
                AskStep(
                    tool=meta["tool"],
                    input=meta["input"],
                    summary=summary,
                )
            )
    return steps


def ask_agent(question: str) -> AskResponse:
    agent = _agent()
    result = agent.invoke({"messages": [HumanMessage(content=question.strip())]})
    messages = result["messages"]
    answer = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            answer = str(msg.content).strip()
            break
    citations = _extract_citations(messages)
    allowed = {(c.book, c.chapter, c.verse) for c in citations}
    answer, removed = filter_answer_to_allowed_refs(answer, allowed)
    return AskResponse(
        question=question.strip(),
        answer=answer,
        citations=citations,
        steps=_extract_steps(messages),
        message=format_removed_refs_message(removed),
    )
