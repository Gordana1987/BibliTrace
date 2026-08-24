"""
Run agent eval set and record full baseline.

Run from backend/:
  python scripts/run_agent_eval.py
  python scripts/run_agent_eval.py --tag v2

Writes:
  data/concept/agent_eval_<tag>_results.json
  data/concept/agent_eval_<tag>_review.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402

from config import AGENT_MODEL, AGENT_SEARCH_LIMIT  # noqa: E402
from services.agent.graph import _agent  # noqa: E402

EVAL_SET = BASE_DIR / "data" / "concept" / "agent_eval_v1.json"

HAIKU_INPUT_COST = 1.0 / 1_000_000
HAIKU_OUTPUT_COST = 5.0 / 1_000_000

REF_PATTERN = re.compile(
    r"(?:Мк|Марко|Мат|Матеј|Лк|Лука|Јн|Јован|Дела|Рим|1\.\s*Кор|2\.\s*Кор|Гал|Еф|Фил|Кол|"
    r"1\.\s*Сол|2\.\s*Сол|1\.\s*Тим|2\.\s*Тим|Тит|Филим|Јевр|Јаков|1\.\s*Пет|2\.\s*Пет|"
    r"1\.\s*Јов|2\.\s*Јов|3\.\s*Јов|Јуд|Откр|Откривење|"
    r"Римљанима|1\.\s*Коринћанима|2\.\s*Коринћанима|Галатима|Ефешанима|Филипљанима|"
    r"Колошанима|1\.\s*Солуњанима|2\.\s*Солуњанима|1\.\s*Тимотеју|2\.\s*Тимотеју|"
    r"Филимону|Јеврејима|Јаковљева|1\.\s*Петрова|2\.\s*Петрова|"
    r"1\.\s*Јованова|2\.\s*Јованова|3\.\s*Јованова|Јудина|Дела апостолска)"
    r"\s+(\d{1,3})\s*[:\.,]\s*(\d{1,3})",
)

BOOK_NORM = {
    "Мк": "Марко", "Мат": "Матеј", "Лк": "Лука", "Јн": "Јован",
    "Дела": "Дела апостолска", "Рим": "Римљанима",
    "Еф": "Ефешанима", "Фил": "Филипљанима", "Кол": "Колошанима",
    "Јевр": "Јеврејима", "Јаков": "Јаковљева",
    "Откр": "Откривење",
}


def normalize_book(raw: str) -> str:
    s = raw.strip()
    return BOOK_NORM.get(s, s)


def extract_refs_from_answer(answer: str) -> set[tuple[str, int, int]]:
    refs: set[tuple[str, int, int]] = set()
    for m in REF_PATTERN.finditer(answer):
        full = m.group(0)
        ch, vs = int(m.group(1)), int(m.group(2))
        book_part = full[: m.start(1) - m.start(0)].strip().rstrip(":., ")
        refs.add((normalize_book(book_part), ch, vs))
    return refs


def run_one(agent, question: str) -> dict:
    t0 = time.time()
    result = agent.invoke({"messages": [HumanMessage(content=question)]})
    elapsed_ms = int((time.time() - t0) * 1000)
    messages = result["messages"]

    answer = ""
    total_input = 0
    total_output = 0
    for msg in messages:
        if isinstance(msg, AIMessage):
            um = getattr(msg, "usage_metadata", None)
            if um:
                total_input += um.get("input_tokens", 0)
                total_output += um.get("output_tokens", 0)
            if msg.content and not msg.tool_calls:
                answer = str(msg.content).strip()

    tools_used: list[str] = []
    returned_refs: set[tuple[str, int, int]] = set()
    steps: list[dict] = []
    tool_inputs: dict[str, dict] = {}

    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for call in msg.tool_calls:
                tools_used.append(call["name"])
                tool_inputs[call["id"]] = {
                    "tool": call["name"],
                    "input": call.get("args") or {},
                }
        if isinstance(msg, ToolMessage):
            meta = tool_inputs.get(msg.tool_call_id, {"tool": "unknown", "input": {}})
            try:
                payload = json.loads(str(msg.content))
                corpus = payload.get("corpus", "dk")
                for hit in payload.get("hits") or []:
                    returned_refs.add((str(hit["book"]).strip(), int(hit["chapter"]), int(hit["verse"])))
            except (json.JSONDecodeError, TypeError):
                pass
            steps.append(meta)

    answer_refs = extract_refs_from_answer(answer)
    fabricated = answer_refs - returned_refs if answer_refs else set()

    cost_usd = total_input * HAIKU_INPUT_COST + total_output * HAIKU_OUTPUT_COST

    return {
        "answer": answer,
        "tools_used": tools_used,
        "steps": steps,
        "n_citations_returned": len(returned_refs),
        "n_refs_in_answer": len(answer_refs),
        "n_fabricated_refs": len(fabricated),
        "fabricated_refs": [
            {"book": b, "chapter": c, "verse": v} for b, c, v in sorted(fabricated)
        ],
        "returned_refs": [
            {"book": b, "chapter": c, "verse": v} for b, c, v in sorted(returned_refs)
        ],
        "input_tokens": total_input,
        "output_tokens": total_output,
        "cost_usd": round(cost_usd, 6),
        "elapsed_ms": elapsed_ms,
    }


def check_tool_accuracy(expected_tools: list[str], actual_tools: list[str]) -> bool:
    if not expected_tools:
        return len(actual_tools) == 0
    return bool(set(expected_tools) & set(actual_tools))


def check_citation_recall(expected_subset: list[dict], returned_refs: set[tuple]) -> dict:
    if not expected_subset:
        return {"expected": 0, "found": 0, "recall": 1.0}
    found = 0
    for exp in expected_subset:
        key = (exp["book"], int(exp["chapter"]), int(exp["verse"]))
        if key in returned_refs:
            found += 1
    return {
        "expected": len(expected_subset),
        "found": found,
        "recall": round(found / len(expected_subset), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run agent eval set.")
    parser.add_argument(
        "--tag",
        default="v1",
        help="Output file tag (default: v1 → agent_eval_v1_results.json).",
    )
    parser.add_argument(
        "--note",
        default="",
        help="Optional note stored in results JSON (e.g. stronger grounding prompt).",
    )
    args = parser.parse_args()
    tag = args.tag.strip() or "v1"
    out_json = BASE_DIR / "data" / "concept" / f"agent_eval_{tag}_results.json"
    out_md = BASE_DIR / "data" / "concept" / f"agent_eval_{tag}_review.md"

    eval_data = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    _agent.cache_clear()
    agent = _agent()
    all_results: list[dict] = []

    md_lines = [
        f"# Agent Eval {tag} — Review\n",
        f"Model: {AGENT_MODEL}  \n",
        f"Search limit: {AGENT_SEARCH_LIMIT}  \n",
        f"Map: **disabled** (goli Embedić)  \n",
        f"Run: {datetime.now(timezone.utc).isoformat()}\n\n",
        "---\n\n",
    ]
    if args.note:
        md_lines.insert(5, f"Note: {args.note}\n\n")

    for q in eval_data["questions"]:
        qid = q["id"]
        question = q["question"]
        print(f"  [{qid}] {question[:60]}...", flush=True)
        raw = run_one(agent, question)

        returned_set = {(r["book"], r["chapter"], r["verse"]) for r in raw["returned_refs"]}
        tool_ok = check_tool_accuracy(q.get("expected_tools", []), raw["tools_used"])
        citation = check_citation_recall(q.get("expected_citations_subset", []), returned_set)

        entry = {
            "id": qid,
            "category": q["category"],
            "question": question,
            **raw,
            "eval": {
                "tool_accuracy": tool_ok,
                "citation_recall": citation,
                "n_fabricated": raw["n_fabricated_refs"],
                "forbidden_notes": q.get("forbidden", []),
                "manual_hallucination_check": None,
            },
        }
        all_results.append(entry)

        md_lines.append(f"## [{qid}] {q['category']}\n\n")
        md_lines.append(f"**Pitanje:** {question}\n\n")
        md_lines.append(f"**Alati:** {', '.join(raw['tools_used']) or '(nijedan)'}\n\n")
        md_lines.append(f"**Odgovor:**\n\n{raw['answer']}\n\n")
        md_lines.append(f"**Metrike:** tool_ok={tool_ok}, "
                        f"citation_recall={citation['recall']} ({citation['found']}/{citation['expected']}), "
                        f"fabricated={raw['n_fabricated_refs']}, "
                        f"citations_returned={raw['n_citations_returned']}, "
                        f"refs_in_answer={raw['n_refs_in_answer']}, "
                        f"cost=${raw['cost_usd']:.4f}, "
                        f"latency={raw['elapsed_ms']}ms, "
                        f"tokens={raw['input_tokens']}+{raw['output_tokens']}\n\n")
        if raw["fabricated_refs"]:
            md_lines.append("**⚠ Fabricated refs:** "
                            + ", ".join(f"{r['book']} {r['chapter']}:{r['verse']}" for r in raw["fabricated_refs"])
                            + "\n\n")
        if q.get("forbidden"):
            md_lines.append("**Forbidden (check manually):**\n")
            for f in q["forbidden"]:
                md_lines.append(f"- [ ] {f}\n")
            md_lines.append("\n")
        md_lines.append("---\n\n")

    n = len(all_results)
    summary = {
        "n_questions": n,
        "tool_accuracy": round(sum(1 for r in all_results if r["eval"]["tool_accuracy"]) / n, 2),
        "mean_citation_recall": round(
            sum(r["eval"]["citation_recall"]["recall"] for r in all_results) / n, 2
        ),
        "total_fabricated": sum(r["n_fabricated_refs"] for r in all_results),
        "mean_citations_returned": round(
            sum(r["n_citations_returned"] for r in all_results) / n, 1
        ),
        "mean_refs_in_answer": round(
            sum(r["n_refs_in_answer"] for r in all_results) / n, 1
        ),
        "total_cost_usd": round(sum(r["cost_usd"] for r in all_results), 4),
        "mean_latency_ms": round(sum(r["elapsed_ms"] for r in all_results) / n),
        "mean_input_tokens": round(sum(r["input_tokens"] for r in all_results) / n),
        "mean_output_tokens": round(sum(r["output_tokens"] for r in all_results) / n),
    }

    output = {
        "eval_tag": tag,
        "eval_set": str(EVAL_SET.relative_to(BASE_DIR)),
        "note": args.note or None,
        "model": AGENT_MODEL,
        "search_limit": AGENT_SEARCH_LIMIT,
        "map_active": False,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "results": all_results,
    }

    out_json.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out_json}")

    md_lines.insert(6, f"## Summary\n\n"
                       f"| Metrika | Vrednost |\n"
                       f"|---|---|\n"
                       f"| Tool accuracy | {summary['tool_accuracy']:.0%} |\n"
                       f"| Mean citation recall | {summary['mean_citation_recall']:.0%} |\n"
                       f"| Total fabricated refs | {summary['total_fabricated']} |\n"
                       f"| Mean citations returned | {summary['mean_citations_returned']} |\n"
                       f"| Mean refs in answer | {summary['mean_refs_in_answer']} |\n"
                       f"| Total cost | ${summary['total_cost_usd']:.4f} |\n"
                       f"| Mean latency | {summary['mean_latency_ms']}ms |\n"
                       f"| Mean tokens (in+out) | {summary['mean_input_tokens']}+{summary['mean_output_tokens']} |\n\n"
                       f"---\n\n")

    out_md.write_text("".join(md_lines), encoding="utf-8")
    print(f"Wrote {out_md}")
    print(f"\nSummary: tool_accuracy={summary['tool_accuracy']:.0%} "
          f"citation_recall={summary['mean_citation_recall']:.0%} "
          f"fabricated={summary['total_fabricated']} "
          f"cost=${summary['total_cost_usd']:.4f}")


if __name__ == "__main__":
    main()
