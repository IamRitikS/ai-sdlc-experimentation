"""Designer agent — parses requirements, seeks clarifications, emits task_graph."""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt

from config import CONFIDENCE_THRESHOLD, get_llm
from state import Phase, SDLCState, Task

_SYSTEM = """You are a senior software architect acting as a Designer.

You will be given a requirements document and a running log of previous Q&A clarifications.

Each turn you must respond with EXACTLY ONE of the two JSON formats below.
No other text. No markdown. No bullet lists. Pure JSON only.

FORMAT A — when you still need one critical piece of information:
{"action": "clarify", "question": "<ONE single sentence question>", "confidence": <float 0-1>}

The "question" value must be a single sentence. Not a list. Not bullet points. One sentence.

FORMAT B — when you have enough context to design the full system:
{"action": "plan", "task_graph": [...], "confidence": <float 0-1>}

task_graph items:
  id (e.g. "T01"), title, description, agent (backend|frontend|database),
  depends_on (list of ids), priority (int, 1=highest), status ("pending")

Critical rules:
- ONE question per turn. If you want to ask multiple things, pick the most important one only.
- Do not repeat a question already answered in the Q&A log.
"""



def designer_node(state: SDLCState) -> dict:
    llm = get_llm("designer")
    messages = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=f"## Requirements Document\n\n{state['requirements_doc']}"),
    ]

    for note in state.get("clarifications", []):
        messages.append(HumanMessage(content=f"## Human Clarification\n\n{note}"))

    response = llm.invoke(messages)
    raw = response.content.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"action": "clarify", "questions": [raw], "confidence": 0.5}

    confidence = parsed.get("confidence", 0.5)

    if parsed.get("action") == "clarify" or confidence < CONFIDENCE_THRESHOLD:
        question = parsed.get("question") or parsed.get("questions", ["Please clarify the requirements."])
        if isinstance(question, list):
            prompt = "\n".join(f"- {q}" for q in question)
        else:
            prompt = f"- {question}"
        human_answer = interrupt(
            {
                "node": "designer",
                "phase": Phase.DESIGN,
                "message": f"Designer needs clarification:\n\n{prompt}",
            }
        )
        return {
            "clarifications": [human_answer],
            "confidence": {"designer": confidence},
            "phase": Phase.DESIGN,
        }

    task_graph: list[Task] = parsed.get("task_graph", [])
    return {
        "task_graph": task_graph,
        "confidence": {"designer": confidence},
        "phase": Phase.PLANNING,
    }
