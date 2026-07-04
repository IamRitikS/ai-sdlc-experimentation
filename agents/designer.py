"""Designer agent — parses requirements, seeks clarifications, emits task_graph."""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt

from config import CONFIDENCE_THRESHOLD, get_llm
from state import Phase, SDLCState, Task

_SYSTEM = """You are a senior software architect acting as a Designer.
Your job:
1. Analyse the requirements document thoroughly.
2. Identify ambiguities. If any exist, list them as questions for the human.
3. Once requirements are clear, decompose the work into an ordered task_graph.

Each task must have:
  - id (string, e.g. "T01")
  - title (short label)
  - description (detailed spec the implementing agent will follow)
  - agent (one of: backend, frontend, database, integrator, devops)
  - depends_on (list of task ids)
  - priority (integer, 1 = highest)
  - status ("pending")

Respond ONLY with valid JSON matching one of:
  {"action": "clarify", "questions": [...], "confidence": <float 0-1>}
  {"action": "plan",    "task_graph": [...], "confidence": <float 0-1>}
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
        questions = parsed.get("questions", ["Please clarify the requirements."])
        prompt = "\n".join(f"- {q}" for q in questions)
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
