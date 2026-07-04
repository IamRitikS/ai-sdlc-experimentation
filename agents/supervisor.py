"""Supervisor agent — assigns tasks to agents, re-routes failures, gates on human approval."""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt

from config import MAX_ITER, get_llm
from state import Phase, SDLCState, Task

_SYSTEM = """You are a technical project manager (Supervisor).
You receive a task_graph and any failure reports, and you decide:
  1. Which tasks to assign to which agents right now (respecting depends_on).
  2. Whether any failed tasks need to be reassigned or re-specced.
  3. Whether human approval is needed before proceeding.

Respond ONLY with valid JSON:
{
  "assignments": {
    "<agent_name>": [<task_id>, ...]
  },
  "needs_human_approval": <true|false>,
  "approval_reason": "<string or null>",
  "notes": "<optional commentary>"
}
"""


def supervisor_node(state: SDLCState) -> dict:
    llm = get_llm("supervisor")
    iteration = state.get("iteration", 0)

    if iteration >= MAX_ITER and state.get("failed_nodes"):
        human_answer = interrupt(
            {
                "node": "supervisor",
                "phase": state.get("phase", Phase.PLANNING),
                "message": (
                    f"Max retries ({MAX_ITER}) reached. "
                    f"Still failing: {state['failed_nodes']}. "
                    "Please review and provide guidance."
                ),
            }
        )
        return {
            "human_feedback": [
                {
                    "at_phase": str(state.get("phase")),
                    "at_node": "supervisor",
                    "message": human_answer,
                }
            ],
            "iteration": 0,
            "failed_nodes": [],
        }

    task_graph_json = json.dumps(state.get("task_graph", []), indent=2)
    failed_json = json.dumps(state.get("failed_nodes", []))

    messages = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(
            content=(
                f"## Task Graph\n{task_graph_json}\n\n"
                f"## Failed Nodes (need retry)\n{failed_json}\n\n"
                f"## Current Iteration\n{iteration}"
            )
        ),
    ]

    response = llm.invoke(messages)
    try:
        parsed = json.loads(response.content.strip())
    except json.JSONDecodeError:
        parsed = {"assignments": {}, "needs_human_approval": True, "approval_reason": response.content}

    if parsed.get("needs_human_approval"):
        human_answer = interrupt(
            {
                "node": "supervisor",
                "phase": state.get("phase", Phase.PLANNING),
                "message": (
                    f"Supervisor requests approval: {parsed.get('approval_reason', '')}"
                ),
            }
        )
        return {
            "human_feedback": [
                {
                    "at_phase": str(state.get("phase")),
                    "at_node": "supervisor",
                    "message": human_answer,
                }
            ],
        }

    raw_assignments = parsed.get("assignments", {})
    task_map = {t["id"]: t for t in state.get("task_graph", [])}

    supervisor_assignments: dict[str, list[Task]] = {}
    for agent, task_ids in raw_assignments.items():
        supervisor_assignments[agent] = [
            task_map[tid] for tid in task_ids if tid in task_map
        ]

    return {
        "supervisor_assignments": supervisor_assignments,
        "iteration": iteration + 1,
        "failed_nodes": [],
        "phase": Phase.DEVELOPMENT,
    }
