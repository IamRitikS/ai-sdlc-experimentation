"""Integrator agent — merges artifacts, validates interface contracts."""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from config import get_llm
from state import Phase, SDLCState
from tools.file_tools import FILE_TOOLS

_SYSTEM = """You are an integration engineer.
Your job:
1. Review all existing artifacts (backend, frontend, database).
2. Identify interface mismatches (API endpoints, request/response shapes, DB model vs API model).
3. Fix conflicts by patching the relevant files.
4. Ensure the backend OpenAPI spec matches what the frontend consumes.

After resolving all conflicts respond with JSON:
{
  "conflicts_found": <int>,
  "conflicts_resolved": <int>,
  "notes": "<summary>",
  "confidence": <float 0-1>
}
"""


def integrator_node(state: SDLCState) -> dict:
    llm = get_llm("integrator")

    artifact_summary = "\n".join(
        f"- {module}: {art['path']} ({art['language']})"
        for module, art in state.get("artifacts", {}).items()
    )

    graph = create_react_agent(llm.bind_tools(FILE_TOOLS), tools=FILE_TOOLS)
    result = graph.invoke(
        {
            "messages": [
                SystemMessage(content=_SYSTEM),
                HumanMessage(
                    content=(
                        f"## Existing Artifacts\n{artifact_summary}\n\n"
                        "Use read_file to inspect each artifact and fix mismatches."
                    )
                ),
            ]
        }
    )

    last_msg = result["messages"][-1].content
    try:
        summary = json.loads(last_msg)
    except json.JSONDecodeError:
        summary = {"conflicts_found": 0, "conflicts_resolved": 0, "confidence": 0.8}

    return {
        "confidence": {"integrator": summary.get("confidence", 0.8)},
        "phase": Phase.TESTING,
    }
