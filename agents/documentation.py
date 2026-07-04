"""Documentation agent — generates README, docstrings, OpenAPI summary."""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from config import get_llm
from state import Phase, SDLCState
from tools.file_tools import FILE_TOOLS

_SYSTEM = """You are a technical writer.
1. Read all source files.
2. Write / update docs/README.md with: project overview, architecture, setup, usage.
3. Add or improve docstrings in Python files (Google style).
4. If an OpenAPI spec exists, write a concise API reference in docs/api.md.
5. Respond with JSON:
{"files_written": [...], "confidence": <float 0-1>}
"""


def documentation_node(state: SDLCState) -> dict:
    llm = get_llm("docs")

    artifact_list = "\n".join(
        f"- {art['path']}" for art in state.get("artifacts", {}).values()
    )

    graph = create_react_agent(llm.bind_tools(FILE_TOOLS), tools=FILE_TOOLS)
    result = graph.invoke(
        {
            "messages": [
                SystemMessage(content=_SYSTEM),
                HumanMessage(content=f"## Source files\n{artifact_list}"),
            ]
        }
    )

    last_msg = result["messages"][-1].content
    try:
        summary = json.loads(last_msg)
    except json.JSONDecodeError:
        summary = {"files_written": [], "confidence": 0.8}

    return {
        "confidence": {"documentation": summary.get("confidence", 0.8)},
        "phase": Phase.DEPLOY,
    }
