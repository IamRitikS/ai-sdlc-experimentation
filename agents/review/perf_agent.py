"""Performance agent — static perf analysis and load test script generation."""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from config import get_llm
from state import SDLCState
from tools.file_tools import FILE_TOOLS

_SYSTEM = """You are a performance engineer.
1. Review backend route handlers and DB queries for N+1 problems, missing indexes, and blocking calls.
2. Write a locust load-test script at tests/perf/locustfile.py targeting the key endpoints.
3. Provide static analysis findings and recommendations.
4. Respond with JSON:
{
  "issues": [{"file_path": "...", "description": "...", "severity": "low|medium|high"}],
  "locust_script_written": <bool>,
  "confidence": <float 0-1>
}
"""


def perf_agent_node(state: SDLCState) -> dict:
    llm = get_llm("review")

    artifact_summary = "\n".join(
        f"- {art['path']}" for art in state.get("artifacts", {}).values()
        if art.get("language") in ("python", "sql")
    )

    graph = create_react_agent(llm.bind_tools(FILE_TOOLS), tools=FILE_TOOLS)
    result = graph.invoke(
        {
            "messages": [
                SystemMessage(content=_SYSTEM),
                HumanMessage(content=f"## Backend/DB files to analyse\n{artifact_summary}"),
            ]
        }
    )

    last_msg = result["messages"][-1].content
    try:
        summary = json.loads(last_msg)
    except json.JSONDecodeError:
        summary = {"issues": [], "locust_script_written": False, "confidence": 0.7}

    high_severity = [i for i in summary.get("issues", []) if i.get("severity") == "high"]
    failed = list(state.get("failed_nodes", []))
    if high_severity:
        failed.append("perf")

    return {
        "perf_results": summary,
        "failed_nodes": failed,
        "confidence": {"perf": summary.get("confidence", 0.7)},
    }
