"""Code review agent — linting, style, complexity."""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from config import get_llm
from state import Comment, SDLCState
from tools.code_runner import run_ruff, run_eslint
from tools.file_tools import FILE_TOOLS

_SYSTEM = """You are a senior code reviewer.
1. Run ruff on Python files and eslint on TypeScript/JavaScript files.
2. Read key source files and comment on: complexity, naming, error handling, patterns.
3. Respond with JSON:
{
  "comments": [{"file_path": "...", "line": <int|null>, "message": "...", "severity": "info|warning|error"}],
  "has_blocking_issues": <bool>,
  "confidence": <float 0-1>
}
"""


def code_reviewer_node(state: SDLCState) -> dict:
    llm = get_llm("review")
    artifact_summary = "\n".join(
        f"- {art['path']}" for art in state.get("artifacts", {}).values()
    )

    tools = FILE_TOOLS + [run_ruff, run_eslint]
    graph = create_react_agent(llm.bind_tools(tools), tools=tools)
    result = graph.invoke(
        {
            "messages": [
                SystemMessage(content=_SYSTEM),
                HumanMessage(content=f"## Files to review\n{artifact_summary}"),
            ]
        }
    )

    last_msg = result["messages"][-1].content
    try:
        summary = json.loads(last_msg)
    except json.JSONDecodeError:
        summary = {"comments": [], "has_blocking_issues": False, "confidence": 0.8}

    raw_comments = summary.get("comments", [])
    comments = [
        Comment(
            agent="code_reviewer",
            file_path=c.get("file_path", ""),
            line=c.get("line"),
            message=c.get("message", ""),
            severity=c.get("severity", "info"),
        )
        for c in raw_comments
    ]

    existing = list(state.get("review_comments", []))
    existing.extend(comments)

    failed = list(state.get("failed_nodes", []))
    if summary.get("has_blocking_issues"):
        failed.append("code_reviewer")

    return {
        "review_comments": existing,
        "failed_nodes": failed,
        "confidence": {"code_reviewer": summary.get("confidence", 0.8)},
    }
