"""Database agent — schema design, migrations, query generation."""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.types import interrupt

from config import CONFIDENCE_THRESHOLD, get_llm
from state import Phase, SDLCState
from tools.file_tools import FILE_TOOLS

_SYSTEM = """You are a senior database architect.
Design normalized schemas (PostgreSQL preferred).
Write Alembic migration files and SQLAlchemy models (or raw SQL DDL if specified).
Flag any destructive operations (DROP, column removal) explicitly.
After writing all files, respond with JSON:
{
  "schema": {<table_name>: {<col>: <type>}},
  "has_destructive_ops": <bool>,
  "artifacts": [{"module": "...", "path": "...", "language": "sql"}],
  "confidence": <float 0-1>
}
"""


def database_node(state: SDLCState) -> dict:
    llm = get_llm("database")
    tasks = state.get("supervisor_assignments", {}).get("database", [])

    if not tasks:
        return {}

    task_details = "\n\n".join(
        f"### Task {t['id']}: {t['title']}\n{t['description']}" for t in tasks
    )

    graph = create_react_agent(llm.bind_tools(FILE_TOOLS), tools=FILE_TOOLS)
    result = graph.invoke(
        {
            "messages": [
                SystemMessage(content=_SYSTEM),
                HumanMessage(content=f"## Your Tasks\n{task_details}"),
            ]
        }
    )

    last_msg = result["messages"][-1].content
    try:
        summary = json.loads(last_msg)
    except json.JSONDecodeError:
        summary = {"schema": {}, "has_destructive_ops": False, "artifacts": [], "confidence": 0.5}

    confidence_val = summary.get("confidence", 0.5)

    if summary.get("has_destructive_ops"):
        human_answer = interrupt(
            {
                "node": "database",
                "phase": Phase.DEVELOPMENT,
                "message": "Database agent detected destructive migrations (DROP/column removal). Please review and approve.",
            }
        )
        return {
            "human_feedback": [
                {"at_phase": "development", "at_node": "database", "message": human_answer}
            ],
            "schema": summary.get("schema", {}),
            "confidence": {"database": confidence_val},
        }

    if confidence_val < CONFIDENCE_THRESHOLD:
        human_answer = interrupt(
            {
                "node": "database",
                "phase": Phase.DEVELOPMENT,
                "message": f"Database agent has low confidence ({confidence_val:.2f}). Review schema:\n{last_msg}",
            }
        )
        return {
            "human_feedback": [
                {"at_phase": "development", "at_node": "database", "message": human_answer}
            ],
            "confidence": {"database": confidence_val},
        }

    return {
        "schema": summary.get("schema", {}),
        "confidence": {"database": confidence_val},
    }
