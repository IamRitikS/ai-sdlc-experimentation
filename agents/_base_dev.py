"""Shared base for dev agents (Backend, Frontend, Database)."""
from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from langgraph.types import interrupt

from config import CONFIDENCE_THRESHOLD, OUTPUT_DIR, get_llm
from state import CodeArtifact, SDLCState, Task


def _build_agent_node(
    agent_key: str,
    system_prompt: str,
    tools: list[BaseTool],
):
    """Return a LangGraph node function for a dev agent."""

    def node(state: SDLCState) -> dict:
        llm = get_llm(agent_key)
        tasks: list[Task] = state.get("supervisor_assignments", {}).get(agent_key, [])

        if not tasks:
            return {}

        existing_artifacts = json.dumps(
            {k: v["path"] for k, v in state.get("artifacts", {}).items()},
            indent=2,
        )
        schema_info = json.dumps(state.get("schema", {}), indent=2)

        task_details = "\n\n".join(
            f"### Task {t['id']}: {t['title']}\n{t['description']}" for t in tasks
        )

        user_message = (
            f"## Your Tasks\n{task_details}\n\n"
            f"## Existing Artifacts (paths)\n{existing_artifacts}\n\n"
            f"## DB Schema\n{schema_info}\n\n"
            "Use your tools to write all required files. "
            "After completing, respond with a JSON summary:\n"
            '{"artifacts": [{"module": "...", "path": "...", "language": "..."}], '
            '"confidence": <float 0-1>}'
        )

        graph = create_react_agent(llm.bind_tools(tools), tools=tools)
        result = graph.invoke(
            {"messages": [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]}
        )

        last_msg = result["messages"][-1].content
        try:
            summary = json.loads(last_msg)
        except json.JSONDecodeError:
            summary = {"artifacts": [], "confidence": 0.5}

        confidence_val = summary.get("confidence", 0.5)

        if confidence_val < CONFIDENCE_THRESHOLD:
            human_answer = interrupt(
                {
                    "node": agent_key,
                    "message": f"{agent_key} agent has low confidence ({confidence_val:.2f}). Please review and guide:\n\n{last_msg}",
                }
            )
            return {
                "human_feedback": [
                    {"at_phase": "development", "at_node": agent_key, "message": human_answer}
                ],
                "confidence": {agent_key: confidence_val},
            }

        new_artifacts: dict[str, CodeArtifact] = {}
        for entry in summary.get("artifacts", []):
            module = entry.get("module", entry.get("path", "unknown"))
            path = entry.get("path", "")
            full_path = Path(OUTPUT_DIR) / path
            content = full_path.read_text(encoding="utf-8") if full_path.exists() else ""
            new_artifacts[module] = CodeArtifact(
                path=path,
                content=content,
                language=entry.get("language", ""),
            )

        merged = dict(state.get("artifacts", {}))
        merged.update(new_artifacts)

        return {
            "artifacts": merged,
            "confidence": {agent_key: confidence_val},
        }

    node.__name__ = f"{agent_key}_node"
    return node
