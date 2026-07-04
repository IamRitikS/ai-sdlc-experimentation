"""Deployment agent — writes Dockerfile, docker-compose, CI config, deploy scripts."""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.types import interrupt

from config import get_llm
from state import Phase, SDLCState
from tools.file_tools import FILE_TOOLS

_SYSTEM = """You are a DevOps / deployment engineer.
1. Write a Dockerfile for the backend and one for the frontend (multi-stage builds).
2. Write a docker-compose.yml that wires backend + frontend + database.
3. Write a GitHub Actions CI/CD workflow at .github/workflows/ci.yml.
4. Write a deploy.sh script for staging.
5. Respond with JSON:
{"files_written": [...], "confidence": <float 0-1>}
"""


def deployment_node(state: SDLCState) -> dict:
    if not state.get("approved_for_deploy", False):
        human_answer = interrupt(
            {
                "node": "deployment",
                "phase": Phase.DEPLOY,
                "message": (
                    "All reviews passed. Ready to generate deployment artifacts "
                    "and push to staging. Approve to proceed."
                ),
            }
        )
        approved = str(human_answer).strip().lower() in ("yes", "y", "approve", "ok", "approved")
        if not approved:
            return {
                "approved_for_deploy": False,
                "human_feedback": [
                    {"at_phase": "deploy", "at_node": "deployment", "message": human_answer}
                ],
            }

    llm = get_llm("deploy")
    artifact_list = "\n".join(
        f"- {art['path']} ({art['language']})"
        for art in state.get("artifacts", {}).values()
    )

    graph = create_react_agent(llm.bind_tools(FILE_TOOLS), tools=FILE_TOOLS)
    result = graph.invoke(
        {
            "messages": [
                SystemMessage(content=_SYSTEM),
                HumanMessage(content=f"## Project artifacts\n{artifact_list}"),
            ]
        }
    )

    last_msg = result["messages"][-1].content
    try:
        summary = json.loads(last_msg)
    except json.JSONDecodeError:
        summary = {"files_written": [], "confidence": 0.8}

    human_final = interrupt(
        {
            "node": "deployment",
            "phase": Phase.DEPLOY,
            "message": (
                f"Deployment artifacts written: {summary.get('files_written', [])}.\n"
                "Type 'approve' to deploy to PRODUCTION or 'reject' to rollback."
            ),
        }
    )

    go_prod = str(human_final).strip().lower() in ("approve", "yes", "y")

    return {
        "approved_for_deploy": go_prod,
        "confidence": {"deployment": summary.get("confidence", 0.8)},
        "human_feedback": [
            {"at_phase": "deploy", "at_node": "deployment", "message": str(human_final)}
        ],
        "phase": Phase.DEPLOY,
    }
