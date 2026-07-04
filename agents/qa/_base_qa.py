"""Shared base for QA agents."""
from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from config import get_llm
from state import SDLCState, TestResult


def _build_qa_node(
    agent_key: str,
    system_prompt: str,
    tools: list[BaseTool],
    llm_prefix: str = "qa",
):
    def node(state: SDLCState) -> dict:
        llm = get_llm(llm_prefix)

        artifact_summary = "\n".join(
            f"- {module}: {art['path']} ({art['language']})"
            for module, art in state.get("artifacts", {}).items()
        )

        graph = create_react_agent(llm.bind_tools(tools), tools=tools)
        result = graph.invoke(
            {
                "messages": [
                    SystemMessage(content=system_prompt),
                    HumanMessage(
                        content=(
                            f"## Artifacts to test\n{artifact_summary}\n\n"
                            "Run the relevant tests and tools. "
                            "Respond with JSON:\n"
                            '{"passed": <bool>, "coverage": <float>, '
                            '"output": "<summary>", "failed_tests": [...]}'
                        )
                    ),
                ]
            }
        )

        last_msg = result["messages"][-1].content
        try:
            summary = json.loads(last_msg)
        except json.JSONDecodeError:
            passed = "PASS" in last_msg.upper()
            summary = {"passed": passed, "coverage": -1.0, "output": last_msg, "failed_tests": []}

        test_result = TestResult(
            agent=agent_key,
            passed=summary.get("passed", False),
            coverage=summary.get("coverage", -1.0),
            output=summary.get("output", ""),
            failed_tests=summary.get("failed_tests", []),
        )

        existing = list(state.get("test_results", []))
        existing.append(test_result)

        failed_nodes = list(state.get("failed_nodes", []))
        if not test_result["passed"]:
            failed_nodes.append(agent_key)

        return {
            "test_results": existing,
            "failed_nodes": failed_nodes,
            "confidence": {agent_key: 1.0 if test_result["passed"] else 0.3},
        }

    node.__name__ = f"{agent_key}_node"
    return node
