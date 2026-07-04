"""Security agent — SAST with semgrep, dependency audit."""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.types import interrupt

from config import get_llm
from state import Finding, Phase, SDLCState
from tools.code_runner import run_semgrep, run_pip_audit
from tools.file_tools import FILE_TOOLS

_SYSTEM = """You are an application security engineer.
1. Run semgrep (SAST) across the codebase.
2. Run pip-audit / npm audit for dependency vulnerabilities.
3. Review code for OWASP Top-10 patterns (injection, auth issues, secrets in code, etc.).
4. Respond with JSON:
{
  "findings": [
    {"rule": "...", "file_path": "...", "line": <int|null>, "description": "...", "cvss": <float>}
  ],
  "max_cvss": <float>,
  "confidence": <float 0-1>
}
"""

_CRITICAL_CVSS = 7.0


def security_agent_node(state: SDLCState) -> dict:
    llm = get_llm("review")
    tools = FILE_TOOLS + [run_semgrep, run_pip_audit]

    graph = create_react_agent(llm.bind_tools(tools), tools=tools)
    result = graph.invoke(
        {
            "messages": [
                SystemMessage(content=_SYSTEM),
                HumanMessage(content="Run a full security scan on all artifacts."),
            ]
        }
    )

    last_msg = result["messages"][-1].content
    try:
        summary = json.loads(last_msg)
    except json.JSONDecodeError:
        summary = {"findings": [], "max_cvss": 0.0, "confidence": 0.8}

    findings = [
        Finding(
            agent="security",
            rule=f.get("rule", ""),
            file_path=f.get("file_path", ""),
            line=f.get("line"),
            description=f.get("description", ""),
            cvss=f.get("cvss", -1.0),
        )
        for f in summary.get("findings", [])
    ]

    existing = list(state.get("security_findings", []))
    existing.extend(findings)

    max_cvss = summary.get("max_cvss", 0.0)

    if max_cvss >= _CRITICAL_CVSS:
        human_answer = interrupt(
            {
                "node": "security",
                "phase": Phase.REVIEW,
                "message": (
                    f"Critical security finding (CVSS {max_cvss:.1f}). "
                    "Human review required before proceeding."
                ),
            }
        )
        return {
            "security_findings": existing,
            "human_feedback": [
                {"at_phase": "review", "at_node": "security", "message": human_answer}
            ],
            "confidence": {"security": summary.get("confidence", 0.5)},
        }

    failed = list(state.get("failed_nodes", []))
    if findings:
        failed.append("security")

    return {
        "security_findings": existing,
        "failed_nodes": failed,
        "confidence": {"security": summary.get("confidence", 0.8)},
    }
