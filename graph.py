"""LangGraph StateGraph — full AI-SDLC pipeline."""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agents.backend import backend_node
from agents.database import database_node
from agents.deployment import deployment_node
from agents.designer import designer_node
from agents.documentation import documentation_node
from agents.frontend import frontend_node
from agents.integrator import integrator_node
from agents.qa.e2e_tester import e2e_tester_node
from agents.qa.integration_tester import integration_tester_node
from agents.qa.unit_tester import unit_tester_node
from agents.review.code_reviewer import code_reviewer_node
from agents.review.perf_agent import perf_agent_node
from agents.review.security_agent import security_agent_node
from agents.supervisor import supervisor_node
from config import MAX_ITER
from state import Phase, SDLCState


# ── Routing helpers ──────────────────────────────────────────────────────────

def _route_after_designer(state: SDLCState) -> str:
    """Loop back to designer until task_graph is produced."""
    if state.get("task_graph"):
        return "supervisor"
    return "designer"


def _route_after_supervisor(state: SDLCState) -> list[Send]:
    """Fan-out dev agents based on supervisor assignments."""
    assignments = state.get("supervisor_assignments", {})
    sends = []
    agent_node_map = {
        "backend": "backend",
        "frontend": "frontend",
        "database": "database",
    }
    for agent_key, tasks in assignments.items():
        node_name = agent_node_map.get(agent_key)
        if node_name and tasks:
            sends.append(Send(node_name, state))
    return sends if sends else [Send("integrator", state)]


def _route_after_integration(state: SDLCState) -> list[Send]:
    """Fan-out all three QA agents in parallel."""
    return [
        Send("unit_tester", state),
        Send("integration_tester", state),
        Send("e2e_tester", state),
    ]


def _route_after_qa(state: SDLCState) -> str:
    """If any QA agent failed, re-route to supervisor; otherwise proceed to review."""
    failed = state.get("failed_nodes", [])
    qa_agents = {"unit_tester", "integration_tester", "e2e_tester"}
    if any(f in qa_agents for f in failed):
        return "supervisor"
    return "review_fan_out"


def _route_after_review_fan_out(state: SDLCState) -> list[Send]:
    """Fan-out all three review agents in parallel."""
    return [
        Send("code_reviewer", state),
        Send("security_agent", state),
        Send("perf_agent", state),
    ]


def _route_after_review(state: SDLCState) -> str:
    """If any review agent flagged issues, re-route to supervisor; else documentation."""
    failed = state.get("failed_nodes", [])
    review_agents = {"code_reviewer", "security", "perf"}
    if any(f in review_agents for f in failed):
        if state.get("iteration", 0) >= MAX_ITER:
            return "documentation"
        return "supervisor"
    return "documentation"


def _route_after_deployment(state: SDLCState) -> str:
    if state.get("approved_for_deploy"):
        return END
    return "supervisor"


# ── Graph assembly ────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    g = StateGraph(SDLCState)

    # Register nodes
    g.add_node("designer", designer_node)
    g.add_node("supervisor", supervisor_node)
    g.add_node("backend", backend_node)
    g.add_node("frontend", frontend_node)
    g.add_node("database", database_node)
    g.add_node("integrator", integrator_node)
    g.add_node("unit_tester", unit_tester_node)
    g.add_node("integration_tester", integration_tester_node)
    g.add_node("e2e_tester", e2e_tester_node)
    g.add_node("code_reviewer", code_reviewer_node)
    g.add_node("security_agent", security_agent_node)
    g.add_node("perf_agent", perf_agent_node)
    g.add_node("documentation", documentation_node)
    g.add_node("deployment", deployment_node)

    # Passthrough fan-out nodes (no logic, just routing via conditional edges)
    g.add_node("review_fan_out", lambda s: {})

    # ── Edges ────────────────────────────────────────────────────────────────

    # Entry
    g.add_edge(START, "designer")

    # Designer → Supervisor (loop until task_graph ready)
    g.add_conditional_edges("designer", _route_after_designer, ["designer", "supervisor"])

    # Supervisor → dev agents (parallel fan-out via Send)
    g.add_conditional_edges("supervisor", _route_after_supervisor, ["backend", "frontend", "database", "integrator"])

    # Dev agents → integrator (all converge)
    g.add_edge("backend", "integrator")
    g.add_edge("frontend", "integrator")
    g.add_edge("database", "integrator")

    # Integrator → QA (parallel fan-out)
    g.add_conditional_edges("integrator", _route_after_integration, ["unit_tester", "integration_tester", "e2e_tester"])

    # QA agents → join → route
    g.add_conditional_edges("unit_tester", _route_after_qa, ["supervisor", "review_fan_out"])
    g.add_conditional_edges("integration_tester", _route_after_qa, ["supervisor", "review_fan_out"])
    g.add_conditional_edges("e2e_tester", _route_after_qa, ["supervisor", "review_fan_out"])

    # Review fan-out → review agents (parallel)
    g.add_conditional_edges("review_fan_out", _route_after_review_fan_out, ["code_reviewer", "security_agent", "perf_agent"])

    # Review agents → route (re-plan or documentation)
    g.add_conditional_edges("code_reviewer", _route_after_review, ["supervisor", "documentation"])
    g.add_conditional_edges("security_agent", _route_after_review, ["supervisor", "documentation"])
    g.add_conditional_edges("perf_agent", _route_after_review, ["supervisor", "documentation"])

    # Documentation → Deployment
    g.add_edge("documentation", "deployment")

    # Deployment → END or back to supervisor (rejected)
    g.add_conditional_edges("deployment", _route_after_deployment, [END, "supervisor"])

    return g


def compile_graph(checkpointer=None):
    g = build_graph()
    return g.compile(checkpointer=checkpointer)
