from __future__ import annotations

from enum import Enum
from typing import Annotated, Any
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages


class Phase(str, Enum):
    DESIGN      = "design"
    PLANNING    = "planning"
    DEVELOPMENT = "development"
    INTEGRATION = "integration"
    TESTING     = "testing"
    REVIEW      = "review"
    DEPLOY      = "deploy"


class Task(TypedDict):
    id: str
    title: str
    description: str
    agent: str                   # which agent owns this task
    depends_on: list[str]        # task ids this one depends on
    priority: int                # lower = higher priority
    status: str                  # pending | in_progress | done | failed


class CodeArtifact(TypedDict):
    path: str                    # relative path inside OUTPUT_DIR
    content: str                 # full file content
    language: str                # python | typescript | sql | yaml | ...


class TestResult(TypedDict):
    agent: str                   # which tester produced this
    passed: bool
    coverage: float              # 0.0–1.0, -1 if not applicable
    output: str                  # raw stdout/stderr
    failed_tests: list[str]


class Comment(TypedDict):
    agent: str
    file_path: str
    line: int | None
    message: str
    severity: str                # info | warning | error


class Finding(TypedDict):
    agent: str
    rule: str
    file_path: str
    line: int | None
    description: str
    cvss: float                  # 0.0–10.0; -1 if N/A


class HumanNote(TypedDict):
    at_phase: str
    at_node: str
    message: str


class SDLCState(TypedDict):
    # ── Designer phase ──────────────────────────────────────────
    requirements_doc: str
    clarifications: Annotated[list[str], add_messages]

    # ── Supervisor planning ──────────────────────────────────────
    task_graph: list[Task]
    supervisor_assignments: dict[str, list[Task]]

    # ── Artifacts ───────────────────────────────────────────────
    artifacts: dict[str, CodeArtifact]
    schema: dict[str, Any]

    # ── QA & Review results ──────────────────────────────────────
    test_results: list[TestResult]
    review_comments: list[Comment]
    security_findings: list[Finding]
    perf_results: dict[str, Any]

    # ── Control flow ─────────────────────────────────────────────
    confidence: dict[str, float]
    failed_nodes: list[str]
    human_feedback: Annotated[list[HumanNote], add_messages]
    iteration: int
    phase: Phase
    approved_for_deploy: bool
    messages: Annotated[list, add_messages]  # scratch-pad for agent reasoning
