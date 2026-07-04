# AI-SDLC: Automated Software Development Lifecycle
## Architecture Mental Model (LangGraph / LangChain)

---

## 1. Core Philosophy

A directed graph of specialized agents sharing a **single mutable state object**,
orchestrated by a Supervisor. Edges are conditional — routing decisions are made at
runtime based on confidence scores, test results, and human approvals.
Every node can interrupt execution and escalate to a human checkpoint.

---

## 2. Shared State Schema

```python
class Phase(str, Enum):
    DESIGN      = "design"       # Designer + human clarification loop
    PLANNING    = "planning"     # Supervisor breaks task_graph into agent assignments
    DEVELOPMENT = "development"  # Backend / Frontend / Database agents (parallel)
    INTEGRATION = "integration"  # Integrator merges artifacts
    TESTING     = "testing"      # QA agents (parallel)
    REVIEW      = "review"       # Code review / Security / Perf (parallel)
    DEPLOY      = "deploy"       # Deployment agent → production

class SDLCState(TypedDict):
    # Input (Designer phase)
    requirements_doc: str               # raw requirements document from human
    clarifications: list[str]           # Q&A exchanges between Designer and human
    task_graph: list[Task]              # ordered task list with specs, emitted by Designer

    # Supervisor planning
    supervisor_assignments: dict[str, list[Task]]  # {agent_name: [tasks assigned]}

    # Artifacts (keyed by module)
    artifacts: dict[str, CodeArtifact]  # {module: {path, content, language}}
    schema: dict                        # DB schema / migrations

    # Review & Test Results
    test_results: list[TestResult]      # unit / integration / e2e
    review_comments: list[Comment]      # code review feedback
    security_findings: list[Finding]    # OWASP / SAST results
    perf_results: dict                  # load test metrics

    # Control Flow
    confidence: dict[str, float]        # per-agent confidence scores (agent → 0.0–1.0)
    failed_nodes: list[str]             # nodes returned to Supervisor for re-delegation
    human_feedback: list[HumanNote]     # injected at any HITL checkpoint
    iteration: int                      # feedback loop counter (bounded by max_iter)
    phase: Phase                        # current SDLC phase (see Phase enum above)
    approved_for_deploy: bool
```

---

## 3. LangGraph Control Flow

```
Requirements doc    ┌──────────────┐
from human   ──────▶│   Designer   │
Clarifications ◀────│  (architect) │
                    └──────┬───────┘
                           │ emits task_graph (list of all tasks to do in order and their specifications)
                           ▼
                    ┌──────────────┐
                    │  Supervisor  │  
                    │  (manager)   │◀─────────────────────────────┐
                    └──────┬───────┘                              │
                           │ (gives tasks to specialized agents)  │ (needs human approval before giving go-ahead)
          ┌────────────────┼─────────────────┐                    │
          ▼                ▼                 ▼                    │ re-route
   ┌─────────────┐  ┌──────────────┐  ┌──────────────┐            │ on failure
   │   Backend   │  │   Frontend   │  │   Database   │            │
   └──────┬──────┘  └──────┬───────┘  └──────┬───────┘            │
          └────────────────┴─────────────────┘                    │
                           │ all complete                         │
                           ▼                                      │
                    ┌──────────────┐                              │
                    │  Integrator  │                              │
                    └──────┬───────┘                              │
                           │                                      │
               ┌───────────┼───────────┐                          │
               ▼           ▼           ▼                          │
         ┌──────────┐ ┌─────────┐ ┌────────┐                      │
         │  Unit    │ │  Integ. │ │  E2E   │── FAIL ─────────────▶┤
         │  Tests   │ │  Tests  │ │  Tests │                      │
         └──────────┘ └─────────┘ └────────┘                      │
               │           │           │                          │
               └───────────┴───────────┘                          │
                           │ all pass                             │
               ┌───────────┼───────────┐                          │
               ▼           ▼           ▼                          │
         ┌──────────┐ ┌─────────┐ ┌────────┐                      │
         │  Code    │ │Security │ │  Perf  │── FAIL ─────────────▶┤
         │  Review  │ │  Scan   │ │  Test  │                      │
         └──────────┘ └─────────┘ └────────┘                      │
                           │                                      │
                           │ all clear                            │
                           ▼                                      │
                 ┌──────────────────┐                             │
                 │  Documentation   │                             │
                 └────────┬─────────┘                             │
                          │                                       │
                          ▼                                       │
                 ┌──────────────────┐                             │
                 │  Human Review    │                             │
                 │  Checkpoint      │                             │
                 │  (HITL, async)   │──── reject ────────────────▶┘
                 └────────┬─────────┘
                          │ approve
                          ▼
                 ┌──────────────────┐
                 │  Deployment      │
                 │  Agent (staging) │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │  Final Human     │
                 │  Approval Gate   │──── reject ─── rollback
                 └────────┬─────────┘
                          │ approve
                          ▼
                       PRODUCTION
```

---

## 4. Feedback Loops (Detailed)

| Source Node       | Failure Condition                        | Routed Back To            |
|-------------------|------------------------------------------|---------------------------|
| Unit Tests        | test failure, coverage < threshold       | Supervisor (delegate)     |
| Integration Tests | contract mismatch, API error             | Supervisor (delegate)     |
| E2E Tests         | UI regression, broken flow               | Supervisor (delegate)     |
| Code Review       | quality issues, style violations         | Supervisor (delegate)     |
| Security Scan     | CVE found, injection risk, secrets leak  | Supervisor (delegate)     |
| Perf Tests        | p99 latency > threshold, memory leak     | Supervisor (delegate)     |
| Human Checkpoint  | rejection with comments                  | Supervisor (re-plan)      |

Max iteration cap (`max_iter`) prevents infinite loops — escalates to human after N retries.

---

## 5. Human-in-the-Loop (HITL) Triggers

Execution **interrupts** and blocks on human input under any of these conditions:

1. **Low confidence** — any agent reports `confidence < 0.7`
2. **Ambiguous requirements** — designer cannot resolve conflicting constraints
3. **Architecture decision** — tech stack choice with meaningful tradeoffs
4. **Security critical** — CVSS score >= 7.0 finding
5. **Max retries exceeded** — feedback loop iterated > N times without resolution
6. **Deployment gate** — always requires explicit human approval to push to production
7. **Destructive migration** — DB changes that DROP columns / tables
8. **External integrations** — credentials, rate limits, billing implications

Human input is injected into `state.human_feedback` and execution resumes from the
interrupted node via LangGraph's `interrupt()` / `Command(resume=...)` mechanism.

---

## 6. Agent Toolsets

| Agent          | Tools                                                           |
|----------------|-----------------------------------------------------------------|
| Designer       | requirements parser, dependency analyzer, task graph builder    |
| Supervisor     | tasks analyzer, agents orchestrator, delegate tasks to agents   |
| Backend        | code writer, file editor, test runner, API spec generator       |
| Frontend       | code writer, component library lookup, design system reference  |
| Database       | schema generator, migration writer, query optimizer             |
| Integrator     | git diff, conflict resolver, interface contract validator       |
| Unit Tester    | test framework runner (pytest / jest), coverage reporter        |
| Integration    | service composer (docker-compose), API tester (httpx)           |
| E2E Tester     | browser automation (Playwright), visual diff                    |
| Code Reviewer  | linter, static analyzer (ruff / eslint), complexity scorer      |
| Security       | SAST (semgrep), dependency audit (pip-audit / npm audit)        |
| Performance    | load tester (locust / k6), profiler, flame graph analyzer       |
| Documentation  | docstring generator, README writer, OpenAPI spec builder        |
| Deployment     | container registry push, k8s apply, env config manager          |

---

## 7. LangGraph Implementation Notes

```
Graph type:    StateGraph (not MessageGraph)
Parallelism:   Send() API for fan-out to Dev Agents and QA/Review Agents simultaneously
Checkpointing: SqliteSaver (dev) / PostgresSaver (prod) for cross-session persistence
HITL:          interrupt() at checkpoint nodes; resume via Command(resume=<human_input>)
Subgraphs:     QA cluster and Review cluster compiled as nested subgraphs
Memory:        Long-term artifact store outside state (vector DB for code search/retrieval)
Retries:       Each node wrapped with exponential backoff + max_retries config
Streaming:     Token-level output to UI via .astream_events()
```

---

## 8. Phase Lifecycle

```
DESIGN --> PLANNING --> DEVELOPMENT --> INTEGRATION --> TESTING --> REVIEW --> DEPLOY
    |           |            |               |             |          |          |
 Designer     Supervisor   Dev Agents      Integrator     QA Agents  Review    Deployment
 Architect                 (parallel)                    (parallel)  Agents    Agent
                                                                    HITL gate HITL gate
```

---

## 9. Key Design Decisions

- **Supervisor pattern** (not full autonomy): Supervisor routes but does not generate
  code. Individual agents own their domain artifacts exclusively.
- **Interface contracts first**: Backend and Frontend agents agree on an OpenAPI spec
  *before* generating implementation — eliminates integration mismatch at the seam.
- **Stateless agents, stateful graph**: Agents are pure functions over state; all memory
  lives in `SDLCState` and the persistent checkpoint store.
- **Confidence-gated HITL**: Agents self-report uncertainty rather than relying on
  hardcoded rules, enabling dynamic and context-sensitive human involvement.
- **Fail-fast QA**: Unit tests run as soon as a module is ready — not after all dev
  agents finish — catching regressions at the earliest possible point in the loop.
- **Bounded iteration**: Every feedback cycle increments `state.iteration`; hitting
  `max_iter` hard-routes to a human checkpoint instead of looping forever.
