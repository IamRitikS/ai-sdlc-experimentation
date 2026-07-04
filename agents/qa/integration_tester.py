"""Integration tester agent — tests API contracts between services."""
from agents.qa._base_qa import _build_qa_node
from tools.code_runner import run_pytest
from tools.file_tools import FILE_TOOLS

_SYSTEM = """You are a QA engineer specialising in integration testing.
1. Review the backend API spec (OpenAPI / routes) and the DB schema.
2. Write integration tests that hit real endpoints and verify DB state.
3. Tests live in tests/integration/.
4. Run pytest on the integration test suite.
5. Report results as JSON.
"""

integration_tester_node = _build_qa_node(
    agent_key="integration_tester",
    system_prompt=_SYSTEM,
    tools=FILE_TOOLS + [run_pytest],
)
