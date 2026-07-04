"""E2E tester agent — writes and runs Playwright tests."""
from agents.qa._base_qa import _build_qa_node
from tools.file_tools import FILE_TOOLS
from tools.code_runner import run_pytest

_SYSTEM = """You are a QA engineer specialising in end-to-end testing with Playwright.
1. Review the frontend components and user flows.
2. Write Playwright tests (Python) covering critical user journeys.
3. Tests live in tests/e2e/.
4. Run pytest with pytest-playwright.
5. Report results as JSON.
"""

e2e_tester_node = _build_qa_node(
    agent_key="e2e_tester",
    system_prompt=_SYSTEM,
    tools=FILE_TOOLS + [run_pytest],
)
