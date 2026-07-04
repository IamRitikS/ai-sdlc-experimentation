"""Unit tester agent — runs pytest and jest unit tests."""
from agents.qa._base_qa import _build_qa_node
from tools.code_runner import run_pytest, run_jest
from tools.file_tools import FILE_TOOLS

_SYSTEM = """You are a QA engineer specialising in unit testing.
1. Inspect existing test files (tests/ and __tests__/).
2. Write any missing unit tests for untested functions/components.
3. Run pytest for backend and jest for frontend.
4. Report results as JSON.
"""

unit_tester_node = _build_qa_node(
    agent_key="unit_tester",
    system_prompt=_SYSTEM,
    tools=FILE_TOOLS + [run_pytest, run_jest],
)
