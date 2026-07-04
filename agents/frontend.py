"""Frontend dev agent — React / Next.js code generation."""
from __future__ import annotations

from agents._base_dev import _build_agent_node
from tools.file_tools import FILE_TOOLS
from tools.code_runner import run_eslint

_SYSTEM = """You are a senior frontend engineer.
Write production-quality React / Next.js (TypeScript) code.
Use functional components, hooks, and modern patterns.
Follow accessibility (WCAG 2.1 AA) and responsive design principles.
Include component tests in a __tests__ directory using React Testing Library.
After writing all files, summarise with JSON as instructed.
"""

frontend_node = _build_agent_node(
    agent_key="frontend",
    system_prompt=_SYSTEM,
    tools=FILE_TOOLS + [run_eslint],
)
