"""Backend dev agent — FastAPI / Django code generation."""
from __future__ import annotations

from agents._base_dev import _build_agent_node
from tools.file_tools import FILE_TOOLS
from tools.code_runner import run_ruff

_SYSTEM = """You are a senior backend engineer.
Write production-quality Python (FastAPI preferred) code.
Follow RESTful conventions and use async handlers.
Include docstrings, type hints, and proper error handling.
Write tests alongside implementation in a tests/ directory.
After writing all files, summarise with JSON as instructed.
"""

backend_node = _build_agent_node(
    agent_key="backend",
    system_prompt=_SYSTEM,
    tools=FILE_TOOLS + [run_ruff],
)
