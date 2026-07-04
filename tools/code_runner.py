"""Tools for running tests, linters, and static analysis."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from langchain_core.tools import tool

from config import OUTPUT_DIR


def _run(cmd: list[str], cwd: str | None = None) -> tuple[int, str]:
    result = subprocess.run(
        cmd,
        cwd=cwd or OUTPUT_DIR,
        capture_output=True,
        text=True,
        timeout=120,
    )
    out = (result.stdout + result.stderr).strip()
    return result.returncode, out


@tool
def run_pytest(sub_dir: str = "", extra_args: str = "--tb=short -q") -> str:
    """Run pytest in the output directory (or a sub-directory).

    Args:
        sub_dir: Sub-directory relative to OUTPUT_DIR to run tests in.
        extra_args: Additional pytest arguments string.
    Returns:
        Combined stdout/stderr from pytest.
    """
    cwd = str(Path(OUTPUT_DIR) / sub_dir) if sub_dir else OUTPUT_DIR
    cmd = [sys.executable, "-m", "pytest"] + extra_args.split()
    code, out = _run(cmd, cwd=cwd)
    prefix = "PASS" if code == 0 else "FAIL"
    return f"[{prefix}]\n{out}"


@tool
def run_jest(sub_dir: str = "", extra_args: str = "--watchAll=false") -> str:
    """Run jest in a frontend sub-directory.

    Args:
        sub_dir: Sub-directory relative to OUTPUT_DIR.
        extra_args: Additional jest CLI arguments.
    Returns:
        Combined stdout/stderr from jest.
    """
    cwd = str(Path(OUTPUT_DIR) / sub_dir) if sub_dir else OUTPUT_DIR
    cmd = ["npx", "jest"] + extra_args.split()
    code, out = _run(cmd, cwd=cwd)
    prefix = "PASS" if code == 0 else "FAIL"
    return f"[{prefix}]\n{out}"


@tool
def run_ruff(relative_path: str = ".") -> str:
    """Run ruff linter on a file or directory inside OUTPUT_DIR.

    Args:
        relative_path: Path relative to OUTPUT_DIR (default: whole project).
    Returns:
        Ruff output.
    """
    target = str(Path(OUTPUT_DIR) / relative_path)
    code, out = _run([sys.executable, "-m", "ruff", "check", target])
    return out or "No issues found."


@tool
def run_eslint(relative_path: str = ".") -> str:
    """Run ESLint on a file or directory inside OUTPUT_DIR.

    Args:
        relative_path: Path relative to OUTPUT_DIR.
    Returns:
        ESLint output.
    """
    target = str(Path(OUTPUT_DIR) / relative_path)
    code, out = _run(["npx", "eslint", target, "--format=compact"])
    return out or "No issues found."


@tool
def run_semgrep(relative_path: str = ".") -> str:
    """Run semgrep SAST scan on a path inside OUTPUT_DIR.

    Args:
        relative_path: Path relative to OUTPUT_DIR.
    Returns:
        Semgrep findings or 'No issues found.'
    """
    target = str(Path(OUTPUT_DIR) / relative_path)
    code, out = _run(["semgrep", "--config=auto", "--json", target])
    return out or "No issues found."


@tool
def run_pip_audit(sub_dir: str = "") -> str:
    """Run pip-audit to check for vulnerable dependencies.

    Args:
        sub_dir: Sub-directory with requirements.txt.
    Returns:
        pip-audit output.
    """
    cwd = str(Path(OUTPUT_DIR) / sub_dir) if sub_dir else OUTPUT_DIR
    code, out = _run(["pip-audit", "-r", "requirements.txt"], cwd=cwd)
    return out or "No vulnerabilities found."


CODE_RUNNER_TOOLS = [run_pytest, run_jest, run_ruff, run_eslint, run_semgrep, run_pip_audit]
