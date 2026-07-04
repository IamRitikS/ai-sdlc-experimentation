"""File system tools exposed to dev agents."""
from __future__ import annotations

import os
from pathlib import Path

from langchain_core.tools import tool

from config import OUTPUT_DIR


def _resolve(relative_path: str) -> Path:
    p = Path(OUTPUT_DIR) / relative_path
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


@tool
def write_file(relative_path: str, content: str) -> str:
    """Write content to a file inside the output directory.

    Args:
        relative_path: Path relative to OUTPUT_DIR (e.g. 'backend/main.py').
        content: Full text content to write.
    Returns:
        Confirmation message with absolute path written.
    """
    p = _resolve(relative_path)
    p.write_text(content, encoding="utf-8")
    return f"Written: {p}"


@tool
def read_file(relative_path: str) -> str:
    """Read a file from the output directory.

    Args:
        relative_path: Path relative to OUTPUT_DIR.
    Returns:
        File content as string, or an error message.
    """
    p = _resolve(relative_path)
    if not p.exists():
        return f"ERROR: {p} does not exist."
    return p.read_text(encoding="utf-8")


@tool
def list_files(sub_dir: str = "") -> str:
    """List all files recursively under OUTPUT_DIR (or a sub-directory).

    Args:
        sub_dir: Optional sub-directory relative to OUTPUT_DIR.
    Returns:
        Newline-separated list of relative file paths.
    """
    root = Path(OUTPUT_DIR) / sub_dir
    if not root.exists():
        return f"Directory does not exist: {root}"
    files = [str(p.relative_to(OUTPUT_DIR)) for p in root.rglob("*") if p.is_file()]
    return "\n".join(sorted(files)) or "(empty)"


@tool
def delete_file(relative_path: str) -> str:
    """Delete a file from the output directory.

    Args:
        relative_path: Path relative to OUTPUT_DIR.
    Returns:
        Confirmation or error.
    """
    p = _resolve(relative_path)
    if not p.exists():
        return f"ERROR: {p} does not exist."
    p.unlink()
    return f"Deleted: {p}"


FILE_TOOLS = [write_file, read_file, list_files, delete_file]
