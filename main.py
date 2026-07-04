"""Entry point — run the AI-SDLC pipeline with streaming output and HITL support."""
from __future__ import annotations

import sys
import uuid

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from checkpointer import get_checkpointer
from graph import compile_graph
from state import Phase, SDLCState

console = Console()


def _print_event(event: dict) -> None:
    for node_name, node_output in event.items():
        if node_name == "__interrupt__":
            continue
        phase = node_output.get("phase", "")
        console.print(
            Panel(
                f"[bold cyan]{node_name}[/bold cyan] → phase: [yellow]{phase}[/yellow]\n"
                + _summarise(node_output),
                title=f"[green]▶ {node_name}[/green]",
                border_style="dim",
            )
        )


def _summarise(output: dict) -> str:
    lines = []
    if "task_graph" in output:
        lines.append(f"Tasks emitted: {len(output['task_graph'])}")
    if "artifacts" in output:
        lines.append(f"Artifacts: {list(output['artifacts'].keys())}")
    if "test_results" in output:
        for r in output["test_results"]:
            status = "✓" if r["passed"] else "✗"
            lines.append(f"  {status} {r['agent']} (coverage={r['coverage']:.0%})")
    if "security_findings" in output:
        lines.append(f"Security findings: {len(output['security_findings'])}")
    if "failed_nodes" in output and output["failed_nodes"]:
        lines.append(f"[red]Failed: {output['failed_nodes']}[/red]")
    return "\n".join(lines) or "(no summary)"


def _handle_interrupt(compiled_graph, config: dict, interrupt_value: dict) -> dict:
    """Present HITL interrupt to the user and collect their response."""
    console.print(
        Panel(
            f"[bold red]⚠ Human Input Required[/bold red]\n\n"
            f"[yellow]Node:[/yellow] {interrupt_value.get('node', '?')}\n"
            f"[yellow]Phase:[/yellow] {interrupt_value.get('phase', '?')}\n\n"
            f"{interrupt_value.get('message', '')}",
            title="[red]HITL Checkpoint[/red]",
            border_style="red",
        )
    )
    human_input = Prompt.ask("[bold]Your response[/bold]")
    return human_input


def run(requirements: str, thread_id: str | None = None) -> None:
    thread_id = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    checkpointer = get_checkpointer()
    app = compile_graph(checkpointer=checkpointer)

    initial_state: SDLCState = {
        "requirements_doc": requirements,
        "clarifications": [],
        "task_graph": [],
        "supervisor_assignments": {},
        "artifacts": {},
        "schema": {},
        "test_results": [],
        "review_comments": [],
        "security_findings": [],
        "perf_results": {},
        "confidence": {},
        "failed_nodes": [],
        "human_feedback": [],
        "iteration": 0,
        "phase": Phase.DESIGN,
        "approved_for_deploy": False,
        "messages": [],
    }

    console.print(Panel(
        f"[bold green]AI-SDLC Starting[/bold green]\nThread: {thread_id}",
        border_style="green",
    ))

    current_input = initial_state

    while True:
        interrupted = False
        interrupt_value = None

        for event in app.stream(current_input, config=config, stream_mode="updates"):
            if "__interrupt__" in event:
                interrupt_value = event["__interrupt__"][0].value
                interrupted = True
                break
            _print_event(event)

        if not interrupted:
            console.print("[bold green]✓ Pipeline complete.[/bold green]")
            break

        human_input = _handle_interrupt(app, config, interrupt_value)
        from langgraph.types import Command
        current_input = Command(resume=human_input)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        req_file = sys.argv[1]
        with open(req_file) as f:
            requirements = f.read()
    else:
        console.print("[bold]Paste your requirements document (end with EOF / Ctrl-D):[/bold]")
        requirements = sys.stdin.read()

    thread = sys.argv[2] if len(sys.argv) > 2 else None
    run(requirements, thread_id=thread)
