"""CLI entry point using Typer."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from orpilot.llm.config import LLMConfig, get_llm
from orpilot.workflow.graph import build_graph
from orpilot.workflow.state import WorkflowState
from orpilot.models.problem import ProblemDefinition
from orpilot.models.data import UserData

app = typer.Typer(
    name="orpilot",
    help="AI Operations Research Agent — LLM-powered OR modeling and solving",
)
console = Console()

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

_NODE_LABELS: dict[str, tuple[str, str]] = {
    "interview": ("blue", "Conducting interview..."),
    "data_collection": ("blue", "Collecting data..."),
    "ir_builder": ("yellow", "Starting model building (translating problem to IR)..."),
    "ir_compiler": ("yellow", "Compiling IR to solver code..."),
    "solver_runner": ("yellow", "Starting model solving..."),
    "reporter": ("green", "Generating solution report..."),
}

_NODE_COMPLETE_LABELS: dict[str, tuple[str, str]] = {
    "interview": ("green", "Interview finished — problem defined."),
    "data_collection": ("green", "Data collection finished — all CSV files loaded."),
    "ir_builder": ("green", "IR model built."),
    "ir_compiler": ("green", "Model building finished — solver code ready."),
    "solver_runner": ("green", "Model solving finished."),
}


def _log_entering_node(node: str) -> None:
    """Print a status line when entering a workflow node."""
    style, msg = _NODE_LABELS.get(node, ("dim", f"Running {node}..."))
    console.print(f"[{style}]>> {msg}[/{style}]")


# ---------------------------------------------------------------------------
# Artifact helpers
# ---------------------------------------------------------------------------


def _save_artifacts(
    state: dict,
    output_dir: str,
    last_saved_code: str,
    con: Console,
) -> str:
    """Save generated code and LP file to output_dir. Returns the last saved code."""
    code = state.get("generated_code", "")
    out = Path(output_dir)

    # Save generated Python code whenever it changes
    if code and code != last_saved_code:
        model_path = out / "model.py"
        model_path.write_text(code, encoding="utf-8")
        con.print(f"[dim]  -> Saved generated code to {model_path}[/dim]")

    # Save IR model whenever it is available
    ir_model = state.get("ir_model")
    if ir_model:
        ir_path = out / "ir.json"
        ir_path.write_text(json.dumps(ir_model, indent=2), encoding="utf-8")
        con.print(f"[dim]  -> Saved IR to {ir_path}[/dim]")

    # Save LP file from solution if available
    solution = state.get("solution")
    if solution and solution.lp_content:
        lp_path = out / "model.lp"
        lp_path.write_text(solution.lp_content, encoding="utf-8")
        con.print(f"[dim]  -> Saved LP file to {lp_path}[/dim]")

    return code if code else last_saved_code


def _parse_variable_dimensions(
    variables: dict,
    group_name: str,
    dimension_labels: list[str] | None = None,
) -> tuple[list[str], list[list]]:
    """Parse variable names into dimension columns for a single variable group.

    Variable names follow the pattern ``prefix_dim1_dim2_...``.  The *prefix*
    (which equals *group_name*) is stripped — only dimension values and the
    solution value appear in the rows.

    Returns (headers, rows).
    """
    import re

    parsed: list[tuple[list[str], object]] = []
    max_dims = 0

    for var_name, value in sorted(variables.items()):
        # Try tuple-style: ship_('WH1',_'CUST2')
        tuple_match = re.match(r"^([^(]+?)_?\((.+)\)$", var_name)
        if tuple_match:
            inner = tuple_match.group(2)
            dims = [
                d.strip().strip("'\"").strip("_").strip()
                for d in inner.split(",")
                if d.strip().strip("'\"").strip("_").strip()
            ]
        else:
            # Underscore-separated: shipment_WH1_CUST1 → strip prefix
            parts = var_name.split("_")
            if len(parts) >= 2:
                # Remove the prefix (group_name may itself contain underscores)
                prefix_parts = group_name.split("_")
                if parts[: len(prefix_parts)] == prefix_parts:
                    dims = parts[len(prefix_parts):]
                else:
                    dims = parts[1:]
            else:
                dims = []

        max_dims = max(max_dims, len(dims))
        parsed.append((dims, value))

    # Build headers
    labels = dimension_labels or []
    headers: list[str] = []
    for i in range(max_dims):
        if i < len(labels):
            headers.append(labels[i])
        else:
            headers.append(f"dim_{i + 1}")
    headers.append("value")

    # Build rows
    rows: list[list] = []
    for dims, value in parsed:
        row: list = []
        for i in range(max_dims):
            row.append(dims[i] if i < len(dims) else "")
        row.append(value)
        rows.append(row)

    return headers, rows


def _save_solution(state: dict, output_dir: str, con: Console) -> None:
    """Save objective value as txt and each variable group as its own CSV."""
    solution = state.get("solution")
    if not solution:
        return

    out = Path(output_dir)

    # Save optimization summary
    summary_path = out / "optimization_summary.txt"
    lines = [
        f"Status: {solution.status.value}",
    ]
    if solution.objective_value is not None:
        lines.append(f"Objective Value: {solution.objective_value}")
    if solution.solve_time_seconds is not None:
        lines.append(f"Solve Time: {solution.solve_time_seconds:.4f}s")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    con.print(f"[dim]  -> Saved optimization summary to {summary_path}[/dim]")

    # Save one CSV per variable group
    if solution.variable_groups:
        for group in solution.variable_groups:
            if not group.variables:
                continue
            filename = f"solution_{group.group_name}.csv"
            csv_path = out / filename
            headers, rows = _parse_variable_dimensions(
                group.variables,
                group_name=group.group_name,
                dimension_labels=group.dimension_labels or None,
            )
            with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(headers)
                for row in rows:
                    writer.writerow(row)
            con.print(f"[dim]  -> Saved {group.group_name} solution values to {csv_path}[/dim]")
    elif solution.variables:
        # Fallback: no groups returned, dump all variables into a single CSV
        csv_path = out / "solution_decisions.csv"
        headers, rows = _parse_variable_dimensions(
            solution.variables,
            group_name="",
        )
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(headers)
            for row in rows:
                writer.writerow(row)
        con.print(f"[dim]  -> Saved decision variable solution values to {csv_path}[/dim]")


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


@app.command()
def run(
    provider: str = typer.Option("openai", "--provider", "-p", help="LLM provider (openai, anthropic)"),
    model: str = typer.Option(None, "--model", "-m", help="Model name override"),
    solver: str = typer.Option("pulp", "--solver", "-s", help="OR solver (pulp, pyomo, ortools)"),
    problem_file: Path = typer.Option(None, "--problem", help="Load problem definition from JSON file"),
    data_file: Path = typer.Option(None, "--data", help="Load data from JSON file"),
    data_dir: Path = typer.Option("./data", "--data-dir", "-d", help="Directory for CSV data files"),
    output_dir: Path = typer.Option(None, "--output-dir", "-o", help="Directory to save generated code, LP file, and solution"),
    max_retries: int = typer.Option(3, "--max-retries", help="Max solver code retries"),
    api_key: str = typer.Option(None, "--api-key", envvar="OPENAI_API_KEY"),
    base_url: str = typer.Option(None, "--base-url", envvar="OPENAI_BASE_URL", help="Custom API base URL (e.g. https://api.deepseek.com)"),
) -> None:
    """Start an interactive ORPilot session."""
    from dotenv import load_dotenv
    load_dotenv()

    llm_config = LLMConfig(provider=provider, model=model, api_key=api_key, base_url=base_url)
    llm = get_llm(llm_config)
    graph = build_graph(llm=llm)

    # Ensure data directory exists
    data_dir.mkdir(parents=True, exist_ok=True)

    # Ensure output directory exists if specified
    output_dir_str = ""
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dir_str = str(output_dir)

    # Initialize state
    state: WorkflowState = {
        "messages": [],
        "problem": None,
        "user_data": None,
        "ir_model": None,
        "generated_code": "",
        "solution": None,
        "report": "",
        "current_node": "interview",
        "solver_name": solver,
        "retry_count": 0,
        "max_retries": max_retries,
        "error_context": "",
        "needs_user_input": False,
        "user_input": "",
        "llm_config": llm_config.__dict__,
        "data_dir": str(data_dir),
        "csv_specs": [],
        "output_dir": output_dir_str,
    }

    # Load problem from file if provided
    if problem_file and problem_file.exists():
        problem = ProblemDefinition.model_validate_json(problem_file.read_text())
        state["problem"] = problem
        state["current_node"] = "data_collection"
        console.print(Panel(f"Loaded problem: {problem.title}", title="Problem"))

    # Load data from file if provided
    if data_file and data_file.exists():
        data = UserData.model_validate_json(data_file.read_text())
        state["user_data"] = data
        if state.get("problem"):
            state["current_node"] = "ir_builder"
        console.print(Panel("Loaded data from file", title="Data"))

    console.print(Panel(
        "Welcome to ORPilot — AI Operations Research Agent\n"
        "I'll help you model and solve optimization problems.\n"
        "Type 'quit' to exit at any time.",
        title="ORPilot",
        border_style="blue",
    ))

    _last_saved_code = ""

    while True:
        # Stream the graph one node at a time so we can log each step.
        for chunk in graph.stream(state, stream_mode="updates"):
            for node_name, node_update in chunk.items():
                if node_name.startswith("__"):
                    continue

                prev_state = state
                state = {**state, **node_update}

                # wait_for_input is an infrastructure node — skip logging.
                if node_name == "wait_for_input":
                    continue

                # The interview node doubles as a router once the problem is
                # defined.  Suppress its "entering" log in that case.
                interview_passthrough = (
                    node_name == "interview"
                    and prev_state.get("problem") is not None
                )
                if not interview_passthrough:
                    _log_entering_node(node_name)

                # ── Milestone completion logging ──────────────────────────
                if node_name == "interview":
                    if (prev_state.get("problem") is None
                            and state.get("problem") is not None):
                        style, msg = _NODE_COMPLETE_LABELS["interview"]
                        console.print(f"[{style}]✓ {msg}[/{style}]")

                elif node_name == "data_collection":
                    if (prev_state.get("user_data") is None
                            and state.get("user_data") is not None):
                        style, msg = _NODE_COMPLETE_LABELS["data_collection"]
                        console.print(f"[{style}]✓ {msg}[/{style}]")

                elif node_name == "ir_builder":
                    if state.get("ir_model"):
                        style, msg = _NODE_COMPLETE_LABELS["ir_builder"]
                        console.print(f"[{style}]✓ {msg}[/{style}]")

                elif node_name == "ir_compiler":
                    if state.get("generated_code"):
                        style, msg = _NODE_COMPLETE_LABELS["ir_compiler"]
                        console.print(f"[{style}]✓ {msg}[/{style}]")

                elif node_name == "solver_runner":
                    solution = state.get("solution")
                    if solution:
                        if solution.error_message:
                            retry = state.get("retry_count", 0)
                            max_r = state.get("max_retries", 3)
                            console.print(
                                f"[red]✗ Solver error (attempt {retry}/{max_r}): "
                                f"{solution.error_message[:200]}[/red]"
                            )
                            console.print("[yellow]   Retrying with error feedback...[/yellow]")
                        else:
                            style, msg = _NODE_COMPLETE_LABELS["solver_runner"]
                            console.print(f"[{style}]✓ {msg}[/{style}]")

        # Save debug artifacts when output_dir is set
        if output_dir_str:
            _last_saved_code = _save_artifacts(state, output_dir_str, _last_saved_code, console)

        # Check if we have a final report
        if state.get("report"):
            # Save solution outputs
            if output_dir_str:
                _save_solution(state, output_dir_str, console)

            console.print()
            console.print(Panel(
                Markdown(state["report"]),
                title="Solution Report",
                border_style="green",
            ))

            # Show solution details
            solution = state.get("solution")
            if solution:
                console.print(f"\nStatus: {solution.status.value}")
                if solution.objective_value is not None:
                    console.print(f"Objective Value: {solution.objective_value}")
                if solution.solve_time_seconds is not None:
                    console.print(f"Solve Time: {solution.solve_time_seconds:.2f}s")
            break

        # If needs user input, prompt
        if state.get("needs_user_input"):
            # Show the last assistant message
            messages = state.get("messages", [])
            if messages and messages[-1]["role"] == "assistant":
                console.print()
                console.print(Markdown(messages[-1]["content"]))
                console.print()

            user_input = console.input("[bold blue]You:[/bold blue] ")
            if user_input.strip().lower() in ("quit", "exit", "q"):
                console.print("Goodbye!")
                raise typer.Exit()

            state["messages"].append({"role": "user", "content": user_input})
            state["needs_user_input"] = False


@app.command()
def config() -> None:
    """Show current configuration."""
    import os
    from dotenv import load_dotenv
    load_dotenv()

    console.print(Panel("ORPilot Configuration", border_style="blue"))
    console.print(f"LLM Provider: {os.getenv('ORPILOT_LLM_PROVIDER', 'openai')}")
    console.print(f"Model: {os.getenv('ORPILOT_MODEL', '(default)')}")
    console.print(f"Default Solver: {os.getenv('ORPILOT_DEFAULT_SOLVER', 'pulp')}")
    console.print(f"OpenAI Key: {'set' if os.getenv('OPENAI_API_KEY') else 'not set'}")
    console.print(f"Anthropic Key: {'set' if os.getenv('ANTHROPIC_API_KEY') else 'not set'}")


if __name__ == "__main__":
    app()
