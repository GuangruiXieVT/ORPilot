"""CLI entry point using Typer."""

from __future__ import annotations

import csv
import datetime
import json
import os
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv

# Load .env before Typer resolves envvar= options (e.g. OPENAI_API_KEY).
load_dotenv()
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from orpilot.config import discover_config_file, load_config_file
from orpilot.llm.config import LLMConfig, get_llm
from orpilot.paths import DATA_DIR
from orpilot.workflow.graph import build_graph
from orpilot.workflow.state import WorkflowState
from orpilot.models.problem import ProblemDefinition
from orpilot.models.data import UserData

app = typer.Typer(
    name="orpilot",
    help="AI Operations Research Agent — LLM-powered OR modeling and solving",
)
console = Console()

def _version_callback(value: bool) -> None:
    if value:
        from orpilot import __version__  # type: ignore[attr-defined]
        typer.echo(f"orpilot {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", "-V", callback=_version_callback, is_eager=True,
        help="Show the version and exit.",
    ),
) -> None:
    pass


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

_NODE_LABELS: dict[str, tuple[str, str]] = {
    "interview": ("blue", "Conducting interview..."),
    "data_collection": ("blue", "Collecting data..."),
    "param_computation": ("blue", "Computing derived parameters..."),
    "direct_code_gen": ("yellow", "Generating solver code..."),
    "solver_runner": ("yellow", "Starting model solving..."),
    "ir_builder_on_demand": ("dim", "Generating IR blueprint..."),
    "reporter": ("green", "Generating solution report..."),
}

_NODE_COMPLETE_LABELS: dict[str, tuple[str, str]] = {
    "interview": ("green", "Interview finished — problem defined."),
    "data_collection": ("green", "Data collection finished — all CSV files loaded."),
    "direct_code_gen": ("green", "Solver code generated."),
    "solver_runner": ("green", "Model solving finished."),
    "ir_builder_on_demand": ("green", "IR blueprint saved."),
}


def _log_entering_node(node: str) -> None:
    """Print a status line when entering a workflow node."""
    style, msg = _NODE_LABELS.get(node, ("dim", f"Running {node}..."))
    console.print(f"[{style}]>> {msg}[/{style}]")


# ---------------------------------------------------------------------------
# Artifact helpers
# ---------------------------------------------------------------------------


_POST_DATA_COLLECTION_NODES = {
    "param_computation",
    "direct_code_gen",
    "ir_builder_on_demand",
    "solver_runner",
    "reporter",
}


def _save_session(state: dict, session_path: Path) -> None:
    """Persist resumable conversation state to *session_path*."""
    problem = state.get("problem")
    payload = {
        "version": 1,
        "current_node": state.get("current_node", "interview"),
        "messages": state.get("messages", []),
        "messages_ctx": state.get("messages_ctx"),
        "problem": problem.model_dump() if problem is not None else None,
        "csv_specs": state.get("csv_specs", []),
        "data_dir": state.get("data_dir", ""),
    }
    session_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_session(session_path: Path, con: Console) -> dict:
    """Load a previously saved session from *session_path*.

    Returns a dict of WorkflowState overrides to apply on top of the freshly
    initialised state.  Also attempts to reload ``user_data`` from CSVs when
    the session was past the data-collection stage.
    """
    from orpilot.models.data import CsvFileSpec, UserData

    payload = json.loads(session_path.read_text(encoding="utf-8"))
    overrides: dict = {}

    overrides["messages"] = payload.get("messages", [])
    # Restore compressed context; fall back to full messages for old session files
    overrides["messages_ctx"] = payload.get("messages_ctx") or payload.get("messages", [])
    overrides["current_node"] = payload.get("current_node", "interview")
    overrides["csv_specs"] = payload.get("csv_specs", [])

    if payload.get("data_dir"):
        overrides["data_dir"] = payload["data_dir"]

    if payload.get("problem"):
        from orpilot.models.problem import ProblemDefinition
        overrides["problem"] = ProblemDefinition.model_validate(payload["problem"])

    # If the last message is from the assistant, the user hasn't replied yet —
    # mark needs_user_input so the CLI prompts before running the graph.
    messages = overrides.get("messages", [])
    if messages and messages[-1]["role"] == "assistant":
        overrides["needs_user_input"] = True

    # If CSV specs + files are available, reload user_data so we don't have to
    # re-confirm files — applies whether we're AT data_collection or past it.
    current_node = overrides["current_node"]
    csv_spec_dicts: list = overrides.get("csv_specs", [])
    data_dir: str = overrides.get("data_dir", "")
    if csv_spec_dicts and data_dir:
        specs = [CsvFileSpec.model_validate(d) for d in csv_spec_dicts]
        try:
            overrides["user_data"] = UserData.load_from_csv_dir(data_dir, specs)
            con.print("[dim]  -> Reloaded CSV data from disk.[/dim]")
        except (FileNotFoundError, ValueError) as exc:
            con.print(f"[yellow]  Warning: could not reload user data: {exc}[/yellow]")
            con.print("[yellow]  Resuming at data_collection to re-confirm files.[/yellow]")
            overrides["current_node"] = "data_collection"

    return overrides


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

    # Save data.json after param_computation has run (so computed derived tables are included).
    # Only save once param_computation is done, identified by the workflow having passed that node.
    # Overwrite on each call so the final data.json always reflects the fully computed user_data.
    if state.get("save_data"):
        user_data = state.get("user_data")
        current_node = state.get("current_node", "")
        past_param_computation = current_node not in ("interview", "data_collection")
        if user_data and past_param_computation:
            data_path = out / "data.json"
            data_path.write_text(json.dumps(user_data.as_dict()), encoding="utf-8")
            con.print(f"[dim]  -> Saved data.json to {data_path}[/dim]")

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

    _SEP = "\x1f"  # Unit Separator — used by IR compiler to delimit dimensions

    parsed: list[tuple[list[str], object]] = []
    max_dims = 0

    for var_name, value in sorted(variables.items()):
        if _SEP in var_name:
            # Unambiguous delimiter: "prefix\x1fdim1\x1fdim2\x1f..."
            parts = var_name.split(_SEP, 1)
            dims = parts[1].split(_SEP) if len(parts) > 1 else []
        else:
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
                # Legacy underscore-separated fallback (no underscores in IDs)
                parts = var_name.split("_")
                if len(parts) >= 2:
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
        # Fallback: no groups returned — auto-group by variable name prefix.
        # Keys use \x1f delimiter: "prefix\x1fdim1\x1fdim2\x1f..." (IR compiler output)
        # or legacy underscore-separated names. Group by prefix before first \x1f,
        # falling back to the first underscore-separated token.
        _SEP = "\x1f"
        groups: dict[str, dict[str, object]] = {}
        for var_name, value in solution.variables.items():
            if _SEP in var_name:
                prefix = var_name.split(_SEP, 1)[0]
            else:
                prefix = var_name.split("_", 1)[0]
            groups.setdefault(prefix, {})[var_name] = value

        for prefix, group_vars in groups.items():
            filename = f"solution_{prefix}.csv" if prefix else "solution_decisions.csv"
            csv_path = out / filename
            headers, rows = _parse_variable_dimensions(
                group_vars,
                group_name=prefix,
            )
            with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(headers)
                for row in rows:
                    writer.writerow(row)
            con.print(f"[dim]  -> Saved {prefix or 'decision'} solution values to {csv_path}[/dim]")


def _write_metrics(state: dict, output_dir: str, solver: str, con: Console) -> None:
    """Write metrics.json to output_dir with per-node token counts and latency."""
    metrics_state = state.get("metrics") or {}
    nodes: dict = metrics_state.get("nodes") or {}

    total_input = sum(n.get("input_tokens", 0) for n in nodes.values())
    total_output = sum(n.get("output_tokens", 0) for n in nodes.values())
    total_latency = round(sum(n.get("latency_s", 0.0) for n in nodes.values()), 2)
    total_retries = sum(max(n.get("retries", 0), 0) for n in nodes.values())

    solution = state.get("solution")
    solution_status = solution.status.value if solution else "unknown"
    objective_value = solution.objective_value if solution else None

    from orpilot.prompts._loader import all_versions
    payload = {
        "run_id": datetime.datetime.now().isoformat(timespec="seconds"),
        "solver": solver,
        "nodes": nodes,
        "totals": {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "latency_s": total_latency,
            "retries": total_retries,
        },
        "solution_status": solution_status,
        "objective_value": objective_value,
        "prompt_versions": all_versions(),
    }

    metrics_path = Path(output_dir) / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    con.print(f"[dim]  -> Saved metrics to {metrics_path}[/dim]")


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


@app.command()
def run(
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to a TOML or JSON config file. CLI options take precedence. Auto-discovered if orpilot.toml exists in the current directory."),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="LLM provider (openai, anthropic, google)"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model name override"),
    solver: Optional[str] = typer.Option(None, "--solver", "-s", help="OR solver backend: pulp, pyomo, ortools, gurobi, cplex"),
    problem_file: Optional[Path] = typer.Option(None, "--problem", help="Load problem definition from JSON file"),
    data_file: Optional[Path] = typer.Option(None, "--data", help="Load data from JSON file"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", "-d", help="Directory for CSV data files"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Directory to save generated code, LP file, and solution"),
    max_retries: Optional[int] = typer.Option(None, "--max-retries", help="Max solver code retries"),
    time_limit: Optional[int] = typer.Option(None, "--time-limit", "-t", help="Max solver run time in seconds. Stops early and returns best solution found."),
    show_solver_log: Optional[bool] = typer.Option(None, "--solver-log/--no-solver-log", help="Stream the solver log to stdout."),
    verbose: Optional[bool] = typer.Option(None, "--verbose/--no-verbose", help="Show full solver and compiler error details"),
    generate_ir: Optional[bool] = typer.Option(None, "--generate-ir/--no-generate-ir", help="After a successful solve, generate an IR blueprint for solver portability"),
    api_key: Optional[str] = typer.Option(None, "--api-key", envvar="OPENAI_API_KEY", help="API key. For Gemini, set GOOGLE_API_KEY in env instead."),
    base_url: Optional[str] = typer.Option(None, "--base-url", envvar="OPENAI_BASE_URL", help="Custom API base URL (e.g. https://api.deepseek.com)"),
    temperature: Optional[float] = typer.Option(None, "--temperature", help="LLM sampling temperature (0.0 = deterministic)"),
    session_file: Optional[Path] = typer.Option(None, "--session", help="Path to session file for save/resume. Defaults to session.json inside --output-dir if set."),
    no_resume: bool = typer.Option(False, "--no-resume", help="Do not load an existing session file; always start fresh."),
    save_data: Optional[bool] = typer.Option(None, "--save-data/--no-save-data", help="Save data.json to output-dir for portability (run model.py on another machine)."),
) -> None:
    """Start an interactive ORPilot session."""
    # --- Load config file ---
    # Explicit --config takes priority; otherwise auto-discover orpilot.toml/json.
    cfg: dict = {}
    resolved_config = config_file
    if resolved_config is None:
        resolved_config = discover_config_file()
    if resolved_config is not None:
        if not resolved_config.exists():
            console.print(f"[red]Config file not found: {resolved_config}[/red]")
            raise typer.Exit(1)
        cfg = load_config_file(resolved_config)
        console.print(f"[dim]Loaded config from {resolved_config}[/dim]")

    # --- Merge: CLI values > config file values > hardcoded defaults ---
    provider        = provider        or cfg.get("provider")        or os.environ.get("ORPILOT_LLM_PROVIDER", "openai")
    model           = model           or cfg.get("model")           or os.environ.get("ORPILOT_MODEL")
    solver          = solver          or cfg.get("solver")          or os.environ.get("ORPILOT_DEFAULT_SOLVER", "pulp")
    max_retries     = max_retries     if max_retries     is not None else cfg.get("max_retries",     3)
    time_limit      = time_limit      if time_limit      is not None else cfg.get("time_limit",      300)
    verbose         = verbose         if verbose         is not None else cfg.get("verbose",         False)
    show_solver_log = show_solver_log if show_solver_log is not None else cfg.get("show_solver_log", False)
    generate_ir     = generate_ir     if generate_ir     is not None else cfg.get("generate_ir",     False)
    save_data       = save_data       if save_data       is not None else cfg.get("save_data",       False)
    temperature     = temperature     if temperature     is not None else cfg.get("temperature",     0.0)
    base_url        = base_url        or cfg.get("base_url")
    api_key         = api_key         or cfg.get("api_key")
    max_tokens      = int(cfg.get("max_tokens", 8192))

    # Resolve path options: relative paths from a config file are relative to
    # the config file's directory; CLI-supplied paths are relative to CWD.
    cfg_dir = resolved_config.parent if resolved_config is not None else Path.cwd()
    if data_dir is None:
        data_dir = (cfg_dir / cfg["data_dir"]).resolve() if "data_dir" in cfg else DATA_DIR
    else:
        data_dir = data_dir.resolve()
    if output_dir is None and "output_dir" in cfg:
        output_dir = (cfg_dir / cfg["output_dir"]).resolve()
    elif output_dir is not None:
        output_dir = output_dir.resolve()

    llm_config = LLMConfig(provider=provider, model=model, api_key=api_key, base_url=base_url, temperature=temperature, max_tokens=max_tokens)
    llm = get_llm(llm_config)
    graph = build_graph(llm=llm)

    # Ensure data directory exists
    data_dir.mkdir(parents=True, exist_ok=True)

    # Ensure output directory exists if specified
    output_dir_str = ""
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dir_str = str(output_dir)

    # Resolve session file path: explicit --session > output_dir/session.json > ./session.json
    resolved_session: Path | None = None
    if session_file is not None:
        resolved_session = session_file.resolve()
    elif output_dir is not None:
        resolved_session = output_dir / "session.json"
    else:
        resolved_session = Path.cwd() / "session.json"

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
        "solver_time_limit": time_limit,
        "show_solver_log": show_solver_log,
        "generate_ir": generate_ir,
        "save_data": save_data,
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
            state["current_node"] = "direct_code_gen"
        console.print(Panel("Loaded data from file", title="Data"))

    # Resume from session file if it exists (and --no-resume was not given).
    # Session overrides take lower priority than explicit --problem/--data flags.
    if (
        resolved_session is not None
        and not no_resume
        and resolved_session.exists()
        and state.get("problem") is None  # don't clobber explicit --problem
    ):
        console.print(f"[dim]Resuming session from {resolved_session}[/dim]")
        session_overrides = _load_session(resolved_session, console)
        state = {**state, **session_overrides}
        node_label = state.get("current_node", "interview")
        console.print(f"[green]>> Resumed at node: {node_label}[/green]")

        # Replay conversation history so the user sees what was discussed
        prior_messages = state.get("messages", [])
        if prior_messages:
            console.print()
            console.rule("[dim]Previous conversation[/dim]")
            for msg in prior_messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "assistant":
                    console.print(Markdown(content))
                else:
                    console.print(f"[bold blue]You:[/bold blue] {content}")
                console.print()
            console.rule("[dim]Resuming here[/dim]")
            console.print()

    console.print(Panel(
        "Welcome to ORPilot — AI Operations Research Agent\n"
        "I'll help you model and solve optimization problems.\n"
        "Type 'quit' to exit at any time.",
        title="ORPilot",
        border_style="blue",
    ))

    _last_saved_code = ""

    while True:
        # If we're waiting for user input (e.g. on resume), collect it before
        # running the graph — otherwise we'd feed an assistant-ending conversation
        # back to the LLM immediately.
        if state.get("needs_user_input"):
            messages = state.get("messages", [])
            if messages and messages[-1]["role"] == "assistant":
                console.print()
                console.print(Markdown(messages[-1]["content"]))
                console.print()

            user_input = console.input("[bold blue]You:[/bold blue] ")
            if user_input.strip().lower() in ("quit", "exit", "q"):
                console.print("Goodbye!")
                raise typer.Exit()

            user_msg = {"role": "user", "content": user_input}
            state["messages"].append(user_msg)
            if state.get("messages_ctx") is None:
                state["messages_ctx"] = list(state["messages"])
            else:
                state["messages_ctx"].append(user_msg)
            state["needs_user_input"] = False

            if resolved_session is not None:
                _save_session(state, resolved_session)

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

                elif node_name == "direct_code_gen":
                    if state.get("generated_code") and not state.get("error_context"):
                        style, msg = _NODE_COMPLETE_LABELS["direct_code_gen"]
                        console.print(f"[{style}]✓ {msg}[/{style}]")

                elif node_name == "ir_builder_on_demand":
                    if state.get("ir_model"):
                        style, msg = _NODE_COMPLETE_LABELS["ir_builder_on_demand"]
                        console.print(f"[{style}]✓ {msg}[/{style}]")
                    else:
                        console.print("[dim]  (IR generation skipped or failed — solve result unaffected)[/dim]")

                elif node_name == "solver_runner":
                    solution = state.get("solution")
                    if solution:
                        if solution.error_message:
                            retry = state.get("retry_count", 0)
                            max_r = state.get("max_retries", 3)
                            if verbose:
                                detail = solution.error_message
                                if solution.solver_output:
                                    detail += f"\n\nSolver output:\n{solution.solver_output}"
                                console.print(Panel(
                                    detail,
                                    title=f"[red]✗ Solver error (attempt {retry}/{max_r})[/red]",
                                    border_style="red",
                                ))
                            else:
                                console.print(
                                    f"[red]✗ Solver error (attempt {retry}/{max_r}): "
                                    f"{solution.error_message[:200]}[/red]"
                                )
                                console.print("[dim]   (use --verbose / -v to see full details)[/dim]")
                            console.print("[yellow]   Retrying with error feedback...[/yellow]")
                        else:
                            style, msg = _NODE_COMPLETE_LABELS["solver_runner"]
                            console.print(f"[{style}]✓ {msg}[/{style}]")
                            # Save solution CSVs immediately — don't wait for reporter,
                            # which may fail independently (e.g. 413 on large solutions).
                            if output_dir_str and solution.status.value in ("optimal", "feasible"):
                                _save_solution(state, output_dir_str, console)

                # Save debug artifacts after each node so OOM during solve doesn't lose model.py
                if output_dir_str:
                    _last_saved_code = _save_artifacts(state, output_dir_str, _last_saved_code, console)

                # Persist session after every node — ensures problem/csv_specs are not
                # lost if the process is interrupted between nodes (e.g. after the
                # interview extracts the problem but before data_collection completes).
                if resolved_session is not None:
                    _save_session(state, resolved_session)

        # Save debug artifacts when output_dir is set
        if output_dir_str:
            _last_saved_code = _save_artifacts(state, output_dir_str, _last_saved_code, console)

        # Final persist after the full graph iteration
        if resolved_session is not None:
            _save_session(state, resolved_session)

        # Check if we have a final report
        if state.get("report"):

            # Keep session file so the conversation history is preserved for reference

            if output_dir_str:
                _write_metrics(state, output_dir_str, solver, console)

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

            user_msg = {"role": "user", "content": user_input}
            state["messages"].append(user_msg)
            if state.get("messages_ctx") is None:
                state["messages_ctx"] = list(state["messages"])
            else:
                state["messages_ctx"].append(user_msg)
            state["needs_user_input"] = False

            # Persist after user input so the new message is captured
            if resolved_session is not None:
                _save_session(state, resolved_session)


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


@app.command()
def solve(
    problem_file: Path = typer.Argument(..., help="Path to a plain-text file containing the problem description and/or embedded data."),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to a TOML or JSON config file. Auto-discovered if orpilot.toml exists in the current directory."),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="LLM provider (openai, anthropic, google)"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model name override"),
    solver: Optional[str] = typer.Option(None, "--solver", "-s", help="OR solver backend: pulp, pyomo, ortools, gurobi, cplex"),
    mode: Optional[str] = typer.Option(None, "--mode", help="direct (TextIngestor + code gen) or ir (TextIngestor + IR builder). Default: direct"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Directory to save generated code, LP file, and solution"),
    max_retries: Optional[int] = typer.Option(None, "--max-retries", help="Max solver code retries"),
    time_limit: Optional[int] = typer.Option(None, "--time-limit", "-t", help="Max solver run time in seconds"),
    verbose: Optional[bool] = typer.Option(None, "--verbose/--no-verbose", help="Show full error details"),
    generate_ir: Optional[bool] = typer.Option(None, "--generate-ir/--no-generate-ir", help="After a successful solve, generate an IR blueprint"),
    api_key: Optional[str] = typer.Option(None, "--api-key", envvar="OPENAI_API_KEY", help="API key. For Gemini, set GOOGLE_API_KEY in env instead."),
    base_url: Optional[str] = typer.Option(None, "--base-url", envvar="OPENAI_BASE_URL", help="Custom API base URL"),
    temperature: Optional[float] = typer.Option(None, "--temperature", help="LLM sampling temperature (0.0 = deterministic)"),
) -> None:
    """Solve an OR problem described in a plain-text file.

    The file is passed through the TextIngestor (which extracts the problem
    structure and any embedded data tables via LLM), then the resulting model
    is solved via direct code generation or the IR builder pipeline.

    Example:

        orpilot solve problem.txt --solver pulp --output-dir output/
    """
    from orpilot.benchmark.case import BenchmarkCase
    from orpilot.benchmark.runner import BenchmarkRunner

    if not problem_file.exists():
        console.print(f"[red]File not found: {problem_file}[/red]")
        raise typer.Exit(1)

    # --- Load config file ---
    cfg: dict = {}
    resolved_config = config_file
    if resolved_config is None:
        resolved_config = discover_config_file()
    if resolved_config is not None:
        if not resolved_config.exists():
            console.print(f"[red]Config file not found: {resolved_config}[/red]")
            raise typer.Exit(1)
        cfg = load_config_file(resolved_config)
        console.print(f"[dim]Loaded config from {resolved_config}[/dim]")

    # Merge: CLI > config file > env > defaults
    provider    = provider    or cfg.get("provider")    or os.environ.get("ORPILOT_LLM_PROVIDER", "openai")
    model       = model       or cfg.get("model")       or os.environ.get("ORPILOT_MODEL")
    solver      = solver      or cfg.get("solver")      or os.environ.get("ORPILOT_DEFAULT_SOLVER", "pulp")
    mode        = mode        or cfg.get("mode",        "direct")
    max_retries = max_retries if max_retries is not None else cfg.get("max_retries", 3)
    time_limit  = time_limit  if time_limit  is not None else cfg.get("time_limit",  300)
    verbose     = verbose     if verbose     is not None else cfg.get("verbose",     False)
    generate_ir = generate_ir if generate_ir is not None else cfg.get("generate_ir", False)
    temperature = temperature if temperature is not None else cfg.get("temperature", 0.0)
    base_url    = base_url    or cfg.get("base_url")
    api_key     = api_key     or cfg.get("api_key")
    max_tokens  = int(cfg.get("max_tokens", 8192))

    cfg_dir = resolved_config.parent if resolved_config is not None else Path.cwd()
    if output_dir is None and "output_dir" in cfg:
        output_dir = (cfg_dir / cfg["output_dir"]).resolve()
    elif output_dir is not None:
        output_dir = output_dir.resolve()

    if mode not in ("direct", "ir"):
        console.print(f"[red]Unknown mode '{mode}'. Choose 'direct' or 'ir'.[/red]")
        raise typer.Exit(1)

    problem_text = problem_file.read_text(encoding="utf-8").strip()

    llm_config = LLMConfig(provider=provider, model=model, api_key=api_key, base_url=base_url, temperature=temperature, max_tokens=max_tokens)
    llm = get_llm(llm_config)

    console.print(Panel(
        f"[bold]{problem_file.name}[/bold]\n"
        f"Mode: {mode}  |  Solver: {solver}  |  Model: {model or '(default)'}",
        title="ORPilot — Solve from file",
        border_style="blue",
    ))

    case = BenchmarkCase(name=problem_file.stem, problem_text=problem_text)
    runner = BenchmarkRunner(timeout=time_limit)

    console.print("[blue]>> Ingesting problem text...[/blue]")
    if mode == "ir":
        result = runner.run_full_pipeline(case, llm, solver=solver)
    else:
        result = runner.run_direct_pipeline(case, llm, solver=solver, generate_ir=generate_ir)

    # --- Display result ---
    console.print()
    if result.status in ("optimal", "feasible"):
        console.print(f"[green bold]✓ {result.status.upper()}[/green bold]")
    else:
        console.print(f"[red bold]✗ {result.status.upper()}[/red bold]")

    if result.objective_value is not None:
        console.print(f"  Objective Value : {result.objective_value:,.4f}")
    if result.solve_time is not None:
        console.print(f"  Solve Time      : {result.solve_time:.2f}s")
    if result.error:
        if verbose:
            console.print(Panel(result.error, title="[red]Error[/red]", border_style="red"))
        else:
            console.print(f"[red]  Error: {result.error[:300]}[/red]")
            console.print("[dim]  (use --verbose to see full details)[/dim]")

    # --- Save artifacts ---
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        if result.generated_code:
            code_path = output_dir / "model.py"
            code_path.write_text(result.generated_code, encoding="utf-8")
            console.print(f"[dim]  -> Saved generated code to {code_path}[/dim]")

        if result.ir_model:
            import json as _json
            ir_path = output_dir / "ir.json"
            ir_path.write_text(_json.dumps(result.ir_model, indent=2), encoding="utf-8")
            console.print(f"[dim]  -> Saved IR to {ir_path}[/dim]")

        if result.tables:
            import csv as _csv
            data_dir = output_dir / "data"
            data_dir.mkdir(exist_ok=True)
            for stem, rows in result.tables.items():
                if not rows:
                    continue
                csv_path = data_dir / f"{stem}.csv"
                with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                    writer = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)
            console.print(f"[dim]  -> Saved data CSVs to {data_dir}[/dim]")

        if result.lp_content:
            lp_path = output_dir / "model.lp"
            lp_path.write_text(result.lp_content, encoding="utf-8")
            console.print(f"[dim]  -> Saved LP file to {lp_path}[/dim]")

    raise typer.Exit(0 if result.status in ("optimal", "feasible") else 1)


@app.command(name="generate-ir")
def generate_ir_cmd(
    output_dir: Path = typer.Argument(..., help="Output folder containing session.json and model.py"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to a TOML or JSON config file. Auto-discovered if orpilot.toml exists in the current directory."),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="LLM provider (openai, anthropic, google)"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model name override"),
    api_key: Optional[str] = typer.Option(None, "--api-key", envvar="OPENAI_API_KEY", help="API key. For Gemini, set GOOGLE_API_KEY in env instead."),
    base_url: Optional[str] = typer.Option(None, "--base-url", envvar="OPENAI_BASE_URL", help="Custom API base URL"),
    temperature: Optional[float] = typer.Option(None, "--temperature", help="LLM sampling temperature (0.0 = deterministic)"),
) -> None:
    """Generate an IR blueprint from an existing output folder.

    Reads session.json (for the problem definition and CSV schemas) and model.py
    (for the working solver code) from the output folder, then calls the
    ir_builder_on_demand node to produce ir.json in the same folder.

    Example:

        orpilot generate-ir output/
    """
    from orpilot.workflow.nodes.ir_builder import ir_builder_on_demand_node

    output_dir = output_dir.resolve()
    session_path = output_dir / "session.json"
    model_path = output_dir / "model.py"

    if not session_path.exists():
        console.print(f"[red]session.json not found in {output_dir}[/red]")
        raise typer.Exit(1)
    if not model_path.exists():
        console.print(f"[red]model.py not found in {output_dir}[/red]")
        raise typer.Exit(1)

    # --- Load config ---
    cfg: dict = {}
    resolved_config = config_file
    if resolved_config is None:
        resolved_config = discover_config_file()
    if resolved_config is not None:
        if not resolved_config.exists():
            console.print(f"[red]Config file not found: {resolved_config}[/red]")
            raise typer.Exit(1)
        cfg = load_config_file(resolved_config)
        console.print(f"[dim]Loaded config from {resolved_config}[/dim]")

    provider    = provider    or cfg.get("provider")    or os.environ.get("ORPILOT_LLM_PROVIDER", "openai")
    model       = model       or cfg.get("model")       or os.environ.get("ORPILOT_MODEL")
    temperature = temperature if temperature is not None else cfg.get("temperature", 0.0)
    base_url    = base_url    or cfg.get("base_url")
    api_key     = api_key     or cfg.get("api_key")
    max_tokens  = int(cfg.get("max_tokens", 8192))

    # --- Load session ---
    console.print(f"[dim]Loading session from {session_path}[/dim]")

    import json as _sess_json
    session_payload = _sess_json.loads(session_path.read_text(encoding="utf-8"))

    from orpilot.models.data import CsvFileSpec, UserData
    from orpilot.models.problem import ProblemDefinition

    problem = None
    if session_payload.get("problem"):
        problem = ProblemDefinition.model_validate(session_payload["problem"])
    if problem is None:
        console.print("[red]session.json does not contain a problem definition.[/red]")
        raise typer.Exit(1)

    # Build user_data: load CSV specs from session AND actual row data from data_dir
    # (if the CSVs are still present on disk).  Row data populates distinct_values in
    # csv_schemas so the LLM can see actual member IDs — critical for shared-source
    # set files (e.g. sets.csv with a set_name category column) so the LLM chooses
    # MEMBER LIST FROM CSV rather than HARDCODED COUNT for time sets like Periods.
    csv_spec_dicts = session_payload.get("csv_specs", [])
    specs = [CsvFileSpec.model_validate(d) for d in csv_spec_dicts]
    data_dir_str = session_payload.get("data_dir", "")
    data_dir_path = Path(data_dir_str) if data_dir_str else None
    if specs and data_dir_path and data_dir_path.is_dir():
        try:
            user_data = UserData.load_from_csv_dir(str(data_dir_path), specs)
            console.print(
                f"[dim]  -> Loaded {len(specs)} CSV schema(s) + row data from {data_dir_path}[/dim]"
            )
        except Exception:
            # CSV files may have moved since the session — fall back to schema-only
            user_data = UserData(csv_specs=specs)
            console.print(
                f"[dim]  -> Loaded {len(specs)} CSV schema(s) from session "
                f"(row data unavailable — CSVs not found in {data_dir_path})[/dim]"
            )
    elif specs:
        user_data = UserData(csv_specs=specs)
        console.print(f"[dim]  -> Loaded {len(specs)} CSV schema(s) from session (no data_dir).[/dim]")
    else:
        user_data = None

    # Augment user_data with any CSV files in data_dir that are not already in csv_specs.
    # param_computation writes files (e.g. bigM.csv) after the session specs were recorded,
    # so they are absent from session.json but present on disk.
    if data_dir_path and data_dir_path.is_dir():
        import csv as _csv2
        from orpilot.models.data import CsvColumnSpec, CsvFileSpec
        known_stems = {Path(s.filename).stem for s in (user_data.csv_specs if user_data else [])}
        extra_specs: list[CsvFileSpec] = []
        extra_tables: dict = {}
        for csv_file in sorted(data_dir_path.glob("*.csv")):
            stem = csv_file.stem
            if stem in known_stems:
                continue
            try:
                with open(csv_file, newline="", encoding="utf-8") as _fh:
                    reader = _csv2.DictReader(_fh)
                    rows = list(reader)
                    headers = list(reader.fieldnames or (rows[0].keys() if rows else []))
                columns = [CsvColumnSpec(name=h, dtype="str") for h in headers]
                extra_specs.append(CsvFileSpec(filename=csv_file.name, columns=columns))
                extra_tables[stem] = rows
                console.print(f"[dim]  -> Found extra CSV (from param_computation): {csv_file.name}[/dim]")
            except Exception:
                pass
        if extra_specs:
            if user_data is None:
                user_data = UserData(csv_specs=extra_specs, raw_tables=extra_tables)
            else:
                user_data = UserData(
                    csv_specs=list(user_data.csv_specs) + extra_specs,
                    raw_tables={**user_data.raw_tables, **extra_tables},
                )

    overrides = {
        "messages": session_payload.get("messages", []),
        "current_node": session_payload.get("current_node", "interview"),
        "csv_specs": csv_spec_dicts,
        "data_dir": session_payload.get("data_dir", ""),
        "problem": problem,
        "user_data": user_data,
    }

    generated_code = model_path.read_text(encoding="utf-8")

    # --- Build state ---
    state: WorkflowState = {
        "problem": problem,
        "user_data": user_data,
        "generated_code": generated_code,
        "messages": overrides.get("messages", []),
        "current_node": "ir_builder_on_demand",
        "needs_user_input": False,
        "ir_model": None,
        "error_context": "",
        "report": "",
        "solution": None,
        "data_dir": overrides.get("data_dir", ""),
        "csv_specs": overrides.get("csv_specs", []),
    }

    # --- Run IR generation ---
    console.print(Panel(
        f"Problem: [bold]{problem.title}[/bold]\n"
        f"Model:   {model or '(default)'}  |  Provider: {provider}",
        title="ORPilot — Generate IR",
        border_style="blue",
    ))
    console.print("[blue]>> Generating IR blueprint...[/blue]")

    llm_config = LLMConfig(
        provider=provider, model=model, api_key=api_key,
        base_url=base_url, temperature=temperature, max_tokens=max_tokens,
    )
    llm = get_llm(llm_config)

    result_state = ir_builder_on_demand_node(state, llm)

    ir_model = result_state.get("ir_model")
    if ir_model:
        ir_path = output_dir / "ir.json"
        ir_path.write_text(json.dumps(ir_model, indent=2), encoding="utf-8")
        console.print(f"[green bold]✓ IR blueprint saved to {ir_path}[/green bold]")
        raise typer.Exit(0)
    else:
        console.print("[red]✗ IR generation failed after 3 attempts.[/red]")
        raise typer.Exit(1)


def _collect_csv_sources(ir: dict, data_dir: Path) -> dict[str, tuple[str, bool]]:
    """Return {stem: (filename, optional)} for all CSV files the generated solve() needs.

    Tries IR ``source`` fields first; falls back to every ``*.csv`` in
    *data_dir* so that IRs generated without explicit source annotations still
    work.  Source values without a file extension are assumed to be ``*.csv``.
    Optional parameters (IR field ``optional: true``) may be absent on disk.
    """
    seen: dict[str, tuple[str, bool]] = {}
    for meta in list(ir.get("sets", {}).values()) + list(ir.get("parameters", {}).values()):
        source = meta.get("source")
        if source:
            p = Path(source)
            stem = p.stem
            fname = p.name if p.suffix else p.name + ".csv"
            is_optional = bool(meta.get("optional", False))
            seen[stem] = (fname, is_optional)
    if not seen:
        # No source annotations in IR — load every CSV present in data_dir
        for csv_file in sorted(data_dir.glob("*.csv")):
            seen[csv_file.stem] = (csv_file.name, False)
    return seen


@app.command(name="compile-ir")
def compile_ir_cmd(
    ir_path: Path = typer.Argument(..., help="Path to ir.json (file or directory containing ir.json)"),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Where to write model.py. Defaults to model.py next to ir.json."),
    solver: Optional[str] = typer.Option(None, "--solver", "-s", help="Solver backend: pulp, pyomo, ortools, gurobi, cplex. Reads from config if omitted."),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", "-d", help="Directory containing CSV input files. Required with --run."),
    run: bool = typer.Option(False, "--run", "-r", help="Solve and save results after compiling. Requires --data-dir. Results (CSVs, optimization_summary.txt, report.md) are saved to the same folder as --out."),
    show_solver_log: Optional[bool] = typer.Option(None, "--solver-log/--no-solver-log", help="Stream the solver log to stdout. Only used with --run."),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="LLM provider for the reporter (openai, anthropic, google). Only used with --run."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model name override for the reporter. Only used with --run."),
    api_key: Optional[str] = typer.Option(None, "--api-key", envvar="OPENAI_API_KEY", help="API key. For Gemini, set GOOGLE_API_KEY in env instead."),
    base_url: Optional[str] = typer.Option(None, "--base-url", envvar="OPENAI_BASE_URL"),
    temperature: Optional[float] = typer.Option(None, "--temperature", help="LLM sampling temperature. Only used with --run."),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to a TOML or JSON config file. Auto-discovered if orpilot.toml exists in the current directory."),
) -> None:
    """Compile an ir.json blueprint into solver-specific Python code (model.py).

    The IR compiler is deterministic — no LLM is involved. The generated
    model.py defines a single solve(data) function that returns a result dict.

    Add --data-dir and --run to execute the full downstream pipeline after
    compiling: solve → save CSVs + optimization_summary.txt → reporter →
    report.md. This mirrors the same process as orpilot run.

    Examples:

        orpilot compile-ir output/ir.json
        orpilot compile-ir output/                         # picks up output/ir.json
        orpilot compile-ir output/ir.json --solver gurobi
        orpilot compile-ir ir.json --data-dir data/ --out data_copy/model.py --run
        orpilot compile-ir path/to/ir.json --out path/to/model.py --solver cplex
    """
    from orpilot.codegen.ir_compiler import IRCompiler

    # Accept a directory → look for ir.json inside it
    ir_path = ir_path.resolve()
    if ir_path.is_dir():
        ir_path = ir_path / "ir.json"

    if not ir_path.exists():
        console.print(f"[red]ir.json not found: {ir_path}[/red]")
        raise typer.Exit(1)

    # Resolve output path: explicit --out, or model.py next to ir.json
    out_path = out.resolve() if out else ir_path.parent / "model.py"

    # Load config for solver default
    cfg: dict = {}
    resolved_config = config_file
    if resolved_config is None:
        resolved_config = discover_config_file()
    if resolved_config is not None and resolved_config.exists():
        cfg = load_config_file(resolved_config)
        console.print(f"[dim]Loaded config from {resolved_config}[/dim]")

    solver = solver or cfg.get("solver") or os.environ.get("ORPILOT_DEFAULT_SOLVER", "pulp")
    show_solver_log = show_solver_log if show_solver_log is not None else cfg.get("show_solver_log", False)

    # Load IR
    import json as _json
    try:
        ir_model = _json.loads(ir_path.read_text(encoding="utf-8"))
    except Exception as exc:
        console.print(f"[red]Failed to read {ir_path}: {exc}[/red]")
        raise typer.Exit(1)

    console.print(f"[dim]IR:       {ir_path}[/dim]")
    console.print(f"[dim]Output:   {out_path}[/dim]")
    console.print(f"[dim]Solver:   {solver}[/dim]")

    # Compile
    try:
        code = IRCompiler().compile(ir_model, solver)
    except Exception as exc:
        console.print(f"[red]Compilation failed: {exc}[/red]")
        raise typer.Exit(1)

    if out_path.parent.exists() and not out_path.parent.is_dir():
        console.print(f"[red]Output path conflict: {out_path.parent} exists as a file, not a directory. Remove it and retry.[/red]")
        raise typer.Exit(1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(code, encoding="utf-8")
    console.print(f"[green bold]✓ model.py written to {out_path}[/green bold]")

    if run:
        if data_dir is None:
            console.print("[red]--run requires --data-dir.[/red]")
            raise typer.Exit(1)

        from orpilot.solver.registry import get_solver
        from orpilot.models.problem import ProblemDefinition
        from orpilot.models.solution import SolveStatus
        from orpilot.workflow.nodes.reporter import reporter_node
        from rich.markdown import Markdown

        results_dir = out_path.parent
        results_dir.mkdir(parents=True, exist_ok=True)

        # --- Load CSV data ---
        console.print("[blue]>> Loading data...[/blue]")
        import csv as _csv
        csv_sources = _collect_csv_sources(ir_model, data_dir.resolve())
        if not csv_sources:
            console.print("[red]No CSV files found in data-dir and no source fields in IR.[/red]")
            raise typer.Exit(1)
        data_dict: dict = {}
        for stem, (fname, is_optional) in csv_sources.items():
            fpath = data_dir.resolve() / fname
            if not fpath.exists():
                if is_optional:
                    data_dict[stem] = []
                    console.print(f"[dim]  Skipped optional {fname} (not found)[/dim]")
                    continue
                console.print(f"[red]Data file not found: {fpath}[/red]")
                raise typer.Exit(1)
            with open(fpath, newline="", encoding="utf-8") as _f:
                data_dict[stem] = list(_csv.DictReader(_f))
            console.print(f"[dim]  Loaded {fname} ({len(data_dict[stem])} rows)[/dim]")

        import json as _json
        data_json_path = results_dir / "data.json"
        data_json_path.write_text(_json.dumps(data_dict, indent=2), encoding="utf-8")
        console.print(f"[dim]  Saved data.json to {data_json_path}[/dim]")

        # --- Solve ---
        console.print(f"[blue]>> Solving ({solver})...[/blue]")
        solution = get_solver(solver).solve(code, data_dict, show_solver_log=show_solver_log)
        console.print(f"  Status: {solution.status.value}")
        if solution.objective_value is not None:
            console.print(f"  Objective: {solution.objective_value}")
        if solution.status.value == "error":
            if solution.error_message:
                console.print(f"[red]  Error: {solution.error_message}[/red]")
            if solution.solver_output:
                console.print(f"[dim]  Solver output:\n{solution.solver_output}[/dim]")

        # --- Save solution CSVs + summary ---
        state_for_save = {"solution": solution}
        _save_solution(state_for_save, str(results_dir), console)

        # --- Reporter ---
        provider_ = provider or cfg.get("provider") or os.environ.get("ORPILOT_LLM_PROVIDER", "openai")
        model_ = model or cfg.get("model") or os.environ.get("ORPILOT_MODEL")
        temperature_ = temperature if temperature is not None else cfg.get("temperature", 0.0)
        base_url_ = base_url or cfg.get("base_url")
        api_key_ = api_key or cfg.get("api_key")
        max_tokens = int(cfg.get("max_tokens", 8192))

        problem_class = ir_model.get("problem_class", "Optimization Model")
        problem = ProblemDefinition(title=problem_class, description=problem_class)
        reporter_state: WorkflowState = {
            "problem": problem,
            "solution": solution,
            "generated_code": code,
            "messages": [],
            "current_node": "reporter",
            "needs_user_input": False,
            "ir_model": ir_model,
            "error_context": "",
            "report": "",
            "data_dir": str(data_dir.resolve()),
            "csv_specs": [],
            "user_data": None,
        }

        console.print("[blue]>> Generating report...[/blue]")
        try:
            llm_config = LLMConfig(
                provider=provider_, model=model_, api_key=api_key_,
                base_url=base_url_, temperature=temperature_, max_tokens=max_tokens,
            )
            llm = get_llm(llm_config)
            reporter_state = reporter_node(reporter_state, llm)
            report = reporter_state.get("report", "")
        except Exception as exc:
            console.print(f"[yellow]Warning: reporter failed ({exc}). Skipping report.[/yellow]")
            report = ""

        if report:
            report_path = results_dir / "report.md"
            report_path.write_text(report, encoding="utf-8")
            console.print(f"[dim]  -> Saved report to {report_path}[/dim]")
            console.print()
            console.print(Panel(Markdown(report), title="Solution Report", border_style="green"))


if __name__ == "__main__":
    app()
