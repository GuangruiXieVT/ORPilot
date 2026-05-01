"""Shared workflow state schema for LangGraph."""

from __future__ import annotations

from typing import Any, TypedDict

from orpilot.models.problem import ProblemDefinition
from orpilot.models.data import UserData
from orpilot.models.solution import SolutionResult


class WorkflowState(TypedDict, total=False):
    """State shared across all workflow nodes."""

    # Full verbatim conversation history — displayed on resume, saved to session.json
    messages: list[dict[str, str]]
    # Compressed context sent to the LLM — replaced with a summary at each phase
    # boundary (after interview, after data collection) to keep token usage bounded.
    # Falls back to `messages` when not set.
    messages_ctx: list[dict[str, str]]

    # Problem definition extracted from interview
    problem: ProblemDefinition | None

    # User-provided data
    user_data: UserData | None

    # JSON IR produced by ir_builder
    ir_model: dict | None

    # Generated solver code
    generated_code: str

    # Solution result
    solution: SolutionResult | None

    # Natural language report
    report: str

    # Workflow control
    current_node: str
    solver_name: str
    retry_count: int
    max_retries: int
    error_context: str

    # Flags for user interaction
    needs_user_input: bool
    user_input: str

    # LLM config
    llm_config: dict[str, Any]

    # CSV data collection
    data_dir: str
    csv_specs: list

    # Debug output
    output_dir: str

    # Solver time limit in seconds (None = no limit)
    solver_time_limit: int | None

    # Whether to stream the solver log to stdout
    show_solver_log: bool

    # When True: generate IR on-demand after a successful solve (for solver portability)
    generate_ir: bool

    # When True: save data.json to output_dir for portability (run model on another machine)
    save_data: bool

    # Observability: per-node token counts and latency (accumulated by graph.py instrumentation)
    metrics: dict
