"""Shared workflow state schema for LangGraph."""

from __future__ import annotations

from typing import Any, TypedDict

from orpilot.models.problem import ProblemDefinition
from orpilot.models.data import UserData
from orpilot.models.solution import SolutionResult


class WorkflowState(TypedDict, total=False):
    """State shared across all workflow nodes."""

    # Conversation history (list of {"role": ..., "content": ...})
    messages: list[dict[str, str]]

    # Problem definition extracted from interview
    problem: ProblemDefinition | None

    # User-provided data
    user_data: UserData | None

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
