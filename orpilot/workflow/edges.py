"""Conditional edge logic for routing between workflow nodes."""

from __future__ import annotations

from orpilot.models.solution import SolveStatus
from orpilot.workflow.state import WorkflowState


def after_interview(state: WorkflowState) -> str:
    """Route after interview node."""
    if state.get("problem") is not None:
        return "data_collection"
    return "wait_for_input"


def after_data_collection(state: WorkflowState) -> str:
    """Route after data collection node."""
    if state.get("user_data") is not None:
        return "param_computation"
    return "wait_for_input"


def after_direct_code_gen(state: WorkflowState) -> str:
    """Route after direct code gen: solver_runner on success, reporter on hard failure."""
    if state.get("current_node") == "reporter":
        return "reporter"
    return "solver_runner"


def after_ir_builder(state: WorkflowState) -> str:
    """Route after IR builder: compiler on success, reporter on failure/unsupported."""
    if state.get("current_node") == "reporter":
        return "reporter"
    return "ir_compiler"


def after_ir_compiler(state: WorkflowState) -> str:
    """Route after IR compiler: solver on success, retry ir_builder or reporter on failure."""
    if state.get("error_context"):
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 3)
        if retry_count < max_retries:
            return "ir_builder"
        return "reporter"
    return "solver_runner"


def after_solver_runner(state: WorkflowState) -> str:
    """Route after solver execution."""
    solution = state.get("solution")
    if solution is None:
        return "ir_builder"

    if solution.status in (SolveStatus.OPTIMAL, SolveStatus.FEASIBLE):
        if state.get("generate_ir", False):
            return "ir_builder_on_demand"
        return "reporter"

    # Failed — retry via ir_builder if budget allows
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)
    if retry_count < max_retries:
        return "ir_builder"

    return "reporter"
