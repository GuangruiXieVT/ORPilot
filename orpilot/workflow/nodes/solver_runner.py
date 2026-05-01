"""Solver runner node — execute the generated model code."""

from __future__ import annotations

from orpilot.solver.registry import get_solver
from orpilot.models.solution import SolveStatus
from orpilot.workflow.state import WorkflowState


def solver_runner_node(state: WorkflowState) -> WorkflowState:
    """Execute the generated solver code and capture results."""
    code = state.get("generated_code", "")
    user_data = state.get("user_data")
    solver_name = state.get("solver_name", "pulp")

    solver = get_solver(solver_name)
    data_dict = user_data.as_dict() if user_data else {}
    time_limit = state.get("solver_time_limit")
    show_solver_log = state.get("show_solver_log", False)

    solution = solver.solve(code, data_dict, time_limit=time_limit, show_solver_log=show_solver_log)

    updates: dict = {
        "solution": solution,
        "current_node": "solver_runner",
        "needs_user_input": False,
    }

    # If solve failed, set error context for retry
    if solution.status in (SolveStatus.ERROR, SolveStatus.INFEASIBLE, SolveStatus.UNBOUNDED):
        retry_count = state.get("retry_count", 0) + 1
        updates["retry_count"] = retry_count
        if solution.status == SolveStatus.UNBOUNDED:
            error_msg = (
                "The model is unbounded — the objective can grow to infinity. "
                "A variable or combination of variables is unconstrained in the objective "
                "direction. Check that all variables are bounded by constraints (e.g. "
                "warehouse capacity limits purchases, demand limits production). "
                "Add any missing upper-bound constraints."
            )
        else:
            error_msg = solution.error_message or solution.solver_output or ""
            # Truncate to avoid 413 when solver repeats an error thousands of times.
            # Keep first 2000 chars (captures the repeating error type) and last 500 chars
            # (captures final status), so the LLM gets the gist without the flood.
            if len(error_msg) > 3000:
                error_msg = error_msg[:2000] + "\n...[truncated]...\n" + error_msg[-500:]
        updates["error_context"] = (
            f"Solve failed with status={solution.status.value}. "
            f"Error: {error_msg}"
        )

    return {**state, **updates}
