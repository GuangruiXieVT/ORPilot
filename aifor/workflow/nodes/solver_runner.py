"""Solver runner node — execute the generated model code."""

from __future__ import annotations

from aifor.solver.registry import get_solver
from aifor.models.solution import SolveStatus
from aifor.workflow.state import WorkflowState


def solver_runner_node(state: WorkflowState) -> WorkflowState:
    """Execute the generated solver code and capture results."""
    code = state.get("generated_code", "")
    user_data = state.get("user_data")
    solver_name = state.get("solver_name", "pulp")

    solver = get_solver(solver_name)
    data_dict = user_data.as_dict() if user_data else {}

    solution = solver.solve(code, data_dict)

    updates: dict = {
        "solution": solution,
        "current_node": "solver_runner",
        "needs_user_input": False,
    }

    # If solve failed, set error context for retry
    if solution.status in (SolveStatus.ERROR, SolveStatus.INFEASIBLE):
        retry_count = state.get("retry_count", 0) + 1
        updates["retry_count"] = retry_count
        error_msg = solution.error_message or solution.solver_output
        updates["error_context"] = (
            f"Solve failed with status={solution.status.value}. "
            f"Error: {error_msg}"
        )

    return {**state, **updates}
