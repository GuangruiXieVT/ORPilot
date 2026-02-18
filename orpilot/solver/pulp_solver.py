"""PuLP solver implementation."""

from __future__ import annotations

import time

from orpilot.codegen.executor import CodeExecutor
from orpilot.models.solution import SolutionResult, SolveStatus, VariableGroup

from .base import BaseSolver


class PuLPSolver(BaseSolver):
    name = "pulp"
    framework = "pulp"

    def __init__(self, timeout: int = 120):
        self._executor = CodeExecutor(
            timeout=timeout,
            allowed_modules=["pulp", "math", "itertools", "collections", "json"],
        )

    def solve(self, code: str, data: dict) -> SolutionResult:
        start = time.monotonic()
        result = self._executor.execute(code, data)
        elapsed = time.monotonic() - start

        lp_content = result.get("lp_content", "")

        if result.get("error"):
            return SolutionResult(
                status=SolveStatus.ERROR,
                error_message=result["error"],
                solver_output=result.get("stdout", ""),
                solve_time_seconds=elapsed,
                lp_content=lp_content,
            )

        solution = result.get("result", {})
        status_map = {
            "optimal": SolveStatus.OPTIMAL,
            "feasible": SolveStatus.FEASIBLE,
            "infeasible": SolveStatus.INFEASIBLE,
            "unbounded": SolveStatus.UNBOUNDED,
        }
        status = status_map.get(
            str(solution.get("status", "error")).lower(),
            SolveStatus.ERROR,
        )

        return SolutionResult(
            status=status,
            objective_value=solution.get("objective_value"),
            variables=solution.get("variables", {}),
            variable_groups=_parse_variable_groups(solution),
            solver_output=result.get("stdout", ""),
            solve_time_seconds=elapsed,
            lp_content=lp_content,
        )


def _parse_variable_groups(solution: dict) -> list[VariableGroup]:
    """Parse variable_groups from the solve() return dict."""
    raw = solution.get("variable_groups", [])
    groups = []
    for g in raw:
        if isinstance(g, dict):
            groups.append(VariableGroup(
                group_name=g.get("group_name", "variables"),
                dimension_labels=g.get("dimension_labels", []),
                variables=g.get("variables", {}),
            ))
    return groups
