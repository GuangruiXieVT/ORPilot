"""System prompt for the direct code generation LLM node."""

from __future__ import annotations

from ._loader import load_prompt

_SOLVER_TEMPLATES: dict[str, str] = {
    solver: load_prompt(f"direct_code_gen_{solver}.md")[0]
    for solver in ("pulp", "pyomo", "ortools", "gurobi", "cplex")
}


def build_system_prompt(solver: str) -> str:
    """Return the system prompt for the given solver, with the solver name injected."""
    template = _SOLVER_TEMPLATES.get(solver.lower(), _SOLVER_TEMPLATES["pulp"])
    return template.format(solver=solver)
