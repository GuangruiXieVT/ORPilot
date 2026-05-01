"""Solver registry and selection."""

from __future__ import annotations

from .base import BaseSolver


_SOLVER_REGISTRY: dict[str, type[BaseSolver]] = {}


def _register_defaults() -> None:
    from .pulp_solver import PuLPSolver
    from .pyomo_solver import PyomoSolver
    from .ortools_solver import ORToolsSolver
    from .gurobi_solver import GurobiSolver
    from .cplex_solver import CplexSolver

    _SOLVER_REGISTRY["pulp"] = PuLPSolver
    _SOLVER_REGISTRY["pyomo"] = PyomoSolver
    _SOLVER_REGISTRY["ortools"] = ORToolsSolver
    _SOLVER_REGISTRY["gurobi"] = GurobiSolver
    _SOLVER_REGISTRY["cplex"] = CplexSolver


def get_solver(name: str = "pulp", **kwargs) -> BaseSolver:
    """Get a solver instance by name."""
    if not _SOLVER_REGISTRY:
        _register_defaults()

    name = name.lower()
    if name not in _SOLVER_REGISTRY:
        available = ", ".join(_SOLVER_REGISTRY)
        raise ValueError(f"Unknown solver: {name!r}. Available: {available}")

    return _SOLVER_REGISTRY[name](**kwargs)


def list_solvers() -> list[str]:
    """List available solver names."""
    if not _SOLVER_REGISTRY:
        _register_defaults()
    return list(_SOLVER_REGISTRY)
