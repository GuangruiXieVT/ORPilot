"""Tests for solver registry."""

import pytest

from aifor.solver.registry import get_solver, list_solvers
from aifor.solver.base import BaseSolver


def test_list_solvers():
    solvers = list_solvers()
    assert "pulp" in solvers
    assert "pyomo" in solvers
    assert "ortools" in solvers


def test_get_pulp_solver():
    solver = get_solver("pulp")
    assert isinstance(solver, BaseSolver)
    assert solver.name == "pulp"


def test_get_unknown_solver():
    with pytest.raises(ValueError, match="Unknown solver"):
        get_solver("nonexistent")
