"""Unit tests for IRCompiler — helper methods, key format regression, and execution."""

from __future__ import annotations

import importlib.util

import pytest

from orpilot.codegen.ir_compiler import IRCompiler
from orpilot.codegen.executor import CodeExecutor

PULP_AVAILABLE = importlib.util.find_spec("pulp") is not None
PYOMO_AVAILABLE = importlib.util.find_spec("pyomo") is not None
ORTOOLS_AVAILABLE = importlib.util.find_spec("ortools") is not None

# ---------------------------------------------------------------------------
# Minimal IR builders
# ---------------------------------------------------------------------------

def _scalar_minimize_ir():
    """Minimize a single scalar continuous variable (lb=2, ub=10)."""
    return {
        "problem_class": "Test",
        "model_type": "Linear Program",
        "sense": "minimize",
        "sets": {},
        "parameters": {},
        "variables": {
            "x": {
                "description": "scalar x",
                "domain": [],
                "type": "continuous",
                "lower_bound": 2,
                "upper_bound": 10,
                "upper_bound_set": None,
                "exclude_diagonal": False,
            }
        },
        "objective": {
            "sense": "minimize",
            "expression": {"type": "variable", "name": "x", "indices": []},
        },
        "constraints": {},
    }


def _knapsack_ir():
    """1-dim binary knapsack: maximize value subject to weight capacity."""
    return {
        "problem_class": "KnapsackTest",
        "model_type": "Integer Program",
        "sense": "maximize",
        "sets": {
            "Items": {
                "size": None,
                "index_symbol": "i",
                "source": "items.csv",
                "column": "item_id",
            }
        },
        "parameters": {
            "value": {
                "domain": ["Items"],
                "type": "float",
                "source": "items.csv",
                "column": "value",
            },
            "weight": {
                "domain": ["Items"],
                "type": "float",
                "source": "items.csv",
                "column": "weight",
            },
            "capacity": {
                "domain": [],
                "type": "float",
                "source": "capacity.csv",
                "column": "capacity",
            },
        },
        "variables": {
            "x": {
                "description": "1 if item selected",
                "domain": ["Items"],
                "type": "binary",
                "lower_bound": 0,
                "upper_bound": 1,
                "upper_bound_set": None,
                "exclude_diagonal": False,
            }
        },
        "objective": {
            "sense": "maximize",
            "expression": {
                "operation": "indexed_sum",
                "over": ["Items"],
                "body": {
                    "operation": "multiply",
                    "left": {"type": "parameter", "name": "value", "indices": ["i"]},
                    "right": {"type": "variable", "name": "x", "indices": ["i"]},
                },
            },
        },
        "constraints": {
            "capacity_limit": {
                "domain": [],
                "expression": {
                    "operation": "indexed_sum",
                    "over": ["Items"],
                    "body": {
                        "operation": "multiply",
                        "left": {"type": "parameter", "name": "weight", "indices": ["i"]},
                        "right": {"type": "variable", "name": "x", "indices": ["i"]},
                    },
                },
                "sense": "<=",
                "rhs": {"type": "parameter", "name": "capacity", "indices": []},
            }
        },
    }


# Items: A(v=5,w=4), B(v=3,w=3), C(v=4,w=2). Capacity=5.
# Optimal: B+C selected (value=7, weight=5)
_KNAPSACK_DATA = {
    "items": [
        {"item_id": "A", "value": "5", "weight": "4"},
        {"item_id": "B", "value": "3", "weight": "3"},
        {"item_id": "C", "value": "4", "weight": "2"},
    ],
    "capacity": [{"capacity": "5"}],
}


def _transport_2dim_ir():
    """2-dim continuous LP: 2-warehouse × 2-customer transportation."""
    return {
        "problem_class": "TransportTest",
        "model_type": "Linear Program",
        "sense": "minimize",
        "sets": {
            "Warehouses": {
                "size": None,
                "index_symbol": "w",
                "source": "warehouses.csv",
                "column": "warehouse_id",
            },
            "Customers": {
                "size": None,
                "index_symbol": "c",
                "source": "customers.csv",
                "column": "customer_id",
            },
        },
        "parameters": {
            "supply": {
                "domain": ["Warehouses"],
                "type": "float",
                "source": "warehouses.csv",
                "column": "supply",
            },
            "demand": {
                "domain": ["Customers"],
                "type": "float",
                "source": "customers.csv",
                "column": "demand",
            },
            "cost": {
                "domain": ["Warehouses", "Customers"],
                "type": "float",
                "source": "costs.csv",
                "column": "cost",
                "index_columns": ["from_id", "to_id"],
                "missing_default": "inf",
            },
        },
        "variables": {
            "flow": {
                "description": "Units shipped from w to c",
                "domain": ["Warehouses", "Customers"],
                "type": "continuous",
                "lower_bound": 0,
                "upper_bound": None,
                "upper_bound_set": None,
                "exclude_diagonal": False,
            }
        },
        "objective": {
            "sense": "minimize",
            "expression": {
                "operation": "indexed_sum",
                "over": ["Warehouses", "Customers"],
                "body": {
                    "operation": "multiply",
                    "left": {"type": "parameter", "name": "cost", "indices": ["w", "c"]},
                    "right": {"type": "variable", "name": "flow", "indices": ["w", "c"]},
                },
            },
        },
        "constraints": {
            "supply_limit": {
                "domain": ["Warehouses"],
                "expression": {
                    "operation": "indexed_sum",
                    "over": ["Customers"],
                    "body": {"type": "variable", "name": "flow", "indices": ["w", "c"]},
                },
                "sense": "<=",
                "rhs": {"type": "parameter", "name": "supply", "indices": ["w"]},
            },
            "demand_met": {
                "domain": ["Customers"],
                "expression": {
                    "operation": "indexed_sum",
                    "over": ["Warehouses"],
                    "body": {"type": "variable", "name": "flow", "indices": ["w", "c"]},
                },
                "sense": "=",
                "rhs": {"type": "parameter", "name": "demand", "indices": ["c"]},
            },
        },
    }


# W1=30, W2=20, C1=20, C2=30. Costs: W1→C1:2, W1→C2:5, W2→C1:3, W2→C2:4
# Optimal: W1→C1=20, W1→C2=10, W2→C2=20. Total cost = 40+50+80 = 170
_TRANSPORT_DATA = {
    "warehouses": [
        {"warehouse_id": "W1", "supply": "30"},
        {"warehouse_id": "W2", "supply": "20"},
    ],
    "customers": [
        {"customer_id": "C1", "demand": "20"},
        {"customer_id": "C2", "demand": "30"},
    ],
    "costs": [
        {"from_id": "W1", "to_id": "C1", "cost": "2"},
        {"from_id": "W1", "to_id": "C2", "cost": "5"},
        {"from_id": "W2", "to_id": "C1", "cost": "3"},
        {"from_id": "W2", "to_id": "C2", "cost": "4"},
    ],
}


def _exclude_diag_ir():
    """Binary variable on Locations×Locations with exclude_diagonal=True."""
    return {
        "problem_class": "DiagTest",
        "model_type": "Integer Program",
        "sense": "minimize",
        "sets": {
            "Locations": {
                "size": None,
                "index_symbol": "l",
                "source": "locs.csv",
                "column": "loc_id",
            }
        },
        "parameters": {},
        "variables": {
            "x": {
                "description": "arc from l1 to l2",
                "domain": ["Locations", "Locations"],
                "type": "binary",
                "lower_bound": 0,
                "upper_bound": 1,
                "upper_bound_set": None,
                "exclude_diagonal": True,
            }
        },
        "objective": {
            "sense": "minimize",
            "expression": {
                "operation": "indexed_sum",
                "over": ["Locations:l1", "Locations:l2"],
                "body": {"type": "variable", "name": "x", "indices": ["l1", "l2"]},
            },
        },
        "constraints": {
            "coverage": {
                "domain": [],
                "expression": {
                    "operation": "indexed_sum",
                    "over": ["Locations:l1", "Locations:l2"],
                    "body": {"type": "variable", "name": "x", "indices": ["l1", "l2"]},
                },
                "sense": "=",
                "rhs": {"type": "constant", "value": 1},
            }
        },
    }


_DIAG_DATA = {
    "locs": [{"loc_id": "A"}, {"loc_id": "B"}],
}


def _set_size_ir():
    """Integer variable u[Customers] with upper_bound_set and set_size in constraint."""
    return {
        "problem_class": "SetSizeTest",
        "model_type": "Integer Program",
        "sense": "minimize",
        "sets": {
            "Customers": {
                "size": None,
                "index_symbol": "c",
                "source": "customers.csv",
                "column": "customer_id",
            }
        },
        "parameters": {},
        "variables": {
            "u": {
                "description": "position variable",
                "domain": ["Customers"],
                "type": "integer",
                "lower_bound": 1,
                "upper_bound": None,
                "upper_bound_set": "Customers",
                "exclude_diagonal": False,
            }
        },
        "objective": {
            "sense": "minimize",
            "expression": {
                "operation": "indexed_sum",
                "over": ["Customers"],
                "body": {"type": "variable", "name": "u", "indices": ["c"]},
            },
        },
        "constraints": {
            "pos_ub": {
                "domain": ["Customers"],
                "expression": {"type": "variable", "name": "u", "indices": ["c"]},
                "sense": "<=",
                "rhs": {"type": "set_size", "set": "Customers"},
            }
        },
    }


_SET_SIZE_DATA = {
    "customers": [
        {"customer_id": "c1"},
        {"customer_id": "c2"},
        {"customer_id": "c3"},
    ],
}


def _dup_domain_ir():
    """Parameter with same set twice in domain (dist[Locations, Locations])."""
    return {
        "problem_class": "DupDomainTest",
        "model_type": "Linear Program",
        "sense": "minimize",
        "sets": {
            "Locations": {
                "size": None,
                "index_symbol": "l",
                "source": "locs.csv",
                "column": "loc_id",
            }
        },
        "parameters": {
            "dist": {
                "domain": ["Locations", "Locations"],
                "type": "float",
                "source": "distances.csv",
                "column": "dist",
                "index_columns": ["from_id", "to_id"],
            }
        },
        "variables": {
            "y": {
                "description": "continuous arc",
                "domain": ["Locations", "Locations"],
                "type": "continuous",
                "lower_bound": 0,
                "upper_bound": None,
                "upper_bound_set": None,
                "exclude_diagonal": True,
            }
        },
        "objective": {
            "sense": "minimize",
            "expression": {
                "operation": "indexed_sum",
                "over": ["Locations:l1", "Locations:l2"],
                "body": {
                    "operation": "multiply",
                    "left": {"type": "parameter", "name": "dist", "indices": ["l1", "l2"]},
                    "right": {"type": "variable", "name": "y", "indices": ["l1", "l2"]},
                },
            },
        },
        "constraints": {
            "flow_balance": {
                "domain": [],
                "expression": {
                    "operation": "indexed_sum",
                    "over": ["Locations:l1", "Locations:l2"],
                    "body": {"type": "variable", "name": "y", "indices": ["l1", "l2"]},
                },
                "sense": "=",
                "rhs": {"type": "constant", "value": 1},
            }
        },
    }


# ---------------------------------------------------------------------------
# Helper method tests
# ---------------------------------------------------------------------------


def test_normalize_ir_strips_aliases_from_domain():
    ir = {
        "problem_class": "T",
        "model_type": "LP",
        "sense": "minimize",
        "sets": {
            "Locations": {"size": None, "index_symbol": "l", "source": None, "column": None}
        },
        "parameters": {
            "dist": {"domain": ["Locations:l1", "Locations:l2"], "type": "float", "source": None}
        },
        "variables": {
            "x": {
                "description": "t",
                "domain": ["Locations:l"],
                "type": "continuous",
                "lower_bound": 0,
                "upper_bound": None,
            }
        },
        "constraints": {
            "c1": {
                "domain": ["Locations:l"],
                "expression": {"type": "constant", "value": 0},
                "sense": "<=",
                "rhs": {"type": "constant", "value": 1},
            }
        },
        "objective": {
            "sense": "minimize",
            "expression": {"type": "constant", "value": 0},
        },
    }
    normalized = IRCompiler._normalize_ir(ir)
    assert normalized["parameters"]["dist"]["domain"] == ["Locations", "Locations"]
    assert normalized["variables"]["x"]["domain"] == ["Locations"]
    assert normalized["constraints"]["c1"]["domain"] == ["Locations"]


def test_domain_idx_vars_unique_sets():
    index_map = {"Trips": "t", "Locations": "l"}
    result = IRCompiler._domain_idx_vars(["Trips"], index_map)
    assert result == ["t"]


def test_domain_idx_vars_duplicate_sets():
    index_map = {"Locations": "l", "Trips": "t"}
    result = IRCompiler._domain_idx_vars(["Locations", "Locations", "Trips"], index_map)
    assert result == ["l1", "l2", "t"]


def test_constraint_diagonal_guard_no_duplicate():
    result = IRCompiler._constraint_diagonal_guard(["Locations", "Trips"], ["l", "t"])
    assert result is None


def test_constraint_diagonal_guard_with_duplicate():
    result = IRCompiler._constraint_diagonal_guard(
        ["Customers", "Customers", "Trips"], ["c1", "c2", "t"]
    )
    assert result == "c1 == c2"


# ---------------------------------------------------------------------------
# Variable key format regression tests (underscore-in-ID bug)
# ---------------------------------------------------------------------------


def test_compiled_keys_use_unit_separator_1dim():
    ir = {
        "problem_class": "Test",
        "model_type": "LP",
        "sense": "minimize",
        "sets": {
            "Items": {
                "size": None,
                "index_symbol": "i",
                "source": "items.csv",
                "column": "item_id",
            }
        },
        "parameters": {},
        "variables": {
            "x": {
                "description": "t",
                "domain": ["Items"],
                "type": "continuous",
                "lower_bound": 0,
                "upper_bound": None,
                "upper_bound_set": None,
                "exclude_diagonal": False,
            }
        },
        "objective": {
            "sense": "minimize",
            "expression": {"type": "constant", "value": 0},
        },
        "constraints": {},
    }
    code = IRCompiler().compile(ir, "pulp")
    # Result keys must use \x1f separator, NOT bare underscore
    assert "\\x1f" in code
    # Specifically the extraction f-string
    assert 'f"x\\x1f{i}"' in code


def test_compiled_keys_use_unit_separator_2dim():
    ir = _transport_2dim_ir()
    code = IRCompiler().compile(ir, "pulp")
    assert "\\x1f" in code
    # 2-dim extraction emits f"flow\x1f{w}\x1f{c}"
    assert 'f"flow\\x1f{w}\\x1f{c}"' in code


def test_compiled_keys_use_unit_separator_ndim():
    """3-dim variable should also use \\x1f separators in result keys."""
    ir = {
        "problem_class": "Test",
        "model_type": "LP",
        "sense": "minimize",
        "sets": {
            "A": {"size": None, "index_symbol": "a", "source": "a.csv", "column": "a_id"},
            "B": {"size": None, "index_symbol": "b", "source": "b.csv", "column": "b_id"},
            "C": {"size": None, "index_symbol": "c", "source": "c.csv", "column": "c_id"},
        },
        "parameters": {},
        "variables": {
            "z": {
                "description": "3-dim",
                "domain": ["A", "B", "C"],
                "type": "continuous",
                "lower_bound": 0,
                "upper_bound": None,
                "upper_bound_set": None,
                "exclude_diagonal": False,
            }
        },
        "objective": {
            "sense": "minimize",
            "expression": {"type": "constant", "value": 0},
        },
        "constraints": {},
    }
    code = IRCompiler().compile(ir, "pulp")
    assert "\\x1f" in code
    # 3-dim: f"z\x1f{a}\x1f{b}\x1f{c}"
    assert 'f"z\\x1f{a}\\x1f{b}\\x1f{c}"' in code


# ---------------------------------------------------------------------------
# Compilation + execution tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not PULP_AVAILABLE, reason="pulp not installed")
def test_pulp_scalar_variable():
    code = IRCompiler().compile(_scalar_minimize_ir(), "pulp")
    result = CodeExecutor(timeout=30).execute(code, {})
    assert result["error"] is None
    r = result["result"]
    assert r["status"] == "optimal"
    assert abs(r["objective_value"] - 2.0) < 0.01
    assert abs(r["variables"]["x"] - 2.0) < 0.01


@pytest.mark.skipif(not PULP_AVAILABLE, reason="pulp not installed")
def test_pulp_1dim_maximize():
    """Knapsack: B+C selected (value=7, weight=5 ≤ 5)."""
    code = IRCompiler().compile(_knapsack_ir(), "pulp")
    result = CodeExecutor(timeout=30).execute(code, _KNAPSACK_DATA)
    assert result["error"] is None
    r = result["result"]
    assert r["status"] == "optimal"
    assert abs(r["objective_value"] - 7.0) < 0.01
    SEP = "\x1f"
    assert abs(r["variables"][f"x{SEP}B"] - 1.0) < 0.01
    assert abs(r["variables"][f"x{SEP}C"] - 1.0) < 0.01
    assert abs(r["variables"][f"x{SEP}A"]) < 0.01  # not selected


@pytest.mark.skipif(not PULP_AVAILABLE, reason="pulp not installed")
def test_pulp_2dim_continuous():
    """2×2 transportation: optimal cost = 170."""
    code = IRCompiler().compile(_transport_2dim_ir(), "pulp")
    result = CodeExecutor(timeout=30).execute(code, _TRANSPORT_DATA)
    assert result["error"] is None
    r = result["result"]
    assert r["status"] == "optimal"
    assert abs(r["objective_value"] - 170.0) < 0.1


@pytest.mark.skipif(not PULP_AVAILABLE, reason="pulp not installed")
def test_pulp_exclude_diagonal():
    """Binary variable on Locations×Locations: diagonal keys absent from result."""
    code = IRCompiler().compile(_exclude_diag_ir(), "pulp")
    result = CodeExecutor(timeout=30).execute(code, _DIAG_DATA)
    assert result["error"] is None
    r = result["result"]
    assert r["status"] == "optimal"
    SEP = "\x1f"
    # Diagonal entries must not appear in result
    assert f"x{SEP}A{SEP}A" not in r["variables"]
    assert f"x{SEP}B{SEP}B" not in r["variables"]
    # Non-diagonal entries must appear
    assert f"x{SEP}A{SEP}B" in r["variables"] or f"x{SEP}B{SEP}A" in r["variables"]


@pytest.mark.skipif(not PULP_AVAILABLE, reason="pulp not installed")
def test_pulp_set_size_expression():
    """set_size node in constraint RHS compiles and solves: minimize sum(u) ≥ 3."""
    code = IRCompiler().compile(_set_size_ir(), "pulp")
    result = CodeExecutor(timeout=30).execute(code, _SET_SIZE_DATA)
    assert result["error"] is None
    r = result["result"]
    assert r["status"] == "optimal"
    # With lb=1 and objective minimize, all u=1, objective=3
    assert abs(r["objective_value"] - 3.0) < 0.01


@pytest.mark.skipif(not PULP_AVAILABLE, reason="pulp not installed")
def test_pulp_upper_bound_set():
    """upper_bound_set field causes compiler to emit len(SetName) as upper bound."""
    code = IRCompiler().compile(_set_size_ir(), "pulp")
    assert "len(Customers)" in code


@pytest.mark.skipif(not PULP_AVAILABLE, reason="pulp not installed")
def test_pulp_duplicate_set_domain():
    """Parameter with duplicate-set domain uses .get() in objective/constraint."""
    code = IRCompiler().compile(_dup_domain_ir(), "pulp")
    # dist.get((...), 0.0) should appear since Locations appears twice in domain
    assert "dist.get((" in code


@pytest.mark.skipif(not PYOMO_AVAILABLE, reason="pyomo not installed")
def test_pyomo_1dim_lp():
    """Pyomo backend: same knapsack problem (binary), objective = 7."""
    code = IRCompiler().compile(_knapsack_ir(), "pyomo")
    result = CodeExecutor(timeout=60).execute(code, _KNAPSACK_DATA)
    r = result["result"]
    # Skip if no Pyomo solver is available
    if r and r.get("status") == "error" and "solver" in r.get("error", "").lower():
        pytest.skip("No Pyomo-compatible solver installed")
    assert result["error"] is None
    assert r["status"] == "optimal"
    assert abs(r["objective_value"] - 7.0) < 0.5


@pytest.mark.skipif(not ORTOOLS_AVAILABLE, reason="ortools not installed")
def test_ortools_1dim_ip():
    """OR-Tools backend: knapsack problem (binary), objective = 7."""
    code = IRCompiler().compile(_knapsack_ir(), "ortools")
    result = CodeExecutor(timeout=30).execute(code, _KNAPSACK_DATA)
    assert result["error"] is None
    r = result["result"]
    assert r["status"] == "optimal"
    assert abs(r["objective_value"] - 7.0) < 0.01
