"""Unit tests for ir_validator.py — semantic checks on IR dicts.

Each test targets a specific check function so failures point directly
to the rule that broke, not to a downstream compile error.
"""

from __future__ import annotations

import pytest

from orpilot.codegen.ir_validator import validate_ir_semantics


# ---------------------------------------------------------------------------
# Minimal valid IR skeleton — used as a base for all tests
# ---------------------------------------------------------------------------

def _base_ir(**overrides) -> dict:
    """Return a minimal valid IR, with optional top-level overrides."""
    ir = {
        "problem_class": "Test",
        "model_type": "Linear Program",
        "sense": "minimize",
        "sets": {},
        "parameters": {},
        "variables": {
            "x": {
                "description": "test var",
                "domain": [],
                "type": "continuous",
                "lower_bound": 0,
                "upper_bound": None,
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
    ir.update(overrides)
    return ir


# ---------------------------------------------------------------------------
# validate_ir_semantics: clean IR passes
# ---------------------------------------------------------------------------

def test_valid_ir_produces_no_errors():
    errors = validate_ir_semantics(_base_ir())
    assert errors == []


# ---------------------------------------------------------------------------
# _check_rhs_variable_nodes
# ---------------------------------------------------------------------------

def test_rhs_bare_variable_flagged():
    ir = _base_ir(constraints={
        "bad_constraint": {
            "domain": [],
            "expression": {"type": "variable", "name": "x", "indices": []},
            "sense": "<=",
            "rhs": {"type": "variable", "name": "x", "indices": []},
        }
    })
    errors = validate_ir_semantics(ir)
    assert any("rhs is a bare variable" in e for e in errors)


def test_rhs_constant_ok():
    ir = _base_ir(constraints={
        "ok_constraint": {
            "domain": [],
            "expression": {"type": "variable", "name": "x", "indices": []},
            "sense": "<=",
            "rhs": {"type": "constant", "value": 10},
        }
    })
    assert validate_ir_semantics(ir) == []


# ---------------------------------------------------------------------------
# _check_variable_times_variable
# ---------------------------------------------------------------------------

def test_variable_times_variable_in_objective_flagged():
    ir = _base_ir()
    ir["variables"]["y"] = {
        "description": "y", "domain": [], "type": "continuous",
        "lower_bound": 0, "upper_bound": None, "upper_bound_set": None, "exclude_diagonal": False,
    }
    ir["objective"]["expression"] = {
        "operation": "multiply",
        "left": {"type": "variable", "name": "x", "indices": []},
        "right": {"type": "variable", "name": "y", "indices": []},
    }
    errors = validate_ir_semantics(ir)
    assert any("Variable × variable" in e or "variable(s)" in e for e in errors)


def test_parameter_times_variable_ok():
    ir = _base_ir(parameters={
        "cost": {"domain": [], "type": "float", "source": None, "column": "cost"},
    })
    ir["objective"]["expression"] = {
        "operation": "multiply",
        "left": {"type": "parameter", "name": "cost", "indices": []},
        "right": {"type": "variable", "name": "x", "indices": []},
    }
    assert validate_ir_semantics(ir) == []


# ---------------------------------------------------------------------------
# _check_alias_in_domain
# ---------------------------------------------------------------------------

def test_alias_in_variable_domain_flagged():
    ir = _base_ir(sets={
        "Locs": {"size": None, "index_symbol": "l", "source": None, "column": None}
    })
    ir["variables"]["x"]["domain"] = ["Locs:k"]
    errors = validate_ir_semantics(ir)
    assert any("alias" in e.lower() or "SetName:alias" in e for e in errors)


def test_alias_in_constraint_domain_flagged():
    ir = _base_ir(sets={
        "Locs": {"size": None, "index_symbol": "l", "source": None, "column": None}
    })
    ir["constraints"]["c1"] = {
        "domain": ["Locs:k"],
        "expression": {"type": "constant", "value": 0},
        "sense": "<=",
        "rhs": {"type": "constant", "value": 1},
    }
    errors = validate_ir_semantics(ir)
    assert any("alias" in e.lower() or "SetName:alias" in e for e in errors)


def test_plain_set_in_domain_ok():
    ir = _base_ir(sets={
        "Locs": {"size": None, "index_symbol": "l", "source": None, "column": None}
    })
    ir["variables"]["x"]["domain"] = ["Locs"]
    assert validate_ir_semantics(ir) == []


# ---------------------------------------------------------------------------
# _check_lag_in_objective
# ---------------------------------------------------------------------------

def test_lag_in_objective_flagged():
    ir = _base_ir()
    ir["objective"]["expression"] = {
        "type": "variable", "name": "x", "indices": [], "lag": -1
    }
    errors = validate_ir_semantics(ir)
    assert any("lag" in e.lower() for e in errors)


def test_no_lag_in_objective_ok():
    ir = _base_ir()
    ir["objective"]["expression"] = {"type": "variable", "name": "x", "indices": []}
    assert validate_ir_semantics(ir) == []


# ---------------------------------------------------------------------------
# _check_lag_without_init
# ---------------------------------------------------------------------------

def test_lag_constraint_without_init_flagged():
    ir = _base_ir(
        sets={"Periods": {"size": None, "index_symbol": "t", "source": "periods.csv",
                          "column": "period_id", "ordered": True}},
        variables={
            "inventory": {
                "description": "inv", "domain": ["Periods"], "type": "continuous",
                "lower_bound": 0, "upper_bound": None, "upper_bound_set": None,
                "exclude_diagonal": False,
            }
        },
    )
    ir["objective"]["expression"] = {
        "operation": "indexed_sum", "over": ["Periods"],
        "body": {"type": "variable", "name": "inventory", "indices": ["t"]},
    }
    ir["constraints"]["inventory_balance"] = {
        "domain": ["Periods"],
        "expression": {
            "operation": "subtract",
            "left": {"type": "variable", "name": "inventory", "indices": ["t"]},
            "right": {"type": "variable", "name": "inventory", "indices": ["t"], "lag": -1},
        },
        "sense": "=",
        "rhs": {"type": "constant", "value": 0},
    }
    errors = validate_ir_semantics(ir)
    assert any("init" in e.lower() for e in errors)


def test_lag_constraint_with_init_ok():
    ir = _base_ir(
        sets={"Periods": {"size": None, "index_symbol": "t", "source": "periods.csv",
                          "column": "period_id", "ordered": True}},
        variables={
            "inventory": {
                "description": "inv", "domain": ["Periods"], "type": "continuous",
                "lower_bound": 0, "upper_bound": None, "upper_bound_set": None,
                "exclude_diagonal": False,
            }
        },
    )
    ir["objective"]["expression"] = {
        "operation": "indexed_sum", "over": ["Periods"],
        "body": {"type": "variable", "name": "inventory", "indices": ["t"]},
    }
    ir["constraints"]["inventory_balance"] = {
        "domain": ["Periods"],
        "expression": {
            "operation": "subtract",
            "left": {"type": "variable", "name": "inventory", "indices": ["t"]},
            "right": {"type": "variable", "name": "inventory", "indices": ["t"], "lag": -1},
        },
        "sense": "=",
        "rhs": {"type": "constant", "value": 0},
    }
    ir["constraints"]["inventory_balance_init"] = {
        "domain": [],
        "expression": {"type": "variable", "name": "inventory", "indices": ["Periods[0]"]},
        "sense": "=",
        "rhs": {"type": "constant", "value": 0},
    }
    errors = validate_ir_semantics(ir)
    assert not any("init" in e.lower() and "lag" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# _check_period_set_with_hardcoded_size
# ---------------------------------------------------------------------------

def test_ordered_set_with_hardcoded_size_flagged():
    ir = _base_ir(sets={
        "Periods": {
            "size": 12,
            "index_symbol": "t",
            "source": None,
            "column": None,
            "ordered": True,
        }
    })
    errors = validate_ir_semantics(ir)
    assert any("hardcoded size" in e.lower() or "ordered" in e.lower() for e in errors)


def test_ordered_set_with_csv_source_ok():
    ir = _base_ir(sets={
        "Periods": {
            "size": None,
            "index_symbol": "t",
            "source": "periods.csv",
            "column": "period_id",
            "ordered": True,
        }
    })
    assert validate_ir_semantics(ir) == []


# ---------------------------------------------------------------------------
# _check_cost_capacity_missing_default
# ---------------------------------------------------------------------------

def test_multidim_cost_param_without_inf_flagged():
    ir = _base_ir(
        sets={
            "A": {"size": None, "index_symbol": "a", "source": "a.csv", "column": "a_id"},
            "B": {"size": None, "index_symbol": "b", "source": "b.csv", "column": "b_id"},
        },
        parameters={
            "cost": {
                "domain": ["A", "B"],
                "type": "float",
                "source": "costs.csv",
                "column": "cost",
                "index_columns": ["a_id", "b_id"],
                "missing_default": "zero",
            }
        },
    )
    errors = validate_ir_semantics(ir)
    assert any("missing_default" in e for e in errors)


def test_multidim_cost_param_with_inf_ok():
    ir = _base_ir(
        sets={
            "A": {"size": None, "index_symbol": "a", "source": "a.csv", "column": "a_id"},
            "B": {"size": None, "index_symbol": "b", "source": "b.csv", "column": "b_id"},
        },
        parameters={
            "cost": {
                "domain": ["A", "B"],
                "type": "float",
                "source": "costs.csv",
                "column": "cost",
                "index_columns": ["a_id", "b_id"],
                "missing_default": "inf",
            }
        },
    )
    ir["variables"]["x"]["domain"] = ["A", "B"]
    # Reference cost in the objective so _check_cost_params_not_in_objective passes
    ir["objective"]["expression"] = {
        "operation": "indexed_sum",
        "over": ["A", "B"],
        "body": {
            "operation": "multiply",
            "left": {"type": "parameter", "name": "cost", "indices": ["a", "b"]},
            "right": {"type": "variable", "name": "x", "indices": ["a", "b"]},
        },
    }
    assert validate_ir_semantics(ir) == []


def test_scalar_capacity_param_not_flagged():
    """Scalar parameters (domain=[]) must not be flagged even if named 'capacity'."""
    ir = _base_ir(parameters={
        "capacity": {
            "domain": [],
            "type": "float",
            "source": "cap.csv",
            "column": "capacity",
        }
    })
    assert validate_ir_semantics(ir) == []


# ---------------------------------------------------------------------------
# _check_objective_nested_subtract
# ---------------------------------------------------------------------------

def test_nested_subtract_in_objective_flagged():
    """subtract(A, subtract(B, C)) flips C's sign — must be caught."""
    ir = _base_ir()
    ir["variables"]["y"] = {
        "description": "y", "domain": [], "type": "continuous",
        "lower_bound": 0, "upper_bound": None, "upper_bound_set": None, "exclude_diagonal": False,
    }
    ir["variables"]["z"] = {
        "description": "z", "domain": [], "type": "continuous",
        "lower_bound": 0, "upper_bound": None, "upper_bound_set": None, "exclude_diagonal": False,
    }
    ir["objective"]["expression"] = {
        "operation": "subtract",
        "left": {"type": "variable", "name": "x", "indices": []},
        "right": {
            "operation": "subtract",
            "left": {"type": "variable", "name": "y", "indices": []},
            "right": {"type": "variable", "name": "z", "indices": []},
        },
    }
    errors = validate_ir_semantics(ir)
    assert any("nested" in e.lower() or "subtract" in e.lower() for e in errors)


def test_flat_subtract_chain_ok():
    """subtract(subtract(A, B), C) is a left-to-right chain — valid."""
    ir = _base_ir()
    ir["variables"]["y"] = {
        "description": "y", "domain": [], "type": "continuous",
        "lower_bound": 0, "upper_bound": None, "upper_bound_set": None, "exclude_diagonal": False,
    }
    ir["variables"]["z"] = {
        "description": "z", "domain": [], "type": "continuous",
        "lower_bound": 0, "upper_bound": None, "upper_bound_set": None, "exclude_diagonal": False,
    }
    ir["objective"]["expression"] = {
        "operation": "subtract",
        "left": {
            "operation": "subtract",
            "left": {"type": "variable", "name": "x", "indices": []},
            "right": {"type": "variable", "name": "y", "indices": []},
        },
        "right": {"type": "variable", "name": "z", "indices": []},
    }
    assert validate_ir_semantics(ir) == []


# ---------------------------------------------------------------------------
# _check_duplicate_domain_index_columns
# ---------------------------------------------------------------------------

def test_dup_domain_without_index_columns_flagged():
    ir = _base_ir(
        sets={"Locs": {"size": None, "index_symbol": "l", "source": "locs.csv", "column": "loc_id"}},
        parameters={
            "dist": {
                "domain": ["Locs", "Locs"],
                "type": "float",
                "source": "distances.csv",
                "column": "dist",
            }
        },
    )
    errors = validate_ir_semantics(ir)
    assert any("index_columns" in e for e in errors)


def test_dup_domain_with_index_columns_ok():
    ir = _base_ir(
        sets={"Locs": {"size": None, "index_symbol": "l", "source": "locs.csv", "column": "loc_id"}},
        parameters={
            "dist": {
                "domain": ["Locs", "Locs"],
                "type": "float",
                "source": "distances.csv",
                "column": "dist",
                "index_columns": ["from_id", "to_id"],
                "missing_default": "inf",
            }
        },
    )
    assert validate_ir_semantics(ir) == []


# ---------------------------------------------------------------------------
# _check_shared_source_without_filter
# ---------------------------------------------------------------------------

def test_shared_source_without_filter_flagged():
    ir = _base_ir(sets={
        "Products": {
            "size": None, "index_symbol": "p",
            "source": "sets.csv", "column": "element",
        },
        "Periods": {
            "size": None, "index_symbol": "t",
            "source": "sets.csv", "column": "element",
        },
    })
    errors = validate_ir_semantics(ir)
    assert any("filter" in e.lower() for e in errors)


def test_shared_source_with_filter_ok():
    ir = _base_ir(sets={
        "Products": {
            "size": None, "index_symbol": "p",
            "source": "sets.csv", "column": "element",
            "filter_column": "set_name", "filter_value": "Products",
        },
        "Periods": {
            "size": None, "index_symbol": "t",
            "source": "sets.csv", "column": "element",
            "filter_column": "set_name", "filter_value": "Periods",
        },
    })
    assert validate_ir_semantics(ir) == []
