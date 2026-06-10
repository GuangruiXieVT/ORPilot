"""Unit tests for orpilot.rag.structural (ir_fingerprint, jaccard, detect_modeling_patterns)."""

from __future__ import annotations

import pytest

from orpilot.rag.structural import ir_fingerprint, jaccard, detect_modeling_patterns


# ---------------------------------------------------------------------------
# jaccard
# ---------------------------------------------------------------------------

def test_jaccard_identical():
    a = {"a", "b", "c"}
    assert jaccard(a, a) == 1.0


def test_jaccard_disjoint():
    assert jaccard({"a", "b"}, {"c", "d"}) == 0.0


def test_jaccard_partial():
    score = jaccard({"a", "b", "c"}, {"b", "c", "d"})
    assert abs(score - 2 / 4) < 1e-9


def test_jaccard_both_empty():
    assert jaccard(set(), set()) == 0.0


def test_jaccard_one_empty():
    assert jaccard({"a"}, set()) == 0.0


# ---------------------------------------------------------------------------
# ir_fingerprint
# ---------------------------------------------------------------------------

_LP_EXAMPLE = {
    "modeling_patterns": [],
    "ir": {
        "sense": "minimize",
        "model_type": "Linear Program",
        "sets": {"Products": {}, "Periods": {"ordered": True}},
        "variables": {
            "x": {"type": "continuous", "domain": ["Products", "Periods"]},
        },
        "constraints": {
            "c1": {"domain": ["Products", "Periods"], "expression": {}, "sparse_filter": None},
            "c2": {"domain": ["Products"], "expression": {}},
        },
        "objective": {},
    },
}

_MIP_EXAMPLE = {
    "modeling_patterns": ["binary variable", "big-M linking"],
    "ir": {
        "sense": "maximize",
        "model_type": "Mixed Integer Program",
        "sets": {"Items": {}, "Bins": {}, "Types": {}},
        "variables": {
            "y": {"type": "binary", "domain": ["Items", "Bins"], "exclude_diagonal": False,
                  "domain_filter": None},
            "z": {"type": "continuous", "domain": ["Items"], "domain_filter": "cost",
                  "exclude_diagonal": False},
        },
        "constraints": {
            "c1": {"domain": ["Items", "Bins"], "expression": {}},
        },
        "objective": {},
    },
}


def test_ir_fingerprint_lp_sense_and_model():
    fp = ir_fingerprint(_LP_EXAMPLE)
    assert "sense:minimize" in fp
    assert "model:lp" in fp


def test_ir_fingerprint_ordered_set():
    fp = ir_fingerprint(_LP_EXAMPLE)
    assert "has:ordered_set" in fp


def test_ir_fingerprint_var_dim():
    fp = ir_fingerprint(_LP_EXAMPLE)
    assert "var_dim:2" in fp


def test_ir_fingerprint_con_dim_multiple():
    fp = ir_fingerprint(_LP_EXAMPLE)
    assert "con_dim:1" in fp
    assert "con_dim:2" in fp


def test_ir_fingerprint_mip_model_type():
    fp = ir_fingerprint(_MIP_EXAMPLE)
    assert "model:mip" in fp
    assert "sense:maximize" in fp


def test_ir_fingerprint_variable_types():
    fp = ir_fingerprint(_MIP_EXAMPLE)
    assert "var_type:binary" in fp
    assert "var_type:continuous" in fp


def test_ir_fingerprint_domain_filter():
    fp = ir_fingerprint(_MIP_EXAMPLE)
    assert "has:domain_filter" in fp


def test_ir_fingerprint_no_exclude_diagonal():
    fp = ir_fingerprint(_MIP_EXAMPLE)
    assert "has:exclude_diagonal" not in fp


def test_ir_fingerprint_patterns_included():
    fp = ir_fingerprint(_MIP_EXAMPLE)
    assert "pat:binary variable" in fp
    assert "pat:big-M linking" in fp


def test_ir_fingerprint_lag_detection():
    example = {
        "modeling_patterns": [],
        "ir": {
            "sense": "minimize",
            "model_type": "Linear Program",
            "sets": {},
            "variables": {},
            "constraints": {
                "c1": {
                    "domain": ["Periods"],
                    "expression": {"type": "variable", "name": "inv", "lag": -1},
                },
            },
            "objective": {},
        },
    }
    fp = ir_fingerprint(example)
    assert "has:lag" in fp


def test_ir_fingerprint_n_sets_buckets():
    small = ir_fingerprint({**_LP_EXAMPLE, "ir": {**_LP_EXAMPLE["ir"], "sets": {"A": {}}}})
    large = ir_fingerprint({**_LP_EXAMPLE, "ir": {**_LP_EXAMPLE["ir"],
                            "sets": {str(i): {} for i in range(8)}}})
    assert "n_sets:small" in small
    assert "n_sets:large" in large


# ---------------------------------------------------------------------------
# detect_modeling_patterns
# ---------------------------------------------------------------------------

def test_detect_modeling_patterns_binary():
    fp = detect_modeling_patterns({"decision_variables": ["binary selection variable for each item"]})
    assert "pat:binary variable" in fp
    # detect_modeling_patterns only returns pat: tokens — no model:/sense: tokens
    assert all(t.startswith("pat:") for t in fp)


def test_detect_modeling_patterns_no_structural_tokens():
    fp = detect_modeling_patterns({"description": "Minimize total shipping cost."})
    assert all(t.startswith("pat:") for t in fp)


def test_detect_modeling_patterns_temporal():
    fp = detect_modeling_patterns({"description": "Inventory carries over between periods each month."})
    assert "pat:temporal lag" in fp


def test_detect_modeling_patterns_routing():
    fp = detect_modeling_patterns({
        "description": "Vehicles travel between different locations via arcs. "
                       "A vehicle cannot travel to itself (i ≠ j). "
                       "Subtour elimination (MTZ) is required."
    })
    assert "pat:exclude diagonal" in fp
    assert "pat:MTZ subtour elimination" in fp


def test_detect_modeling_patterns_works_with_all_fields():
    fp = detect_modeling_patterns({
        "title": "Production Planning",
        "description": "Minimize cost over planning periods.",
        "objective": "minimize",
        "objective_description": "Total production and inventory cost",
        "decision_variables": ["produce[p,t]: units produced", "inventory[p,t]: stock held"],
        "parameters": ["demand[p,t]", "holding cost"],
        "constraints": [{"description": "inventory balance each period"}],
    })
    assert "pat:temporal lag" in fp


def test_detect_modeling_patterns_empty():
    fp = detect_modeling_patterns({})
    assert isinstance(fp, set)
