"""Unit tests for codegen/data_validator.py — CSV data validation after collection."""

from __future__ import annotations

import pytest

from orpilot.codegen.data_validator import validate_collected_data
from orpilot.models.data import UserData


def _user_data(**tables) -> UserData:
    return UserData(raw_tables=tables)


# ---------------------------------------------------------------------------
# sets.csv presence and structure
# ---------------------------------------------------------------------------

def test_missing_sets_csv_flagged():
    ud = _user_data(costs=[{"from": "W1", "to": "C1", "cost": "5"}])
    errors = validate_collected_data(ud)
    assert any("sets.csv is missing" in e for e in errors)


def test_empty_sets_csv_flagged():
    ud = _user_data(sets=[])
    errors = validate_collected_data(ud)
    assert any("empty" in e.lower() for e in errors)


def test_sets_csv_missing_set_name_column_flagged():
    ud = _user_data(sets=[{"name": "W1", "element": "W1"}])
    errors = validate_collected_data(ud)
    assert any("set_name" in e for e in errors)


def test_sets_csv_missing_element_column_flagged():
    ud = _user_data(sets=[{"set_name": "Warehouses", "id": "W1"}])
    errors = validate_collected_data(ud)
    assert any("element" in e for e in errors)


# ---------------------------------------------------------------------------
# Valid data passes
# ---------------------------------------------------------------------------

def test_valid_data_passes():
    ud = _user_data(
        sets=[
            {"set_name": "Warehouses", "element": "W1"},
            {"set_name": "Warehouses", "element": "W2"},
            {"set_name": "Customers", "element": "C1"},
        ],
        supply=[{"warehouse_id": "W1", "supply": "100"}, {"warehouse_id": "W2", "supply": "80"}],
    )
    assert validate_collected_data(ud) == []


# ---------------------------------------------------------------------------
# Ambiguous element IDs (same ID in multiple sets)
# ---------------------------------------------------------------------------

def test_duplicate_element_across_sets_flagged():
    ud = _user_data(sets=[
        {"set_name": "Warehouses", "element": "node_1"},
        {"set_name": "Customers", "element": "node_1"},
    ])
    errors = validate_collected_data(ud)
    assert any("node_1" in e for e in errors)


def test_unique_elements_across_sets_ok():
    ud = _user_data(sets=[
        {"set_name": "Warehouses", "element": "W1"},
        {"set_name": "Customers", "element": "C1"},
    ])
    assert validate_collected_data(ud) == []


# ---------------------------------------------------------------------------
# Mixed entity column detection
# ---------------------------------------------------------------------------

def test_mixed_entity_column_flagged():
    """A parameter column that contains IDs from two different sets is flagged."""
    ud = _user_data(
        sets=[
            {"set_name": "Warehouses", "element": "W1"},
            {"set_name": "Customers", "element": "C1"},
        ],
        flows=[
            # 'node' column mixes Warehouses and Customers IDs
            {"node": "W1", "volume": "10"},
            {"node": "C1", "volume": "5"},
        ],
    )
    errors = validate_collected_data(ud)
    assert any("mixes" in e.lower() for e in errors)


def test_single_entity_column_ok():
    """A column containing only IDs from one set is fine."""
    ud = _user_data(
        sets=[
            {"set_name": "Warehouses", "element": "W1"},
            {"set_name": "Warehouses", "element": "W2"},
            {"set_name": "Customers", "element": "C1"},
        ],
        supply=[
            {"warehouse_id": "W1", "supply": "100"},
            {"warehouse_id": "W2", "supply": "80"},
        ],
    )
    assert validate_collected_data(ud) == []


def test_value_columns_not_flagged():
    """Numeric value columns never contain entity IDs and must not be flagged."""
    ud = _user_data(
        sets=[
            {"set_name": "Items", "element": "A"},
            {"set_name": "Items", "element": "B"},
        ],
        items=[
            {"item_id": "A", "value": "5", "weight": "3"},
            {"item_id": "B", "value": "8", "weight": "2"},
        ],
    )
    assert validate_collected_data(ud) == []
