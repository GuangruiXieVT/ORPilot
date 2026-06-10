"""Unit tests for expand_synonyms in orpilot.rag.indexer."""

from __future__ import annotations

import pytest

from orpilot.rag.indexer import expand_synonyms


def _has(text: str, term: str) -> bool:
    return term.lower() in text.lower()


# ---------------------------------------------------------------------------
# Facility types
# ---------------------------------------------------------------------------

def test_warehouse_expands_to_distribution_center():
    out = expand_synonyms("store products in a warehouse")
    assert _has(out, "distribution center")
    assert _has(out, "fulfillment center")


def test_distribution_center_expands_to_warehouse():
    out = expand_synonyms("ship from a distribution center to stores")
    assert _has(out, "warehouse")


def test_fulfillment_center_expands_to_warehouse():
    out = expand_synonyms("orders ship from a fulfillment center")
    assert _has(out, "warehouse")


def test_plant_expands_to_factory():
    out = expand_synonyms("goods manufactured at each plant")
    assert _has(out, "factory")
    assert _has(out, "production site")


def test_factory_expands_to_plant():
    out = expand_synonyms("the factory produces components")
    assert _has(out, "plant")


def test_manufacturing_facility_expands_to_plant():
    out = expand_synonyms("each manufacturing facility has capacity")
    assert _has(out, "plant")
    assert _has(out, "factory")


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------

def test_supplier_expands_to_vendor():
    out = expand_synonyms("cost from each supplier")
    assert _has(out, "vendor")


def test_vendor_expands_to_supplier():
    out = expand_synonyms("purchase from external vendors")
    assert _has(out, "supplier")


# ---------------------------------------------------------------------------
# Costs
# ---------------------------------------------------------------------------

def test_holding_cost_expands_to_carrying_cost():
    out = expand_synonyms("minimize holding cost per unit per period")
    assert _has(out, "carrying cost")
    assert _has(out, "storage cost")


def test_carrying_cost_expands_to_holding_cost():
    out = expand_synonyms("reduce carrying cost in inventory")
    assert _has(out, "holding cost")


def test_transportation_cost_expands_to_shipping_cost():
    out = expand_synonyms("minimize total transportation cost")
    assert _has(out, "shipping cost")
    assert _has(out, "freight cost")


def test_shipping_cost_expands_to_transportation_cost():
    out = expand_synonyms("per-unit shipping cost between nodes")
    assert _has(out, "transportation cost")


def test_setup_cost_expands_to_changeover_cost():
    out = expand_synonyms("fixed setup cost per production run")
    assert _has(out, "changeover cost")


def test_changeover_cost_expands_to_setup_cost():
    out = expand_synonyms("incur a changeover cost when switching")
    assert _has(out, "setup cost")


# ---------------------------------------------------------------------------
# Facility status
# ---------------------------------------------------------------------------

def test_open_expands_to_activate():
    out = expand_synonyms("choose which facilities to open")
    assert _has(out, "activate")


def test_activate_expands_to_open():
    out = expand_synonyms("decision to activate a site")
    assert _has(out, "open")


def test_shut_down_expands_to_close():
    out = expand_synonyms("cost to shut down a facility")
    assert _has(out, "close")
    assert _has(out, "deactivate")


def test_deactivate_expands_to_close():
    out = expand_synonyms("deactivation of production sites")
    assert _has(out, "close")


# ---------------------------------------------------------------------------
# Inventory states
# ---------------------------------------------------------------------------

def test_backlog_expands_to_backorder():
    out = expand_synonyms("allow demand backlog each period")
    assert _has(out, "backorder")


def test_backorder_expands_to_backlog():
    out = expand_synonyms("penalize backorder quantities")
    assert _has(out, "backlog")


def test_safety_stock_expands_to_buffer_stock():
    out = expand_synonyms("maintain minimum safety stock level")
    assert _has(out, "buffer stock")
    assert _has(out, "safety inventory")


# ---------------------------------------------------------------------------
# No false expansion: unrelated text unchanged
# ---------------------------------------------------------------------------

def test_no_expansion_when_no_match():
    text = "minimize total production cost subject to demand constraints"
    out = expand_synonyms(text)
    # No synonym groups should fire — output is just the original text
    assert out == text


def test_original_text_preserved():
    text = "store inventory in a warehouse each period"
    out = expand_synonyms(text)
    assert out.startswith(text)


def test_already_present_term_not_duplicated():
    import re
    # Both "warehouse" and "distribution center" already present
    text = "ship from warehouse to distribution center"
    out = expand_synonyms(text)
    # The exact terms that were already in the text must not be appended again
    assert len(re.findall(r"\bwarehouse\b", out, re.IGNORECASE)) == 1
    assert len(re.findall(r"\bdistribution center\b", out, re.IGNORECASE)) == 1
