"""Tests for on-demand IR builder prompt construction."""

from __future__ import annotations

from orpilot.workflow.nodes.ir_builder import _build_on_demand_user_content


def test_on_demand_prompt_includes_rag_as_secondary_reference():
    content = _build_on_demand_user_content(
        user_payload={
            "problem": {"title": "Facility Location"},
            "csv_schemas": {"sets": {"columns": {"set_name": "str"}}},
        },
        generated_code="def solve(data):\n    return {'status': 'optimal'}",
        rag_block="### Reference 1: facility_location\nIR pattern...",
    )

    assert "## On-Demand IR Generation From Working Solver Code" in content
    assert "## Retrieved IR Pattern References" in content
    assert "secondary references for IR idioms only" in content
    assert "working code" in content
    assert "authoritative" in content
    assert "facility_location" in content
    assert "def solve(data):" in content
    assert "Do not copy constraints, variables, or assumptions" in content


def test_on_demand_prompt_omits_rag_section_when_empty():
    content = _build_on_demand_user_content(
        user_payload={"problem": {"title": "Knapsack"}},
        generated_code="def solve(data):\n    return {'status': 'optimal'}",
        rag_block="",
    )

    assert "## Retrieved IR Pattern References" not in content
    assert "## Working Python Solver Code" in content
    assert "solver-agnostic ORPilot IR" in content
