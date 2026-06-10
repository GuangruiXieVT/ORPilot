"""Tests for the solution validator node.

Three scenarios with semantically wrong IRs that still produce an "optimal" solution:

  A. FIXED-COST-PER-PERIOD (objective error)
     Wrong IR charges a one-time facility-opening cost every period.
     Solver reports objective = T × fixed_cost + production_cost.
     Correct computation = 1 × fixed_cost + production_cost.
     → objective mismatch detected; no constraint violation.

  B. PER-PRODUCT CAPACITY (constraint error)
     Wrong IR applies a machine-capacity limit per product instead of to the
     total production sum.  The solution respects individual limits but
     exceeds the shared machine capacity.
     Objective values match because the revenue formula is correct.
     → constraint violation detected; no objective error.

  C. BOTH ERRORS PRESENT
     Fixed-cost-per-period + per-product capacity in the same solution.
     → both objective mismatch and constraint violation detected.

Tests are structured in four tiers:
  Tier 1 — _check_result unit tests (dict → error message, no subprocess)
  Tier 2 — _execute_script tests (hand-written scripts, real subprocess, no LLM)
  Tier 3 — solution_validator_node integration tests (mocked LLM)
  Tier 4 — live LLM tests (real API call; run with pytest --llm)
"""

from __future__ import annotations

import textwrap
from unittest.mock import MagicMock

import pytest

from orpilot.models.data import UserData
from orpilot.models.problem import ProblemDefinition
from orpilot.models.solution import SolutionResult, SolveStatus, VariableGroup
from orpilot.workflow.nodes.solution_validator import (
    _check_result,
    _execute_script,
    solution_validator_node,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_llm(script: str) -> MagicMock:
    """Return a mock LLM whose chat() always returns the given script."""
    llm = MagicMock()
    llm.chat = MagicMock(return_value=script)
    llm.reset_usage = MagicMock()
    llm.get_usage = MagicMock(return_value={"input_tokens": 0, "output_tokens": 0})
    return llm


SEP = "\x1f"

# ---------------------------------------------------------------------------
# Scenario A helpers — fixed cost per period (objective error)
# ---------------------------------------------------------------------------
#
# Facility A, periods P1/P2/P3.  One-time opening cost = $1000.
# Wrong IR: fixed_cost[f] × open[f,t] → charges $1000 each period.
# Solution: open[A,P1]=1, open[A,P2]=1, open[A,P3]=1, production[A,P1]=100
# Wrong reported obj = 3 × 1000 + 100 × 5 = 3500
# Correct obj        = 1 × 1000 + 100 × 5 = 1500  → diff ≈ 133%

_SCENARIO_A_SOLUTION = SolutionResult(
    status=SolveStatus.OPTIMAL,
    objective_value=3500.0,
    variable_groups=[
        VariableGroup(
            group_name="open",
            dimension_labels=["facility", "period"],
            variables={
                f"open{SEP}A{SEP}P1": 1.0,
                f"open{SEP}A{SEP}P2": 1.0,
                f"open{SEP}A{SEP}P3": 1.0,
            },
        ),
        VariableGroup(
            group_name="production",
            dimension_labels=["facility", "period"],
            variables={
                f"production{SEP}A{SEP}P1": 100.0,
                f"production{SEP}A{SEP}P2": 0.0,
                f"production{SEP}A{SEP}P3": 0.0,
            },
        ),
    ],
)

_SCENARIO_A_USER_DATA = UserData(
    raw_tables={
        "facilities": [{"facility": "A", "fixed_cost": 1000.0, "prod_cost": 5.0}],
        "periods": [{"period": "P1"}, {"period": "P2"}, {"period": "P3"}],
    }
)

# Validation script the LLM *should* write for scenario A:
# implements one-time fixed cost (deduplicate open by facility).
_SCENARIO_A_SCRIPT = textwrap.dedent("""\
    import json
    with open("data.json") as f:
        data = json.load(f)
    input_data = data["input"]
    sol_data = data["solution"]
    reported_obj = data["reported_obj"]
    TOL = 1e-4

    facilities = {r["facility"]: r for r in input_data.get("facilities", [])}

    # One-time: facility counts if ever opened in any period
    facility_ever_open = {}
    for r in sol_data.get("open", []):
        if r["value"] > 0.5:
            facility_ever_open[r["facility"]] = 1.0

    production = {(r["facility"], r["period"]): r["value"]
                  for r in sol_data.get("production", [])}

    term_fixed = sum(facilities[f]["fixed_cost"] * v
                     for f, v in facility_ever_open.items())
    term_prod = sum(facilities[f]["prod_cost"] * v
                    for (f, _p), v in production.items())
    computed_obj = term_fixed + term_prod

    violations = []
    print("__ORPILOT_VALIDATION__:" + json.dumps({
        "computed_obj": computed_obj,
        "reported_obj": reported_obj,
        "terms": {"fixed_cost": term_fixed, "production_cost": term_prod},
        "violations": violations,
        "notes": "One-time fixed cost must not be multiplied by number of periods open",
    }))
""")


# ---------------------------------------------------------------------------
# Scenario B helpers — per-product capacity (constraint error)
# ---------------------------------------------------------------------------
#
# 2 products (X, Y), shared machine capacity = 10.
# Wrong IR: production[p] ≤ capacity per product → allows total > 10.
# Solution: X=8, Y=7 → total=15 > 10  (each individually ≤ 10, so wrong IR OK).
# Prices: X=$5, Y=$6 → reported obj = 8*5 + 7*6 = 82 (revenue, no fixed cost).
# Correct objective = same 82; only the capacity constraint is violated.

_SCENARIO_B_SOLUTION = SolutionResult(
    status=SolveStatus.OPTIMAL,
    objective_value=82.0,
    variable_groups=[
        VariableGroup(
            group_name="production",
            dimension_labels=["product"],
            variables={
                f"production{SEP}X": 8.0,
                f"production{SEP}Y": 7.0,
            },
        ),
    ],
)

_SCENARIO_B_USER_DATA = UserData(
    raw_tables={
        "products": [
            {"product": "X", "price": 5.0},
            {"product": "Y", "price": 6.0},
        ],
        "machine": [{"capacity": 10.0}],
    }
)

_SCENARIO_B_SCRIPT = textwrap.dedent("""\
    import json
    with open("data.json") as f:
        data = json.load(f)
    input_data = data["input"]
    sol_data = data["solution"]
    reported_obj = data["reported_obj"]
    TOL = 1e-4

    prices = {r["product"]: r["price"] for r in input_data.get("products", [])}
    capacity = input_data.get("machine", [{}])[0].get("capacity", 0)

    production = {r["product"]: r["value"]
                  for r in sol_data.get("production", [])}

    # Objective: maximize revenue
    term_revenue = sum(prices.get(p, 0) * v for p, v in production.items())
    computed_obj = term_revenue

    violations = []
    # Shared machine capacity: sum of all production <= capacity
    total_prod = sum(production.values())
    if total_prod > capacity + TOL:
        violations.append({
            "constraint": "machine_capacity",
            "index": "machine",
            "lhs": total_prod,
            "rhs": capacity,
        })

    print("__ORPILOT_VALIDATION__:" + json.dumps({
        "computed_obj": computed_obj,
        "reported_obj": reported_obj,
        "terms": {"revenue": term_revenue},
        "violations": violations,
        "notes": "Machine capacity applies to total production, not per product",
    }))
""")


# ---------------------------------------------------------------------------
# Tier 1: _check_result unit tests
# ---------------------------------------------------------------------------

class TestCheckResult:
    def test_objective_mismatch_only(self):
        result = {
            "computed_obj": 1500.0,
            "reported_obj": 3500.0,
            "terms": {"fixed_cost": 1000.0, "production_cost": 500.0},
            "violations": [],
            "notes": "fixed cost charged per period instead of once",
        }
        error = _check_result(result, 3500.0)
        assert error is not None
        assert "Objective mismatch" in error
        assert "Constraint violations" not in error
        # Term breakdown present for diagnosis
        assert "fixed_cost" in error
        assert "production_cost" in error

    def test_constraint_violation_only(self):
        result = {
            "computed_obj": 82.0,
            "reported_obj": 82.0,  # objective is correct
            "terms": {"revenue": 82.0},
            "violations": [
                {"constraint": "machine_capacity", "index": "machine", "lhs": 15.0, "rhs": 10.0}
            ],
            "notes": "",
        }
        error = _check_result(result, 82.0)
        assert error is not None
        assert "Constraint violations" in error
        assert "machine_capacity" in error
        assert "Objective mismatch" not in error

    def test_both_errors_reported(self):
        result = {
            "computed_obj": 1500.0,
            "reported_obj": 3500.0,
            "terms": {"fixed_cost": 1000.0, "production_cost": 500.0},
            "violations": [
                {"constraint": "machine_capacity", "index": "machine", "lhs": 15.0, "rhs": 10.0}
            ],
            "notes": "",
        }
        error = _check_result(result, 3500.0)
        assert error is not None
        assert "Objective mismatch" in error
        assert "Constraint violations" in error

    def test_pass_returns_none(self):
        result = {
            "computed_obj": 1500.0,
            "reported_obj": 1500.0,
            "terms": {"fixed_cost": 1000.0, "production_cost": 500.0},
            "violations": [],
            "notes": "",
        }
        assert _check_result(result, 1500.0) is None

    def test_within_one_percent_tolerance(self):
        # 0.5% difference — should pass
        result = {
            "computed_obj": 1507.5,
            "reported_obj": 1500.0,
            "terms": {"cost": 1507.5},
            "violations": [],
            "notes": "",
        }
        assert _check_result(result, 1500.0) is None

    def test_notes_included_in_error(self):
        result = {
            "computed_obj": 1500.0,
            "reported_obj": 3500.0,
            "terms": {},
            "violations": [],
            "notes": "fixed cost is charged per period, should be one-time",
        }
        error = _check_result(result, 3500.0)
        assert "fixed cost is charged per period" in error


# ---------------------------------------------------------------------------
# Tier 2: _execute_script tests (real subprocess, no LLM)
# ---------------------------------------------------------------------------

class TestExecuteScript:
    """These tests run real Python subprocesses with hand-written scripts.
    They verify that the script execution machinery works and that the
    hand-written "correct" scripts actually catch the semantic errors.
    """

    def test_scenario_a_script_detects_objective_mismatch(self):
        """Fixed-cost-per-period script: computed obj = 1500, reported = 3500."""
        from orpilot.workflow.nodes.solution_validator import _build_data_bundle
        bundle = _build_data_bundle(_SCENARIO_A_SOLUTION, _SCENARIO_A_USER_DATA)

        result = _execute_script(_SCENARIO_A_SCRIPT, bundle)

        assert result is not None
        assert "error" not in result
        assert abs(result["computed_obj"] - 1500.0) < 1e-6
        assert abs(result["reported_obj"] - 3500.0) < 1e-6
        assert result["violations"] == []
        # _check_result should flag this
        error = _check_result(result, 3500.0)
        assert error is not None
        assert "Objective mismatch" in error
        assert "Constraint violations" not in error

    def test_scenario_b_script_detects_constraint_violation(self):
        """Per-product capacity script: obj matches, total production 15 > cap 10."""
        from orpilot.workflow.nodes.solution_validator import _build_data_bundle
        bundle = _build_data_bundle(_SCENARIO_B_SOLUTION, _SCENARIO_B_USER_DATA)

        result = _execute_script(_SCENARIO_B_SCRIPT, bundle)

        assert result is not None
        assert "error" not in result
        assert abs(result["computed_obj"] - 82.0) < 1e-6
        assert abs(result["reported_obj"] - 82.0) < 1e-6
        assert len(result["violations"]) == 1
        assert result["violations"][0]["constraint"] == "machine_capacity"
        # _check_result should flag this
        error = _check_result(result, 82.0)
        assert error is not None
        assert "Constraint violations" in error
        assert "Objective mismatch" not in error

    def test_correct_solution_passes(self):
        """When the solution is genuinely correct the script returns no issues."""
        # Solution that satisfies both correct objective and constraints:
        # open[A] once, produce 100, reported obj = 1000 + 100*5 = 1500
        correct_solution = SolutionResult(
            status=SolveStatus.OPTIMAL,
            objective_value=1500.0,
            variable_groups=[
                VariableGroup(
                    group_name="open",
                    dimension_labels=["facility", "period"],
                    variables={f"open{SEP}A{SEP}P1": 1.0},
                ),
                VariableGroup(
                    group_name="production",
                    dimension_labels=["facility", "period"],
                    variables={f"production{SEP}A{SEP}P1": 100.0},
                ),
            ],
        )
        from orpilot.workflow.nodes.solution_validator import _build_data_bundle
        bundle = _build_data_bundle(correct_solution, _SCENARIO_A_USER_DATA)

        result = _execute_script(_SCENARIO_A_SCRIPT, bundle)

        assert result is not None
        assert "error" not in result
        assert abs(result["computed_obj"] - 1500.0) < 1e-6
        assert result["violations"] == []
        assert _check_result(result, 1500.0) is None

    def test_broken_script_returns_error_dict(self):
        bad_script = "this is not valid Python ???"
        result = _execute_script(bad_script, {"input": {}, "solution": {}, "reported_obj": 0.0})
        assert result is not None
        assert "error" in result

    def test_script_missing_marker_returns_error(self):
        script = "x = 1 + 1\nprint('done')"
        result = _execute_script(script, {"input": {}, "solution": {}, "reported_obj": 0.0})
        assert result is not None
        assert "error" in result


# ---------------------------------------------------------------------------
# Tier 3: solution_validator_node integration tests (mocked LLM)
# ---------------------------------------------------------------------------

def _base_state(solution: SolutionResult, user_data: UserData, ir: dict) -> dict:
    return {
        "ir_model": ir,
        "solution": solution,
        "user_data": user_data,
        "problem": ProblemDefinition(
            title="Test problem",
            description="Minimize cost. Fixed facility opening cost is a ONE-TIME cost.",
        ),
        "retry_count": 0,
        "validation_attempts": 0,
    }


# Minimal placeholder IR dicts — the node only uses these to pass context to
# the LLM prompt; since the LLM is mocked the exact content doesn't matter.
_IR_WRONG_FIXED_COST = {
    "sets": {"Facilities": {}, "Periods": {}},
    "variables": {"open": {"domain": ["Facilities", "Periods"]},
                  "production": {"domain": ["Facilities", "Periods"]}},
    "objective": {"sense": "minimize", "expression": {}},
    "constraints": {},
}

_IR_WRONG_CAPACITY = {
    "sets": {"Products": {}},
    "variables": {"production": {"domain": ["Products"]}},
    "objective": {"sense": "maximize", "expression": {}},
    "constraints": {},
}


class TestSolutionValidatorNode:
    def test_scenario_a_node_sets_objective_error(self):
        """Node with mocked LLM returning correct one-time-cost script detects obj mismatch."""
        llm = _make_llm(_SCENARIO_A_SCRIPT)
        state = _base_state(_SCENARIO_A_SOLUTION, _SCENARIO_A_USER_DATA, _IR_WRONG_FIXED_COST)

        result = solution_validator_node(state, llm)

        assert result["current_node"] == "solution_validator"
        assert result.get("error_context"), "expected error_context to be set"
        assert "Objective mismatch" in result["error_context"]
        assert "Constraint violations" not in result["error_context"]
        assert result.get("validation_attempts", 0) == 1
        assert result.get("retry_count", 0) == 1

    def test_scenario_b_node_sets_constraint_error(self):
        """Node with mocked LLM returning shared-capacity script detects constraint violation."""
        llm = _make_llm(_SCENARIO_B_SCRIPT)
        state = _base_state(_SCENARIO_B_SOLUTION, _SCENARIO_B_USER_DATA, _IR_WRONG_CAPACITY)

        result = solution_validator_node(state, llm)

        assert result["current_node"] == "solution_validator"
        assert result.get("error_context"), "expected error_context to be set"
        assert "Constraint violations" in result["error_context"]
        assert "machine_capacity" in result["error_context"]
        assert "Objective mismatch" not in result["error_context"]
        assert result.get("validation_attempts", 0) == 1

    def test_correct_solution_passes_through(self):
        """Node reports no error when the solution is genuinely correct."""
        # One period, one-time open, correct obj
        correct_solution = SolutionResult(
            status=SolveStatus.OPTIMAL,
            objective_value=1500.0,
            variable_groups=[
                VariableGroup(
                    group_name="open",
                    dimension_labels=["facility", "period"],
                    variables={f"open{SEP}A{SEP}P1": 1.0},
                ),
                VariableGroup(
                    group_name="production",
                    dimension_labels=["facility", "period"],
                    variables={f"production{SEP}A{SEP}P1": 100.0},
                ),
            ],
        )
        llm = _make_llm(_SCENARIO_A_SCRIPT)
        state = _base_state(correct_solution, _SCENARIO_A_USER_DATA, _IR_WRONG_FIXED_COST)

        result = solution_validator_node(state, llm)

        assert result["current_node"] == "solution_validator"
        assert not result.get("error_context")
        assert result.get("validation_attempts", 0) == 0

    def test_validates_when_no_ir_model(self):
        """Node validates direct_code_gen solutions even when no IR is available."""
        llm = _make_llm(_SCENARIO_A_SCRIPT)
        state = {
            "ir_model": None,
            "solution": _SCENARIO_A_SOLUTION,
            "user_data": _SCENARIO_A_USER_DATA,
            "problem": ProblemDefinition(title="Test"),
        }

        result = solution_validator_node(state, llm)

        assert result["current_node"] == "solution_validator"
        assert result.get("error_context")
        assert "Objective mismatch" in result["error_context"]
        llm.chat.assert_called()

    def test_script_failure_does_not_block(self):
        """When the generated script crashes, validation is inconclusive and does not block."""
        llm = _make_llm("this is not valid Python ???")
        state = _base_state(_SCENARIO_A_SOLUTION, _SCENARIO_A_USER_DATA, _IR_WRONG_FIXED_COST)

        result = solution_validator_node(state, llm)

        # Should NOT set error_context — inconclusive means we proceed
        assert not result.get("error_context")
        assert result["current_node"] == "solution_validator"


# ---------------------------------------------------------------------------
# Tier 4: live LLM tests (real API call via orpilot.toml)
# ---------------------------------------------------------------------------
#
# Run with:  pytest tests/test_workflow/test_solution_validator.py --llm -v -s
#
# These tests let the real LLM write the validation script from the prompt and
# scenario data.  They verify that the LLM correctly picks up on semantic cues
# in the problem description ("ONE-TIME cost", "shared machine capacity") and
# generates code that catches the modeling error.
#
# Problem descriptions are made deliberately explicit so even a weaker model
# can succeed.  The assertions are intentionally loose — we only check that
# the right error type is flagged, not the exact wording.

@pytest.fixture(scope="module")
def live_llm(request):
    """Build a real LLM from orpilot.toml; skip the test if the key is absent."""
    if not request.config.getoption("--llm"):
        pytest.skip("pass --llm to run live LLM tests")

    from pathlib import Path
    from orpilot.config import discover_config_file, load_config_file
    from orpilot.llm.config import LLMConfig, get_llm

    config_path = discover_config_file()
    if config_path is None:
        pytest.skip("No orpilot.toml found")

    raw = load_config_file(config_path)
    api_key = raw.get("api_key", "")
    if not api_key or api_key.startswith("your_"):
        pytest.skip("api_key not configured in orpilot.toml (still the placeholder)")

    llm_config = LLMConfig(
        provider=raw.get("provider", "openai"),
        model=raw.get("model"),
        api_key=api_key,
        base_url=raw.get("base_url"),
        temperature=raw.get("temperature", 0.0),
        max_tokens=raw.get("max_tokens", 8192),
    )
    return get_llm(llm_config)


def _state_with_description(solution, user_data, ir, description):
    """Like _base_state but with an explicit problem description."""
    return {
        "ir_model": ir,
        "solution": solution,
        "user_data": user_data,
        "problem": ProblemDefinition(title="Test problem", description=description),
        "retry_count": 0,
        "validation_attempts": 0,
    }


# ---------------------------------------------------------------------------
# Scenario C data — both objective and constraint errors
# ---------------------------------------------------------------------------
#
# Facility A, 3 periods (P1, P2, P3).  Same data structure as Scenario A but
# with a per-period machine capacity (50 units) added to the facilities table.
#
# Wrong IR:
#   1. Fixed cost charged every period (should be one-time).
#   2. Capacity enforced as a cumulative total (sum ≤ 150) instead of per-period
#      (production[f,t] ≤ 50).  The solution over-produces in P1 to exploit this.
#
# Solution: open all 3 periods, production[A,P1]=80, production[A,P2]=30.
#   - Wrong reported obj = 3×1000 + (80+30)×5 = 3550.
#   - Correct obj        = 1×1000 + 110×5      = 1550  (one-time fixed cost).
#   - Capacity violation: production[A,P1]=80 > per-period limit 50.

_SCENARIO_C_SOLUTION = SolutionResult(
    status=SolveStatus.OPTIMAL,
    objective_value=3550.0,
    variable_groups=[
        VariableGroup(
            group_name="open",
            dimension_labels=["facility", "period"],
            variables={
                f"open{SEP}A{SEP}P1": 1.0,
                f"open{SEP}A{SEP}P2": 1.0,
                f"open{SEP}A{SEP}P3": 1.0,
            },
        ),
        VariableGroup(
            group_name="production",
            dimension_labels=["facility", "period"],
            variables={
                f"production{SEP}A{SEP}P1": 80.0,
                f"production{SEP}A{SEP}P2": 30.0,
                f"production{SEP}A{SEP}P3": 0.0,
            },
        ),
    ],
)

_SCENARIO_C_USER_DATA = UserData(
    raw_tables={
        "facilities": [
            {"facility": "A", "fixed_cost": 1000.0, "prod_cost": 5.0, "capacity": 50.0}
        ],
        "periods": [{"period": "P1"}, {"period": "P2"}, {"period": "P3"}],
    }
)

_IR_WRONG_BOTH = {
    "sets": {"Facilities": {}, "Periods": {}},
    "variables": {
        "open": {"domain": ["Facilities", "Periods"]},
        "production": {"domain": ["Facilities", "Periods"]},
    },
    "objective": {"sense": "minimize", "expression": {}},
    "constraints": {},
}

# ---------------------------------------------------------------------------
# Scenario D — correct multi-period solution (no false positive)
# ---------------------------------------------------------------------------
#
# Facility open for 2 periods, fixed cost charged ONCE.
# production[A,P1]=50, production[A,P2]=50.
# Correct reported obj = 1000 + (50+50)×5 = 1500.

_SCENARIO_D_SOLUTION = SolutionResult(
    status=SolveStatus.OPTIMAL,
    objective_value=1500.0,
    variable_groups=[
        VariableGroup(
            group_name="open",
            dimension_labels=["facility", "period"],
            variables={
                f"open{SEP}A{SEP}P1": 1.0,
                f"open{SEP}A{SEP}P2": 1.0,
            },
        ),
        VariableGroup(
            group_name="production",
            dimension_labels=["facility", "period"],
            variables={
                f"production{SEP}A{SEP}P1": 50.0,
                f"production{SEP}A{SEP}P2": 50.0,
            },
        ),
    ],
)

# ---------------------------------------------------------------------------
# Scenario E — correct capacity solution (no false positive)
# ---------------------------------------------------------------------------
#
# Products X=4, Y=5, total=9 ≤ capacity=10.
# Revenue = 4×5 + 5×6 = 50.

_SCENARIO_E_SOLUTION = SolutionResult(
    status=SolveStatus.OPTIMAL,
    objective_value=50.0,
    variable_groups=[
        VariableGroup(
            group_name="production",
            dimension_labels=["product"],
            variables={
                f"production{SEP}X": 4.0,
                f"production{SEP}Y": 5.0,
            },
        ),
    ],
)


@pytest.mark.llm
class TestSolutionValidatorNodeLive:
    """Live LLM tests — require --llm flag and a configured api_key in orpilot.toml."""

    def test_scenario_a_llm_detects_objective_error(self, live_llm):
        """LLM should read 'ONE-TIME opening cost' in the description and detect
        that the reported objective (charged 3× across periods) is too high."""
        description = (
            "Minimize total cost. "
            "Each facility has a ONE-TIME fixed opening cost paid only once "
            "regardless of how many periods it operates. "
            "There is also a per-unit production cost paid each period. "
            "Facility A has fixed_cost=1000 and prod_cost=5. "
            "The reported objective includes fixed costs charged every period, "
            "which is wrong — it should be charged once."
        )
        state = _state_with_description(
            _SCENARIO_A_SOLUTION, _SCENARIO_A_USER_DATA, _IR_WRONG_FIXED_COST, description
        )

        result = solution_validator_node(state, live_llm)

        assert result["current_node"] == "solution_validator"
        error = result.get("error_context", "")
        assert error, (
            "LLM failed to detect the objective error. "
            f"Node returned no error_context. "
            f"(computed_obj should be 1500, reported_obj is 3500)"
        )
        assert "Objective mismatch" in error, (
            f"Expected 'Objective mismatch' in error but got:\n{error}"
        )

    def test_scenario_b_llm_detects_constraint_violation(self, live_llm):
        """LLM should read 'shared machine capacity' and detect that total
        production (15) exceeds the machine limit (10)."""
        description = (
            "Maximize revenue from producing products X and Y. "
            "There is a single machine with a TOTAL capacity of 10 units "
            "shared across ALL products — the SUM of production quantities "
            "must not exceed 10. "
            "Product X yields $5/unit, product Y yields $6/unit. "
            "The solution has X=8 and Y=7 (total=15), which violates the "
            "shared machine capacity of 10."
        )
        state = _state_with_description(
            _SCENARIO_B_SOLUTION, _SCENARIO_B_USER_DATA, _IR_WRONG_CAPACITY, description
        )

        result = solution_validator_node(state, live_llm)

        assert result["current_node"] == "solution_validator"
        error = result.get("error_context", "")
        assert error, (
            "LLM failed to detect the constraint violation. "
            f"Node returned no error_context. "
            f"(total production 15 > capacity 10 should be flagged)"
        )
        assert "Constraint violations" in error, (
            f"Expected 'Constraint violations' in error but got:\n{error}"
        )

    def test_correct_solution_no_false_positive(self, live_llm):
        """LLM must NOT flag a solution that is genuinely correct.
        Facility open once for 1 period, reported obj = 1000 + 100*5 = 1500."""
        description = (
            "Minimize total cost. "
            "Each facility has a ONE-TIME fixed opening cost. "
            "There is also a per-unit production cost paid each period. "
            "Facility A has fixed_cost=1000 and prod_cost=5."
        )
        correct_solution = SolutionResult(
            status=SolveStatus.OPTIMAL,
            objective_value=1500.0,
            variable_groups=[
                VariableGroup(
                    group_name="open",
                    dimension_labels=["facility", "period"],
                    variables={f"open{SEP}A{SEP}P1": 1.0},
                ),
                VariableGroup(
                    group_name="production",
                    dimension_labels=["facility", "period"],
                    variables={f"production{SEP}A{SEP}P1": 100.0},
                ),
            ],
        )
        state = _state_with_description(
            correct_solution, _SCENARIO_A_USER_DATA, _IR_WRONG_FIXED_COST, description
        )

        result = solution_validator_node(state, live_llm)

        assert result["current_node"] == "solution_validator"
        assert not result.get("error_context"), (
            f"False positive: LLM flagged a correct solution.\n"
            f"error_context: {result.get('error_context')}"
        )

    def test_scenario_c_llm_detects_both_errors(self, live_llm):
        """LLM should flag BOTH an objective mismatch (fixed cost charged per period)
        AND a constraint violation (production in P1 exceeds the per-period capacity)."""
        description = (
            "Minimize total cost. "
            "Facility A has a ONE-TIME fixed opening cost of 1000, paid only once "
            "regardless of how many periods it operates. "
            "Production costs 5 per unit. "
            "The machine capacity is 50 units PER PERIOD — production in any single "
            "period must not exceed 50. "
            "The solution has the facility open for all 3 periods with "
            "production[A,P1]=80, production[A,P2]=30, production[A,P3]=0. "
            "The reported objective of 3550 charges the fixed cost each period (wrong); "
            "the correct one-time cost gives 1000 + 110×5 = 1550. "
            "Also, production[A,P1]=80 exceeds the per-period capacity of 50."
        )
        state = _state_with_description(
            _SCENARIO_C_SOLUTION, _SCENARIO_C_USER_DATA, _IR_WRONG_BOTH, description
        )

        result = solution_validator_node(state, live_llm)

        assert result["current_node"] == "solution_validator"
        error = result.get("error_context", "")
        assert error, "LLM failed to detect any error in scenario C"
        assert "Objective mismatch" in error, (
            f"Expected 'Objective mismatch' in error but got:\n{error}"
        )
        assert "Constraint violations" in error, (
            f"Expected 'Constraint violations' in error but got:\n{error}"
        )

    def test_multiperiod_correct_solution_no_false_positive(self, live_llm):
        """Facility open for 2 periods but fixed cost correctly charged once.
        Reported obj = 1000 + (50+50)×5 = 1500. LLM must not flag this."""
        description = (
            "Minimize total cost. "
            "The facility has a ONE-TIME fixed opening cost of 1000, paid once "
            "regardless of how many periods it is open. "
            "Production costs 5 per unit per period. "
            "The facility is open in periods P1 and P2, producing 50 units each. "
            "The reported objective of 1500 = 1000 (one-time) + 50×5 + 50×5 is correct."
        )
        state = _state_with_description(
            _SCENARIO_D_SOLUTION, _SCENARIO_A_USER_DATA, _IR_WRONG_FIXED_COST, description
        )

        result = solution_validator_node(state, live_llm)

        assert result["current_node"] == "solution_validator"
        assert not result.get("error_context"), (
            f"False positive on multi-period correct solution.\n"
            f"error_context: {result.get('error_context')}"
        )

    def test_within_capacity_no_false_positive(self, live_llm):
        """Products X=4 and Y=5 give total=9, within the shared capacity of 10.
        Revenue = 4×5 + 5×6 = 50. LLM must not flag any constraint violation."""
        description = (
            "Maximize revenue from producing products X and Y. "
            "There is a single machine with a TOTAL shared capacity of 10 units. "
            "The SUM of all production must not exceed 10. "
            "Product X yields 5 per unit, product Y yields 6 per unit. "
            "The solution produces X=4 and Y=5 (total=9 ≤ 10), giving revenue=50. "
            "This is a valid solution with no violations."
        )
        state = _state_with_description(
            _SCENARIO_E_SOLUTION, _SCENARIO_B_USER_DATA, _IR_WRONG_CAPACITY, description
        )

        result = solution_validator_node(state, live_llm)

        assert result["current_node"] == "solution_validator"
        assert not result.get("error_context"), (
            f"False positive on within-capacity solution.\n"
            f"error_context: {result.get('error_context')}"
        )
