"""Tests for workflow edge routing logic."""

from orpilot.models.problem import ProblemDefinition
from orpilot.models.data import UserData
from orpilot.models.solution import SolutionResult, SolveStatus
from orpilot.workflow.edges import (
    after_interview,
    after_data_collection,
    after_param_computation,
    after_solver_runner,
    after_solution_validator,
)


def test_after_interview_complete():
    state = {"problem": ProblemDefinition(title="test")}
    assert after_interview(state) == "data_collection"


def test_after_interview_incomplete():
    state = {"problem": None}
    assert after_interview(state) == "wait_for_input"


def test_after_data_collection_complete():
    state = {"user_data": UserData()}
    assert after_data_collection(state) == "param_computation"


def test_after_data_collection_incomplete():
    state = {"user_data": None}
    assert after_data_collection(state) == "wait_for_input"


def test_after_param_computation_defaults_to_direct():
    state = {}
    assert after_param_computation(state) == "direct_code_gen"


def test_after_param_computation_ir_mode():
    state = {"mode": "ir"}
    assert after_param_computation(state) == "ir_builder"


def test_after_solver_optimal():
    state = {"solution": SolutionResult(status=SolveStatus.OPTIMAL)}
    assert after_solver_runner(state) == "solution_validator"


def test_after_solver_error_with_retries():
    state = {
        "solution": SolutionResult(status=SolveStatus.ERROR),
        "retry_count": 1,
        "max_retries": 3,
    }
    assert after_solver_runner(state) == "direct_code_gen"


def test_after_solver_error_with_retries_ir_mode():
    state = {
        "solution": SolutionResult(status=SolveStatus.ERROR),
        "retry_count": 1,
        "max_retries": 3,
        "mode": "ir",
    }
    assert after_solver_runner(state) == "ir_builder"


def test_after_solver_error_exhausted():
    state = {
        "solution": SolutionResult(status=SolveStatus.ERROR),
        "retry_count": 3,
        "max_retries": 3,
    }
    assert after_solver_runner(state) == "reporter"


def test_after_solution_validator_pass():
    state = {"error_context": "", "validation_attempts": 0}
    assert after_solution_validator(state) == "reporter"


def test_after_solution_validator_fail_retries():
    state = {"error_context": "VALIDATION FAILED: ...", "validation_attempts": 1}
    assert after_solution_validator(state) == "direct_code_gen"


def test_after_solution_validator_fail_retries_ir_mode():
    state = {"error_context": "VALIDATION FAILED: ...", "validation_attempts": 1, "mode": "ir"}
    assert after_solution_validator(state) == "ir_builder"


def test_after_solution_validator_pass_generates_ir_after_direct_validation():
    state = {"error_context": "", "validation_attempts": 0, "generate_ir": True}
    assert after_solution_validator(state) == "ir_builder_on_demand"


def test_after_solution_validator_fail_exhausted():
    state = {"error_context": "VALIDATION FAILED: ...", "validation_attempts": 2}
    assert after_solution_validator(state) == "reporter"
