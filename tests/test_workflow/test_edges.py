"""Tests for workflow edge routing logic."""

from aifor.models.problem import ProblemDefinition
from aifor.models.data import UserData
from aifor.models.solution import SolutionResult, SolveStatus
from aifor.workflow.edges import after_interview, after_data_collection, after_solver_runner


def test_after_interview_complete():
    state = {"problem": ProblemDefinition(title="test")}
    assert after_interview(state) == "data_collection"


def test_after_interview_incomplete():
    state = {"problem": None}
    assert after_interview(state) == "wait_for_input"


def test_after_data_collection_complete():
    state = {"user_data": UserData()}
    assert after_data_collection(state) == "model_builder"


def test_after_data_collection_incomplete():
    state = {"user_data": None}
    assert after_data_collection(state) == "wait_for_input"


def test_after_solver_optimal():
    state = {"solution": SolutionResult(status=SolveStatus.OPTIMAL)}
    assert after_solver_runner(state) == "reporter"


def test_after_solver_error_with_retries():
    state = {
        "solution": SolutionResult(status=SolveStatus.ERROR),
        "retry_count": 1,
        "max_retries": 3,
    }
    assert after_solver_runner(state) == "model_builder"


def test_after_solver_error_exhausted():
    state = {
        "solution": SolutionResult(status=SolveStatus.ERROR),
        "retry_count": 3,
        "max_retries": 3,
    }
    assert after_solver_runner(state) == "reporter"
