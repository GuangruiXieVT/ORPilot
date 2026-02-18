"""Tests for the code executor."""

from orpilot.codegen.executor import CodeExecutor


def test_executor_simple():
    executor = CodeExecutor(timeout=10)
    code = '''
def solve(data):
    x = data.get("x", 1)
    y = data.get("y", 2)
    return {"status": "optimal", "objective_value": x + y, "variables": {"x": x, "y": y}}
'''
    result = executor.execute(code, {"x": 3, "y": 4})
    assert result["error"] is None
    assert result["result"]["status"] == "optimal"
    assert result["result"]["objective_value"] == 7


def test_executor_error():
    executor = CodeExecutor(timeout=10)
    code = '''
def solve(data):
    raise ValueError("something went wrong")
'''
    result = executor.execute(code, {})
    assert result["result"]["status"] == "error"


def test_executor_timeout():
    executor = CodeExecutor(timeout=2)
    code = '''
import time
def solve(data):
    time.sleep(10)
    return {"status": "optimal"}
'''
    result = executor.execute(code, {})
    assert result["error"] is not None
    assert "timed out" in result["error"].lower()


def test_executor_syntax_error():
    executor = CodeExecutor(timeout=5)
    code = "def solve(data):\n    return {invalid syntax"
    result = executor.execute(code, {})
    assert result["error"] is not None
