"""Tests for workflow state and models."""

import os
import tempfile

from orpilot.models.problem import ProblemDefinition, ProblemType, ObjectiveType, Constraint
from orpilot.models.data import UserData, DataParameter, CsvFileSpec, CsvColumnSpec
from orpilot.models.solution import SolutionResult, SolveStatus


def test_problem_definition():
    problem = ProblemDefinition(
        title="Diet Problem",
        description="Find the cheapest diet that meets nutritional requirements",
        problem_type=ProblemType.LINEAR_PROGRAMMING,
        objective=ObjectiveType.MINIMIZE,
        objective_description="Total cost of food items",
        constraints=[
            Constraint(description="Minimum 2000 calories", expression="sum(cal_i * x_i) >= 2000"),
            Constraint(description="Maximum 65g fat", expression="sum(fat_i * x_i) <= 65"),
        ],
        decision_variables=["Amount of each food item to include"],
    )
    assert problem.title == "Diet Problem"
    assert len(problem.constraints) == 2
    assert problem.problem_type == ProblemType.LINEAR_PROGRAMMING


def test_user_data():
    data = UserData(
        parameters=[
            DataParameter(name="num_foods", value=3),
            DataParameter(name="budget", value=100.0),
        ],
        raw_tables={
            "foods": [
                {"name": "Corn", "cost": 0.18, "calories": 72},
                {"name": "Milk", "cost": 0.23, "calories": 121},
            ]
        },
    )
    flat = data.as_dict()
    assert flat["num_foods"] == 3
    assert flat["budget"] == 100.0
    assert len(flat["foods"]) == 2


def test_solution_result():
    sol = SolutionResult(
        status=SolveStatus.OPTIMAL,
        objective_value=42.5,
        variables={"x1": 3.0, "x2": 7.0},
        solve_time_seconds=0.05,
    )
    assert sol.status == SolveStatus.OPTIMAL
    assert sol.objective_value == 42.5


def test_solution_json_roundtrip():
    sol = SolutionResult(
        status=SolveStatus.INFEASIBLE,
        error_message="No feasible solution found",
    )
    json_str = sol.model_dump_json()
    restored = SolutionResult.model_validate_json(json_str)
    assert restored.status == SolveStatus.INFEASIBLE
    assert restored.error_message == "No feasible solution found"


def test_csv_file_spec_model():
    spec = CsvFileSpec(
        filename="costs.csv",
        description="Cost data for each item",
        columns=[
            CsvColumnSpec(name="item", dtype="str", description="Item name"),
            CsvColumnSpec(name="cost", dtype="float", description="Unit cost"),
        ],
    )
    assert spec.filename == "costs.csv"
    assert len(spec.columns) == 2
    assert spec.columns[0].name == "item"
    assert spec.columns[1].dtype == "float"

    # Roundtrip
    restored = CsvFileSpec.model_validate_json(spec.model_dump_json())
    assert restored.filename == spec.filename
    assert len(restored.columns) == 2


def test_load_from_csv_dir():
    specs = [
        CsvFileSpec(
            filename="foods.csv",
            description="Food items with nutritional info",
            columns=[
                CsvColumnSpec(name="name", dtype="str", description="Food name"),
                CsvColumnSpec(name="cost", dtype="float", description="Unit cost"),
                CsvColumnSpec(name="calories", dtype="int", description="Calories per unit"),
            ],
        ),
        CsvFileSpec(
            filename="limits.csv",
            description="Nutritional limits",
            columns=[
                CsvColumnSpec(name="nutrient", dtype="str", description="Nutrient name"),
                CsvColumnSpec(name="min_value", dtype="float", description="Minimum required"),
            ],
        ),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write foods.csv
        with open(os.path.join(tmpdir, "foods.csv"), "w", newline="") as f:
            f.write("name,cost,calories\n")
            f.write("Corn,0.18,72\n")
            f.write("Milk,0.23,121\n")
            f.write("Bread,0.05,65\n")

        # Write limits.csv
        with open(os.path.join(tmpdir, "limits.csv"), "w", newline="") as f:
            f.write("nutrient,min_value\n")
            f.write("calories,2000.0\n")
            f.write("protein,50.0\n")

        user_data = UserData.load_from_csv_dir(tmpdir, specs)

        # Check foods table
        assert "foods" in user_data.raw_tables
        foods = user_data.raw_tables["foods"]
        assert len(foods) == 3
        assert foods[0]["name"] == "Corn"
        assert foods[0]["cost"] == 0.18
        assert foods[0]["calories"] == 72
        assert isinstance(foods[0]["cost"], float)
        assert isinstance(foods[0]["calories"], int)

        # Check limits table
        assert "limits" in user_data.raw_tables
        limits = user_data.raw_tables["limits"]
        assert len(limits) == 2
        assert limits[0]["nutrient"] == "calories"
        assert limits[0]["min_value"] == 2000.0

        # Check as_dict includes tables
        flat = user_data.as_dict()
        assert "foods" in flat
        assert "limits" in flat

        # Check metadata
        assert user_data.csv_dir == tmpdir
        assert len(user_data.csv_specs) == 2


def test_load_from_csv_dir_missing_file():
    specs = [
        CsvFileSpec(filename="missing.csv", description="Does not exist"),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            UserData.load_from_csv_dir(tmpdir, specs)
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError as exc:
            assert "missing.csv" in str(exc)
