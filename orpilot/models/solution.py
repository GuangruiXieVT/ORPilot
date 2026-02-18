"""Solution result schemas."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SolveStatus(str, Enum):
    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    ERROR = "error"
    TIMEOUT = "timeout"


class VariableGroup(BaseModel):
    """A group of decision variables that share the same type and dimensions."""

    group_name: str = Field(description="Descriptive snake_case name, used as CSV filename")
    dimension_labels: list[str] = Field(default_factory=list, description="Column names for dimensions")
    variables: dict[str, Any] = Field(default_factory=dict, description="Variable name -> value")


class SolutionResult(BaseModel):
    """Result from solving an OR model."""

    status: SolveStatus = Field(SolveStatus.ERROR)
    objective_value: float | None = Field(None, description="Optimal objective value")
    variables: dict[str, Any] = Field(default_factory=dict, description="Decision variable values (flat)")
    variable_groups: list[VariableGroup] = Field(default_factory=list, description="Variables grouped by type")
    solver_output: str = Field("", description="Raw solver output / logs")
    error_message: str = Field("", description="Error details if solve failed")
    solve_time_seconds: float | None = Field(None)
    lp_content: str = Field("", description="LP file content if generated")
