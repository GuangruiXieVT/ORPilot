"""Problem definition schemas."""

from enum import Enum

from pydantic import BaseModel, Field


class ProblemType(str, Enum):
    LINEAR_PROGRAMMING = "linear_programming"
    INTEGER_PROGRAMMING = "integer_programming"
    MIXED_INTEGER = "mixed_integer"
    TRANSPORTATION = "transportation"
    ASSIGNMENT = "assignment"
    SCHEDULING = "scheduling"
    NETWORK_FLOW = "network_flow"
    OTHER = "other"


class ObjectiveType(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class Constraint(BaseModel):
    description: str = Field(..., description="Natural language description of the constraint")
    expression: str | None = Field(None, description="Mathematical expression if available")


class ProblemDefinition(BaseModel):
    """Structured definition of an OR problem extracted from user interview."""

    title: str = Field("", description="Short title for the problem")
    description: str = Field("", description="Full natural-language description")
    problem_type: ProblemType = Field(ProblemType.OTHER, description="Classification of the problem")
    objective: ObjectiveType = Field(ObjectiveType.MINIMIZE, description="Optimization direction")
    objective_description: str = Field("", description="What is being optimized")
    constraints: list[Constraint] = Field(default_factory=list)
    decision_variables: list[str] = Field(
        default_factory=list,
        description="Natural language descriptions of decision variables",
    )
    additional_notes: str = Field("", description="Any extra context from the user")
    csv_file_paths: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of table name to absolute CSV file path (e.g. {'costs': '/data/costs.csv'})",
    )
