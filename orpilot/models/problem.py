"""Problem definition schemas."""

from enum import Enum

from pydantic import BaseModel, Field, model_validator


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
    parameters: list[str] = Field(
        default_factory=list,
        description="Natural language descriptions of parameters with their indices, e.g. 'cost[i,j]: shipping cost from i to j'",
    )
    additional_notes: str = Field("", description="Any extra context from the user")

    @model_validator(mode="before")
    @classmethod
    def _coerce_none_lists(cls, data):
        if isinstance(data, dict):
            for field in ("constraints", "decision_variables", "parameters"):
                if data.get(field) is None:
                    data[field] = []
        return data
    csv_file_paths: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of table name to absolute CSV file path (e.g. {'costs': '/data/costs.csv'})",
    )
