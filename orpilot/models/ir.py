"""Pydantic models for the Intermediate Representation (IR) of an OR model."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class IRSet(BaseModel):
    size: int | None
    index_symbol: str
    source: str | None
    column: str | None
    # Optional: derive set size from a scalar integer stored in a parameter CSV.
    # When set, the compiler emits list(range(int(data[size_source][0][size_column]))).
    size_source: str | None = None
    size_column: str | None = None
    # When True, the set has a meaningful ordering and its members support lag references
    # (e.g. Months, Periods, Shifts).  The compiler emits enumerate() loops for any
    # constraint that contains a variable/parameter node with a non-zero "lag" field.
    ordered: bool = False


class IRParameter(BaseModel):
    domain: list[str]
    type: str
    source: str | None
    column: str | None = None  # CSV column that holds this parameter's values
    index_columns: list[str] | None = None  # per-index CSV key columns; overrides set_column lookup
    missing_default: str = "zero"  # "zero" → 0.0, "inf" → float('inf') for missing index combinations
    optional: bool = False  # when True, the source CSV may be absent; missing file loads as empty list


class IRVariable(BaseModel):
    description: str
    label: str | None = None  # short snake_case name for output files, e.g. "shipments"
    domain: list[str]
    type: str
    lower_bound: float | None
    upper_bound: float | None
    upper_bound_set: str | None = None  # when set, compiler emits len(SetName) as the upper bound
    exclude_diagonal: bool = False  # when True, exclude (i, i, ...) keys from the variable dict


class IRConstraint(BaseModel):
    domain: list[str]
    expression: dict[str, Any]
    sense: str
    rhs: dict[str, Any]


class IRObjective(BaseModel):
    sense: str
    expression: dict[str, Any]


class IRModel(BaseModel):
    problem_class: str
    model_type: str
    sense: str
    sets: dict[str, IRSet]
    parameters: dict[str, IRParameter]
    variables: dict[str, IRVariable]
    constraints: dict[str, IRConstraint]
    objective: IRObjective
