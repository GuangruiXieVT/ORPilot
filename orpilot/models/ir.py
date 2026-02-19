"""Pydantic models for the Intermediate Representation (IR) of an OR model."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class IRSet(BaseModel):
    size: int | None
    index_symbol: str
    source: str | None
    column: str | None


class IRParameter(BaseModel):
    domain: list[str]
    type: str
    source: str | None
    column: str | None = None  # CSV column that holds this parameter's values


class IRVariable(BaseModel):
    description: str
    label: str | None = None  # short snake_case name for output files, e.g. "shipments"
    domain: list[str]
    type: str
    lower_bound: float | None
    upper_bound: float | None


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
