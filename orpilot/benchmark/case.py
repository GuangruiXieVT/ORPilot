"""Benchmark case and result data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BenchmarkCase:
    """A single benchmark problem with optional pre-supplied artifacts."""

    name: str
    problem_text: str
    # Pre-extracted tables: stem → list of row dicts (Mode B / C)
    tables: dict[str, list[dict[str, Any]]] | None = None
    # Reference IR JSON dict (Mode C)
    ir_model: dict | None = None
    expected_objective: float | None = None
    expected_status: str = "optimal"
    objective_tolerance: float = 1e-4
    source: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    """Result of running a single benchmark case."""

    case_name: str
    solver: str
    mode: str  # "compiler", "ir_builder", or "full"
    status: str
    objective_value: float | None
    expected_objective: float | None
    objective_tolerance: float
    ir_model: dict | None = None
    generated_code: str = ""
    lp_content: str = ""
    error: str | None = None
    solve_time: float | None = None
    # Token usage for this case: {"input_tokens": int, "output_tokens": int}
    metrics: dict = field(default_factory=dict)
    # Extracted data tables: stem → list of row dicts (populated by solve pipelines)
    tables: dict[str, list[dict[str, Any]]] | None = None

    @property
    def passed(self) -> bool:
        """Return True when the objective value matches the expected value within tolerance."""
        if self.error:
            return False
        if self.expected_objective is None or self.objective_value is None:
            return False
        return abs(self.objective_value - self.expected_objective) <= self.objective_tolerance
