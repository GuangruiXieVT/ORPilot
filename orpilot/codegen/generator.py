"""Orchestrates LLM code generation for OR models."""

from __future__ import annotations

import json
import re
from pathlib import Path

from orpilot.llm.base import BaseLLM
from orpilot.models.problem import ProblemDefinition
from orpilot.models.data import UserData
from orpilot.prompts import codegen as codegen_prompts

_SAMPLE_ROWS = 5  # number of example rows sent to the LLM per table


class CodeGenerator:
    """Uses an LLM to generate solver code for an OR problem."""

    def __init__(self, llm: BaseLLM):
        self._llm = llm

    def generate(
        self,
        problem: ProblemDefinition,
        data: UserData,
        solver_framework: str = "pulp",
    ) -> str:
        """Generate solver code for the given problem and data."""
        prompt = codegen_prompts.SYSTEM_PROMPT.format(
            solver_framework=solver_framework,
            problem_json=problem.model_dump_json(indent=2),
            data_json=json.dumps(_make_data_context(data), indent=2),
        )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Generate the solver code now."},
        ]

        response = self._llm.chat(messages)
        return self._extract_code(response)

    def retry(
        self,
        previous_code: str,
        error: str,
        solver_framework: str = "pulp",
    ) -> str:
        """Retry code generation with error context."""
        prompt = codegen_prompts.RETRY_PROMPT.format(
            error=error,
            previous_code=previous_code,
        )

        messages = [
            {"role": "system", "content": f"You are an expert {solver_framework} programmer."},
            {"role": "user", "content": prompt},
        ]

        response = self._llm.chat(messages)
        return self._extract_code(response)

    @staticmethod
    def _extract_code(response: str) -> str:
        """Extract Python code from LLM response, stripping markdown fences."""
        # Try to extract from code blocks
        pattern = r"```(?:python)?\s*\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()
        # If no code blocks, assume the whole response is code
        return response.strip()


def _make_data_context(data: UserData, max_sample_rows: int = _SAMPLE_ROWS) -> dict:
    """Build a compact schema + sample dict for the LLM code-generation prompt.

    The full dataset is never embedded in the prompt — only column schemas and
    up to *max_sample_rows* example rows per table are included so the LLM can
    understand the structure without consuming the entire context window.
    """
    context: dict = {}

    if data.parameters:
        context["parameters"] = {p.name: p.value for p in data.parameters}

    spec_by_stem = {Path(s.filename).stem: s for s in data.csv_specs}
    for table_name, rows in data.raw_tables.items():
        spec = spec_by_stem.get(table_name)
        entry: dict = {"total_rows": len(rows), "sample_rows": rows[:max_sample_rows]}
        if spec:
            entry["columns"] = [
                {"name": c.name, "dtype": c.dtype, "description": c.description}
                for c in spec.columns
            ]
        context[table_name] = entry

    return context
