"""Reporter node — translate solution into a natural language report."""

from __future__ import annotations

import json

from orpilot.llm.base import BaseLLM
from orpilot.prompts import report as report_prompts
from orpilot.workflow.state import WorkflowState


def _build_csv_files_text(solution) -> str:
    """Describe each output CSV file's structure with a sample of its rows."""
    if not solution or not solution.variable_groups:
        return "(none)"

    sections: list[str] = []
    for group in solution.variable_groups:
        filename = f"solution_{group.group_name}.csv"
        labels = list(group.dimension_labels or [])
        columns = labels + ["value"]
        header = ", ".join(columns)

        # Collect up to 5 non-zero rows as illustrative examples
        sample_rows: list[str] = []
        _SEP = "\x1f"
        for var_name, value in sorted((group.variables or {}).items()):
            if value is not None and round(float(value), 6) != 0:
                if _SEP in var_name:
                    parts = var_name.split(_SEP, 1)
                    dims = parts[1].split(_SEP) if len(parts) > 1 else []
                else:
                    parts = var_name.split("_", len(labels))  # legacy fallback
                    dims = parts[len(parts) - len(labels):] if labels else []
                row = ", ".join(dims + [str(round(float(value), 4))])
                sample_rows.append(f"    {row}")
            if len(sample_rows) >= 5:
                break

        section = f"File: {filename}\n  Columns: {header}"
        if sample_rows:
            section += "\n  Sample rows (non-zero):\n" + "\n".join(sample_rows)
        sections.append(section)

    return "\n\n".join(sections)


def reporter_node(state: WorkflowState, llm: BaseLLM) -> WorkflowState:
    """Generate a natural-language report from the solution."""
    problem = state.get("problem")
    solution = state.get("solution")

    # Limit variables_text to a representative sample to avoid 413 on large models.
    # The csv_files_text already provides structured non-zero samples per group.
    _MAX_VARS = 50
    if solution and solution.variables:
        nonzero = {k: v for k, v in solution.variables.items()
                   if v is not None and round(float(v), 8) != 0}
        sample = dict(list(nonzero.items())[:_MAX_VARS])
        variables_text = json.dumps(sample, indent=2)
        if len(nonzero) > _MAX_VARS:
            variables_text += f"\n... ({len(nonzero) - _MAX_VARS} more non-zero variables omitted)"
    else:
        variables_text = "{}"

    csv_files_text = _build_csv_files_text(solution)

    # Truncate solver_output to avoid large logs inflating the prompt.
    solver_output = (solution.solver_output or "") if solution else ""
    if len(solver_output) > 3000:
        solver_output = solver_output[:2000] + "\n...[truncated]...\n" + solver_output[-500:]

    prompt = report_prompts.SYSTEM_PROMPT.format(
        problem_description=problem.description if problem else "Unknown",
        status=solution.status.value if solution else "unknown",
        objective_value=solution.objective_value if solution else "N/A",
        variables_text=variables_text,
        solver_output=solver_output,
        csv_files_text=csv_files_text,
    )

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "Generate the report."},
    ]

    report = llm.chat(messages)

    return {
        **state,
        "report": report,
        "current_node": "reporter",
        "needs_user_input": False,
    }
