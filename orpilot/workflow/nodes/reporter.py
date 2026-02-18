"""Reporter node — translate solution into a natural language report."""

from __future__ import annotations

import json

from orpilot.llm.base import BaseLLM
from orpilot.prompts import report as report_prompts
from orpilot.workflow.state import WorkflowState


def reporter_node(state: WorkflowState, llm: BaseLLM) -> WorkflowState:
    """Generate a natural-language report from the solution."""
    problem = state.get("problem")
    solution = state.get("solution")

    variables_text = json.dumps(solution.variables, indent=2) if solution else "{}"

    prompt = report_prompts.SYSTEM_PROMPT.format(
        problem_description=problem.description if problem else "Unknown",
        status=solution.status.value if solution else "unknown",
        objective_value=solution.objective_value if solution else "N/A",
        variables_text=variables_text,
        solver_output=solution.solver_output if solution else "",
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
