"""IR builder node — translate ProblemDefinition JSON into a strict JSON IR via LLM."""

from __future__ import annotations

import json
import re
from pathlib import Path

from orpilot.llm.base import BaseLLM
from orpilot.models.ir import IRModel
from orpilot.prompts import ir_builder as ir_builder_prompts
from orpilot.workflow.state import WorkflowState


def _strip_fences(text: str) -> str:
    """Remove markdown code fences from LLM response."""
    pattern = r"```(?:json)?\s*\n?(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def ir_builder_node(state: WorkflowState, llm: BaseLLM) -> WorkflowState:
    """Call the LLM to translate the ProblemDefinition into a JSON IR.

    Retries up to 2 times on parse/validation failure.
    On UNSUPPORTED_MODEL error: short-circuits to reporter with a user message.
    """
    problem = state["problem"]
    user_data = state.get("user_data")

    # Build csv_schemas: table_stem → [col1, col2, ...] so the LLM knows
    # exact column names and can fill "column" fields without guessing.
    csv_schemas: dict[str, list[str]] = {}
    if user_data and user_data.csv_specs:
        for spec in user_data.csv_specs:
            stem = Path(spec.filename).stem
            csv_schemas[stem] = [c.name for c in spec.columns]

    user_payload: dict = {"problem": json.loads(problem.model_dump_json())}
    if csv_schemas:
        user_payload["csv_schemas"] = csv_schemas

    messages = [
        {"role": "system", "content": ir_builder_prompts.SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload)},
    ]

    for attempt in range(3):  # initial attempt + up to 2 retries
        response = llm.chat(messages)
        try:
            ir_dict = json.loads(_strip_fences(response))
            if ir_dict.get("error") == "UNSUPPORTED_MODEL":
                return {
                    **state,
                    "report": (
                        "This problem cannot be represented as a linear or mixed-integer "
                        "program. Please reformulate your problem and try again."
                    ),
                    "current_node": "reporter",
                }
            IRModel.model_validate(ir_dict)
            return {**state, "ir_model": ir_dict, "current_node": "ir_builder"}
        except Exception as exc:
            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": f"Validation failed: {exc}. Return corrected JSON only.",
            })

    raise RuntimeError("IR builder failed after 3 attempts")
