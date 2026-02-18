"""Model builder node — generate OR model code via LLM."""

from __future__ import annotations

from aifor.codegen.generator import CodeGenerator
from aifor.llm.base import BaseLLM
from aifor.workflow.state import WorkflowState


def model_builder_node(state: WorkflowState, llm: BaseLLM) -> WorkflowState:
    """Generate OR model code using the LLM.

    If there's an error context from a previous failed attempt, retry with
    error information.
    """
    problem = state.get("problem")
    user_data = state.get("user_data")
    solver_name = state.get("solver_name", "pulp")
    error_context = state.get("error_context", "")
    previous_code = state.get("generated_code", "")

    generator = CodeGenerator(llm)

    if error_context and previous_code:
        code = generator.retry(
            previous_code=previous_code,
            error=error_context,
            solver_framework=solver_name,
        )
    else:
        code = generator.generate(
            problem=problem,
            data=user_data,
            solver_framework=solver_name,
        )

    return {
        **state,
        "generated_code": code,
        "current_node": "model_builder",
        "needs_user_input": False,
        "error_context": "",
    }
