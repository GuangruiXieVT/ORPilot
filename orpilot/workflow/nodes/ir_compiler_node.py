"""IR compiler node — deterministic IR → solver code, with LLM retry on failure."""

from __future__ import annotations

import re

from orpilot.codegen.ir_compiler import IRCompiler
from orpilot.llm.base import BaseLLM
from orpilot.prompts import ir_compiler as ir_compiler_prompts
from orpilot.workflow.state import WorkflowState


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences from LLM response."""
    pattern = r"```(?:python)?\s*\n?(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _llm_fix(llm: BaseLLM, previous_code: str, error: str, solver: str) -> str:
    """Ask the LLM to fix broken solver code."""
    prompt = ir_compiler_prompts.RETRY_PROMPT.format(
        error=error,
        previous_code=previous_code,
    )
    messages = [
        {"role": "system", "content": f"You are an expert {solver} programmer."},
        {"role": "user", "content": prompt},
    ]
    response = llm.chat(messages)
    return _strip_code_fences(response)


def ir_compiler_node(state: WorkflowState, llm: BaseLLM) -> WorkflowState:
    """Compile the IR to solver code, or LLM-fix on retry."""
    ir_model = state.get("ir_model")
    error_ctx = state.get("error_context", "")
    prev_code = state.get("generated_code", "")
    solver = state.get("solver_name", "pulp")

    if error_ctx and prev_code:
        code = _llm_fix(llm, prev_code, error_ctx, solver)
    else:
        code = IRCompiler().compile(ir_model, solver)

    return {
        **state,
        "generated_code": code,
        "error_context": "",
        "current_node": "ir_compiler",
    }
