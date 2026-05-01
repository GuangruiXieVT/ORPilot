"""IR builder node — translate ProblemDefinition JSON into a strict JSON IR via LLM."""

from __future__ import annotations

import json
from pathlib import Path

from orpilot.codegen.ir_validator import validate_ir_semantics
from orpilot.llm.base import BaseLLM, ToolDefinition
from orpilot.models.ir import IRModel
from orpilot.prompts import ir_builder as ir_builder_prompts
from orpilot.workflow.state import WorkflowState

_MAX_ITERATIONS = 12

_TOOLS = [
    ToolDefinition(
        name="submit_ir",
        description=(
            "Submit the complete IR JSON for schema and semantic validation. "
            "If validation fails you will receive the exact error — fix only that part "
            "and call submit_ir again. Call this when you have built the full IR."
        ),
        parameters={
            "type": "object",
            "properties": {
                "ir": {
                    "type": "object",
                    "description": "The complete IR as a JSON object matching the IR schema.",
                }
            },
            "required": ["ir"],
        },
    ),
    ToolDefinition(
        name="report_unsupported",
        description=(
            "Call this ONLY when the problem cannot be modeled as a linear or "
            "mixed-integer program (e.g. non-linear objectives, purely combinatorial "
            "problems not expressible as MIP)."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
    ),
]

_TOOL_INSTRUCTIONS = """
## Tool Usage

You have two tools:

1. **submit_ir(ir)** — Submit your complete IR for validation. On failure the error identifies \
the exact problem; fix only that part and call submit_ir again. On success you are done.

2. **report_unsupported()** — Call ONLY if the problem cannot be modeled as an LP or MIP.

CSV column names and distinct values are already provided in `csv_schemas` — use them directly. \
Do NOT output raw JSON text. Always deliver the IR via submit_ir.
"""

_UNSUPPORTED_REPORT = (
    "This problem cannot be represented as a linear or mixed-integer program. "
    "Please reformulate your problem and try again."
)



def _check_hardcoded_indices(ir_dict: dict, csv_schemas: dict) -> list[str]:
    """Return errors for hardcoded string indices that don't exist in any known set member list.

    Catches hallucinated depot labels (e.g. "D0") that should be looked up from
    csv_schemas distinct_values instead. Only runs when csv_schemas is non-empty.
    """
    if not csv_schemas:
        return []

    # Build the set of all known member IDs across all sets
    all_members: set[str] = set()
    for set_name, set_meta in ir_dict.get("sets", {}).items():
        source = set_meta.get("source", "")
        column = set_meta.get("column", "")
        stem = Path(source).stem if source else ""
        if stem and column and stem in csv_schemas:
            vals = csv_schemas[stem].get("distinct_values", {}).get(column, [])
            all_members.update(str(v) for v in vals)

    if not all_members:
        return []

    # Collect all loop-variable names (index_symbols + numbered variants for duplicates)
    loop_vars: set[str] = set()
    for set_meta in ir_dict.get("sets", {}).values():
        sym = set_meta.get("index_symbol")
        if sym:
            loop_vars.add(sym)
            for suffix in ("1", "2", "3"):
                loop_vars.add(sym + suffix)

    bad: dict[str, str] = {}  # literal → variable name (dedup by literal)

    def _walk(expr: object) -> None:
        if not isinstance(expr, dict):
            return
        if expr.get("type") == "variable":
            var_name = expr.get("name", "?")
            for idx in expr.get("indices", []):
                if isinstance(idx, str) and idx not in loop_vars and idx not in all_members:
                    bad[idx] = var_name
        for key in ("left", "right", "body"):
            _walk(expr.get(key))

    for c_meta in ir_dict.get("constraints", {}).values():
        if isinstance(c_meta, dict):
            _walk(c_meta.get("expression"))
            _walk(c_meta.get("rhs"))
    _walk(ir_dict.get("objective", {}).get("expression"))

    if not bad:
        return []
    sample = sorted(all_members)[:12]
    return [
        f"Hardcoded index '{lit}' in variable '{vname}' is not a recognized set member. "
        f"Known members include: {sample}. "
        f"Look up the actual ID from csv_schemas distinct_values — do NOT invent labels."
        for lit, vname in bad.items()
    ]


def _exec_submit_ir(ir_dict: dict) -> str | None:
    """Validate ir_dict. Returns None on success, error string on failure."""
    if "_parse_error" in ir_dict:
        return ir_dict["_parse_error"]
    if not ir_dict.get("variables"):
        return "Rejected: IR has no variables. Submit a complete IR with all sets, parameters, variables, constraints, and objective filled in."
    if not ir_dict.get("constraints"):
        return "Rejected: IR has no constraints. Submit a complete IR with all sets, parameters, variables, constraints, and objective filled in."
    try:
        IRModel.model_validate(ir_dict)
        errors = validate_ir_semantics(ir_dict)
        if errors:
            return "Semantic validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        return None
    except Exception as exc:
        return f"Validation failed: {exc}"


def _build_csv_schemas(user_data) -> dict:
    csv_schemas: dict[str, dict] = {}
    if user_data and user_data.csv_specs:
        for spec in user_data.csv_specs:
            stem = Path(spec.filename).stem
            rows = user_data.raw_tables.get(stem, [])
            col_names = [c.name for c in spec.columns]
            distinct: dict[str, list] = {}
            for col in col_names:
                seen: dict = {}
                for row in rows:
                    v = str(row[col]) if col in row else ""
                    seen[v] = None
                distinct[col] = list(seen.keys())[:30]
            csv_schemas[stem] = {"columns": col_names, "distinct_values": distinct}
    return csv_schemas


def _run_tool_loop(
    llm: BaseLLM,
    messages: list[dict],
    csv_schemas: dict,
) -> tuple[dict | None, str | None, str | None]:
    """Run the agentic tool loop.

    Returns (ir_dict, unsupported_msg, failure_msg).
    Exactly one of the three will be non-None.
    """
    for _ in range(_MAX_ITERATIONS):
        response = llm.chat_with_tools(messages, _TOOLS)

        if not response.is_tool_use:
            # LLM returned plain text — nudge it to use submit_ir.
            # Use extend_messages so reasoning_content (DeepSeek) is preserved.
            messages = llm.extend_messages(messages, response, {})
            messages.append({
                "role": "user",
                "content": (
                    "Please call submit_ir with your complete IR JSON, "
                    "or report_unsupported if the problem cannot be modeled."
                ),
            })
            continue

        results: dict[str, str] = {}
        submitted_ir: dict | None = None
        is_unsupported = False

        for tc in response.tool_calls:
            if tc.name == "submit_ir":
                ir_dict = tc.arguments.get("ir", {})
                error = _exec_submit_ir(ir_dict)
                if error is None:
                    idx_errors = _check_hardcoded_indices(ir_dict, csv_schemas)
                    if idx_errors:
                        error = "Hardcoded index validation failed:\n" + "\n".join(
                            f"  - {e}" for e in idx_errors
                        )
                if error is None:
                    submitted_ir = ir_dict
                    results[tc.id] = "IR accepted. Compilation will proceed."
                else:
                    results[tc.id] = error
            elif tc.name == "report_unsupported":
                is_unsupported = True
                results[tc.id] = "Acknowledged."
            else:
                results[tc.id] = f"Unknown tool '{tc.name}'."

        if is_unsupported:
            return None, _UNSUPPORTED_REPORT, None

        if submitted_ir is not None:
            return submitted_ir, None, None

        messages = llm.extend_messages(messages, response, results)

    return None, None, "The model could not be built after multiple attempts. Please refine your problem description or data and try again."


def ir_builder_node(state: WorkflowState, llm: BaseLLM) -> WorkflowState:
    """Translate the ProblemDefinition into a validated JSON IR using an agentic tool loop."""
    problem = state["problem"]
    user_data = state.get("user_data")
    csv_schemas = _build_csv_schemas(user_data)

    user_payload: dict = {"problem": json.loads(problem.model_dump_json())}
    if csv_schemas:
        user_payload["csv_schemas"] = csv_schemas

    error_context = state.get("error_context")
    existing_ir = state.get("ir_model")
    generated_code = state.get("generated_code", "")

    system = ir_builder_prompts.SYSTEM_PROMPT + "\n\n" + _TOOL_INSTRUCTIONS

    if error_context and existing_ir:
        code_block = f"\nCompiled code:\n```python\n{generated_code}\n```" if generated_code else ""
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    json.dumps(user_payload) + "\n\n"
                    f"Your previous IR attempt:\n```json\n{json.dumps(existing_ir, indent=2)}\n```\n\n"
                    f"Running it produced this error:\n{error_context}{code_block}\n\n"
                    "Identify the root cause in the IR and call submit_ir with the corrected version."
                ),
            },
        ]
    elif error_context:
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    json.dumps(user_payload) + "\n\n"
                    f"Warning: parameter computation failed and some derived CSVs "
                    f"may not be available:\n{error_context}\n\n"
                    "If a required CSV is missing, set source to null."
                ),
            },
        ]
    else:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_payload)},
        ]

    ir_dict, unsupported, failure = _run_tool_loop(llm, messages, csv_schemas)

    if unsupported:
        return {**state, "report": unsupported, "current_node": "reporter"}
    if failure:
        return {**state, "report": failure, "current_node": "reporter"}
    return {**state, "ir_model": ir_dict, "error_context": "", "current_node": "ir_builder"}


def ir_builder_on_demand_node(state: WorkflowState, llm: BaseLLM) -> WorkflowState:
    """Generate IR on-demand from the problem description + working Python code.

    Called after a successful solve to produce a solver-agnostic IR blueprint.
    On failure the node simply skips IR (the solve was already successful).
    """
    problem = state["problem"]
    user_data = state.get("user_data")
    generated_code = state.get("generated_code", "")
    csv_schemas = _build_csv_schemas(user_data)

    user_payload: dict = {"problem": json.loads(problem.model_dump_json())}
    if csv_schemas:
        user_payload["csv_schemas"] = csv_schemas

    code_block = f"```python\n{generated_code}\n```" if generated_code else ""
    system = ir_builder_prompts.SYSTEM_PROMPT + "\n\n" + _TOOL_INSTRUCTIONS
    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                json.dumps(user_payload) + "\n\n"
                "The following Python solver code already solves this problem correctly. "
                "Use it as a structural reference (sets, variables, parameters, constraints, "
                "objective direction) to generate the IR. The IR must be solver-agnostic — "
                "extract the abstract model structure, do not transliterate solver-specific "
                "syntax (e.g. PuLP's lpSum or LpVariable).\n\n"
                f"{code_block}\n\n"
                "Call submit_ir with the completed IR."
            ),
        },
    ]

    ir_dict, unsupported, _ = _run_tool_loop(llm, messages, csv_schemas)

    if unsupported or ir_dict is None:
        return {**state, "current_node": "ir_builder_on_demand"}
    return {**state, "ir_model": ir_dict, "current_node": "ir_builder_on_demand"}
