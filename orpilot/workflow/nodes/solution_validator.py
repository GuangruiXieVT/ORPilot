"""Solution validator node — verify an optimal solution against the problem specification."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from orpilot.llm.base import BaseLLM
from orpilot.models.problem import ProblemDefinition
from orpilot.models.solution import SolutionResult
from orpilot.models.data import UserData
from orpilot.prompts import solution_validator as sv_prompts
from orpilot.workflow.state import WorkflowState

_VALIDATION_MARKER = "__ORPILOT_VALIDATION__:"
_SCRIPT_TIMEOUT = 30
_OBJ_REL_TOL = 0.01  # 1% relative tolerance for objective mismatch


def _decode_variable_groups(solution: SolutionResult) -> dict[str, list[dict]]:
    """Convert solution.variable_groups from the \x1f-encoded format to list-of-records."""
    SEP = "\x1f"
    result: dict[str, list[dict]] = {}
    for group in solution.variable_groups:
        records: list[dict] = []
        for key, val in (group.variables or {}).items():
            parts = key.split(SEP)
            # parts[0] is the variable name prefix, parts[1:] are index values
            indices = parts[1:] if len(parts) > 1 else []
            record: dict = {}
            for i, label in enumerate(group.dimension_labels or []):
                record[label] = indices[i] if i < len(indices) else ""
            record["value"] = float(val) if val is not None else 0.0
            records.append(record)
        result[group.group_name] = records
    return result


def _build_data_bundle(solution: SolutionResult, user_data: UserData | None) -> dict:
    """Build the JSON-serializable data bundle for the validation subprocess."""
    input_tables: dict = {}
    if user_data:
        for table_name, rows in user_data.raw_tables.items():
            input_tables[table_name] = rows

    return {
        "input": input_tables,
        "solution": _decode_variable_groups(solution),
        "reported_obj": solution.objective_value,
    }


def _build_schema_context(user_data: UserData | None, solution: SolutionResult) -> tuple[str, str]:
    """Return (input_schema_text, solution_schema_text) for the LLM prompt."""
    input_lines: list[str] = []
    if user_data:
        for table_name, rows in user_data.raw_tables.items():
            cols = list(rows[0].keys()) if rows else []
            sample = json.dumps(rows[0]) if rows else "{}"
            input_lines.append(
                f'data["input"]["{table_name}"]: {len(rows)} rows, '
                f'columns={cols}, sample={sample}'
            )
    input_schema = "\n".join(input_lines) or "(no input data)"

    sol_lines: list[str] = []
    SEP = "\x1f"
    for group in solution.variable_groups:
        labels = list(group.dimension_labels or [])
        n = len(group.variables or {})
        # Show up to 2 sample records in decoded form
        samples = []
        for key, val in list((group.variables or {}).items())[:2]:
            parts = key.split(SEP)
            indices = parts[1:] if len(parts) > 1 else []
            rec = {labels[i]: indices[i] for i in range(min(len(labels), len(indices)))}
            rec["value"] = round(float(val), 4) if val is not None else 0.0
            samples.append(json.dumps(rec))
        sample_str = ", ".join(samples)
        sol_lines.append(
            f'data["solution"]["{group.group_name}"]: {n} records, '
            f'dimensions={labels}, e.g. [{sample_str}]'
        )
    solution_schema = "\n".join(sol_lines) or "(no solution variables)"

    return input_schema, solution_schema


def _format_problem(problem: ProblemDefinition) -> str:
    """Serialize ProblemDefinition to a readable string for the validator prompt."""
    parts: list[str] = []
    if problem.title:
        parts.append(f"Title: {problem.title}")
    if problem.description:
        parts.append(problem.description)
    if problem.objective_description:
        parts.append(f"Objective ({problem.objective.value}): {problem.objective_description}")
    if problem.constraints:
        parts.append("Constraints:")
        for c in problem.constraints:
            parts.append(f"  - {c.description}")
    if problem.decision_variables:
        parts.append("Decision variables:")
        for v in problem.decision_variables:
            parts.append(f"  - {v}")
    if problem.parameters:
        parts.append("Parameters:")
        for p in problem.parameters:
            parts.append(f"  - {p}")
    if problem.additional_notes:
        parts.append(f"Notes: {problem.additional_notes}")
    return "\n".join(parts) if parts else "(no description)"


def _strip_code_fences(code: str) -> str:
    lines = code.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


def _execute_script(script: str, data_bundle: dict) -> dict | None:
    """Run the validation script in a subprocess; return parsed JSON result or None."""
    script = _strip_code_fences(script)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        (tmpdir_path / "data.json").write_text(
            json.dumps(data_bundle), encoding="utf-8"
        )
        (tmpdir_path / "validate.py").write_text(script, encoding="utf-8")

        try:
            proc = subprocess.run(
                [sys.executable, "validate.py"],
                capture_output=True,
                text=True,
                timeout=_SCRIPT_TIMEOUT,
                cwd=tmpdir,
            )
        except subprocess.TimeoutExpired:
            return {"error": f"Validation script timed out after {_SCRIPT_TIMEOUT}s"}

        if proc.returncode != 0:
            return {"error": (proc.stderr or f"Exit code {proc.returncode}")[:1000]}

        for line in proc.stdout.splitlines():
            if line.startswith(_VALIDATION_MARKER):
                try:
                    return json.loads(line[len(_VALIDATION_MARKER):])
                except json.JSONDecodeError as exc:
                    return {"error": f"JSON decode failed: {exc}"}

        return {"error": "No validation output found. stdout: " + proc.stdout[:500]}


def _check_result(result: dict, reported_obj: float | None) -> str | None:
    """Return a structured error message if validation failed, None if passed."""
    computed = result.get("computed_obj")
    reported = result.get("reported_obj") or reported_obj
    violations = result.get("violations") or []
    notes = result.get("notes") or ""
    terms = result.get("terms") or {}

    issues: list[str] = []

    if computed is not None and reported is not None and reported != 0:
        rel_diff = abs(computed - reported) / abs(reported)
        if rel_diff > _OBJ_REL_TOL:
            pct = (computed - reported) / abs(reported) * 100
            terms_str = "\n    ".join(f"{k}: {v:.6g}" for k, v in terms.items())
            issues.append(
                f"Objective mismatch: computed={computed:.6g}, reported={reported:.6g} ({pct:+.1f}%)\n"
                f"  Term breakdown:\n    {terms_str}"
            )

    if violations:
        viol_strs = [
            f"  {v.get('constraint', '?')}[{v.get('index', '?')}]: "
            f"lhs={v.get('lhs', '?')} vs rhs={v.get('rhs', '?')}"
            for v in violations[:5]
        ]
        issues.append("Constraint violations (sample):\n" + "\n".join(viol_strs))

    if not issues:
        return None

    parts = ["SOLUTION VALIDATION FAILED:"] + issues
    if notes:
        parts.append(f"Validator notes: {notes}")
    return "\n".join(parts)


def solution_validator_node(state: WorkflowState, llm: BaseLLM) -> WorkflowState:
    """Verify the optimal solution against the problem specification.

    Works with or without an IR model:
    - With IR: passes problem description + IR JSON + CSV schemas + solution.
    - Without IR (direct code gen path): passes problem description + CSV schemas + solution.

    Skips silently only when the validation script itself fails to execute (inconclusive).
    On failure, sets error_context and increments validation_attempts.
    """
    ir_model = state.get("ir_model")
    solution: SolutionResult = state["solution"]
    user_data: UserData | None = state.get("user_data")
    problem = state.get("problem")

    problem_description = _format_problem(problem) if problem else "(no description)"
    ir_json = json.dumps(ir_model, indent=2) if ir_model else None
    data_bundle = _build_data_bundle(solution, user_data)
    input_schema, solution_schema = _build_schema_context(user_data, solution)

    system = sv_prompts.SYSTEM_PROMPT
    user_msg = sv_prompts.build_user_message(
        problem_description=problem_description,
        input_schema=input_schema,
        solution_schema=solution_schema,
        reported_obj=solution.objective_value,
        ir_json=ir_json,
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]

    # Generate script; allow one retry on execution error
    result: dict | None = None
    for attempt in range(2):
        script = llm.chat(messages)
        exec_result = _execute_script(script, data_bundle)

        if exec_result is None or "error" not in exec_result:
            result = exec_result
            break

        if attempt == 0:
            print(f"[Validator] Script error (attempt {attempt+1}): {exec_result['error'][:200]}")
            messages = messages + [
                {"role": "assistant", "content": script},
                {
                    "role": "user",
                    "content": (
                        f"The script failed:\n{exec_result['error']}\n\n"
                        "Fix the error and rewrite the complete script."
                    ),
                },
            ]

    if result is None or "error" in (result or {}):
        # Inconclusive — don't block the user
        print("[Validator] Validation inconclusive, proceeding to reporter.")
        return {
            **state,
            "current_node": "solution_validator",
            "validator_inconclusive": state.get("validator_inconclusive", 0) + 1,
        }

    error_msg = _check_result(result, solution.objective_value)

    if error_msg:
        print(f"[Validator] {error_msg[:300]}")
        return {
            **state,
            "current_node": "solution_validator",
            "error_context": error_msg,
            "retry_count": state.get("retry_count", 0) + 1,
            "validation_attempts": state.get("validation_attempts", 0) + 1,
        }

    print("[Validator] Solution passed validation.")
    return {**state, "current_node": "solution_validator"}
