"""Parameter computation node — derive model-ready parameters from raw user data."""

from __future__ import annotations

import csv
import itertools
import math
import traceback
from pathlib import Path
from typing import Any

from orpilot.llm.base import BaseLLM, ToolDefinition
from orpilot.models.data import CsvColumnSpec, CsvFileSpec, UserData, _cast_value
from orpilot.paths import DATA_DIR
from orpilot.prompts import param_computation as prompts
from orpilot.workflow.state import WorkflowState

_MAX_ITERATIONS = 8

_TOOLS = [
    ToolDefinition(
        name="no_computation_needed",
        description=(
            "Call this when all parameters can be read directly from the raw tables "
            "and no transformation or derivation is required."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
    ),
    ToolDefinition(
        name="execute_script",
        description=(
            "Execute a Python computation script. The script runs with `data` (dict of "
            "table rows), `data_dir` (output path), and `csv`, `math`, `itertools`, `Path` "
            "pre-imported. It must set `output_files` at the end. "
            "The result tells you whether it succeeded or failed (with traceback). "
            "Fix and re-call if it failed."
        ),
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python script to execute. Must set output_files at the end.",
                }
            },
            "required": ["code"],
        },
    ),
]


def param_computation_node(state: WorkflowState, llm: BaseLLM) -> WorkflowState:
    """Compute derived parameters via an agentic tool loop.

    The LLM calls no_computation_needed() or execute_script(code). On script
    failure the traceback is returned as the tool result so the LLM can fix and
    re-execute in the same agentic turn.
    """
    user_data: UserData | None = state.get("user_data")
    problem = state.get("problem")
    data_dir = state.get("data_dir", str(DATA_DIR))

    if user_data is None:
        return {**state, "current_node": "param_computation"}

    table_schemas = _describe_tables(user_data)
    problem_json = problem.model_dump_json(indent=2) if problem else "{}"

    messages: list[dict] = [
        {"role": "system", "content": prompts.SYSTEM_PROMPT},
        {
            "role": "user",
            "content": prompts.USER_PROMPT_TEMPLATE.format(
                problem_json=problem_json,
                table_schemas=table_schemas,
            ),
        },
    ]

    accumulated_files: list[dict] = []

    for _ in range(_MAX_ITERATIONS):
        response = llm.chat_with_tools(messages, _TOOLS)

        if not response.is_tool_use:
            # LLM returned text without calling a tool — nudge it
            messages.append({"role": "assistant", "content": response.text or ""})
            messages.append({
                "role": "user",
                "content": (
                    "Please call no_computation_needed() if no transformation is required, "
                    "or execute_script(code) with the computation script."
                ),
            })
            continue

        results: dict[str, str] = {}
        done_no_computation = False
        script_succeeded = False

        for tc in response.tool_calls:
            if tc.name == "no_computation_needed":
                results[tc.id] = "OK, skipping computation."
                done_no_computation = True
            elif tc.name == "execute_script":
                code = tc.arguments.get("code", "")
                files, error = _run_computation(code, user_data.as_dict(), data_dir)
                if error:
                    results[tc.id] = f"Script failed:\n{error}"
                elif files:
                    accumulated_files.extend(files)
                    results[tc.id] = (
                        f"Success: generated {[f['filename'] for f in files]}. "
                        "You are done — do not call execute_script again unless additional "
                        "derived parameters are needed."
                    )
                    script_succeeded = True
                else:
                    results[tc.id] = (
                        "Script ran without error but output_files was empty. "
                        "Make sure the script sets output_files at the end."
                    )
            else:
                results[tc.id] = f"Unknown tool '{tc.name}'."

        if done_no_computation:
            return {**state, "current_node": "param_computation"}

        messages = llm.extend_messages(messages, response, results)

        if script_succeeded:
            break

    if accumulated_files:
        updated_user_data = _merge_computed_files(user_data, accumulated_files, data_dir)
        updates: dict[str, Any] = {
            "user_data": updated_user_data,
            "current_node": "param_computation",
            "csv_specs": [spec.model_dump() for spec in updated_user_data.csv_specs],
        }
        if problem:
            new_paths = {
                Path(f["filename"]).stem: str((Path(data_dir) / f["filename"]).resolve())
                for f in accumulated_files
                if (Path(data_dir) / f["filename"]).is_file()
            }
            merged_paths = {**(problem.csv_file_paths or {}), **new_paths}
            updates["problem"] = problem.model_copy(update={"csv_file_paths": merged_paths})
        return {**state, **updates}

    # All iterations exhausted without success
    return {
        **state,
        "error_context": (
            "Parameter computation failed after multiple attempts and no derived CSVs "
            "were produced. The IR may have parameters with no data source."
        ),
        "current_node": "param_computation",
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _describe_tables(user_data: UserData) -> str:
    lines: list[str] = []
    if user_data.csv_specs:
        for spec in user_data.csv_specs:
            stem = Path(spec.filename).stem
            cols = ", ".join(f"{c.name} ({c.dtype})" for c in spec.columns)
            lines.append(f"- {stem}: [{cols}]")
    else:
        for name, rows in user_data.raw_tables.items():
            if rows:
                cols = ", ".join(rows[0].keys())
                lines.append(f"- {name}: [{cols}]")
    return "\n".join(lines) if lines else "(none)"


def _run_computation(
    code: str, data: dict[str, Any], data_dir: str
) -> tuple[list[dict], str | None]:
    namespace: dict[str, Any] = {
        "data": data,
        "data_dir": data_dir,
        "output_files": [],
        "csv": csv,
        "math": math,
        "itertools": itertools,
        "Path": Path,
    }
    try:
        exec(code, namespace)  # noqa: S102
        return namespace.get("output_files", []), None
    except Exception:
        return [], traceback.format_exc()


def _merge_computed_files(
    user_data: UserData,
    output_files: list[dict],
    data_dir: str,
) -> UserData:
    new_raw_tables = dict(user_data.raw_tables)
    new_csv_specs = list(user_data.csv_specs)

    for file_info in output_files:
        filename = file_info.get("filename", "")
        if not filename:
            continue
        stem = Path(filename).stem
        filepath = Path(data_dir) / filename
        if not filepath.is_file():
            continue

        col_defs: list[dict] = file_info.get("columns", [])
        columns = [
            CsvColumnSpec(
                name=c["name"],
                dtype=c.get("dtype", "str"),
                description=c.get("description", ""),
            )
            for c in col_defs
        ]
        col_dtypes = {c.name: c.dtype for c in columns}

        rows: list[dict[str, Any]] = []
        with open(filepath, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                typed_row: dict[str, Any] = {}
                for key, value in row.items():
                    try:
                        typed_row[key] = _cast_value(value, col_dtypes.get(key, "str"))
                    except Exception:
                        typed_row[key] = value
                rows.append(typed_row)

        new_raw_tables[stem] = rows
        new_csv_specs.append(
            CsvFileSpec(
                filename=filename,
                description=file_info.get("description", ""),
                columns=columns,
            )
        )

    return user_data.model_copy(update={
        "raw_tables": new_raw_tables,
        "csv_specs": new_csv_specs,
    })
