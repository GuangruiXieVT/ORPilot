"""Data collection node — guide user to provide CSV data files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from orpilot.llm.base import BaseLLM
from orpilot.models.data import CsvFileSpec, UserData
from orpilot.prompts import data_guide
from orpilot.workflow.state import WorkflowState


def data_collection_node(state: WorkflowState, llm: BaseLLM) -> WorkflowState:
    """Guide the user to provide CSV data for the OR model.

    Phase 1 (spec): LLM defines CSV specs via conversation.  When the LLM
    signals ``[DATA_SPEC_READY]``, extract ``CsvFileSpec`` list and ask the
    user to place files in ``data_dir``.

    Phase 2 (load): User confirms files are ready.  Load via
    ``UserData.load_from_csv_dir()``; if any are missing, report which ones
    and ask again.
    """
    csv_specs: list[dict[str, Any]] = state.get("csv_specs", [])

    if csv_specs:
        return _phase_load(state, csv_specs)
    return _phase_spec(state, llm)


def _phase_spec(state: WorkflowState, llm: BaseLLM) -> WorkflowState:
    """Phase 1: LLM defines the CSV file specifications."""
    messages = list(state.get("messages", []))
    problem = state.get("problem")
    data_dir = state.get("data_dir", "./data")

    problem_json = problem.model_dump_json(indent=2) if problem else "{}"

    system_prompt = data_guide.SYSTEM_PROMPT.format(
        problem_json=problem_json,
    )
    llm_messages = [{"role": "system", "content": system_prompt}]
    llm_messages.extend(messages)

    response = llm.chat(llm_messages)
    messages.append({"role": "assistant", "content": response})

    updates: dict[str, Any] = {
        "messages": messages,
        "current_node": "data_collection",
    }

    if "[DATA_SPEC_READY]" in response:
        # Extract structured CSV specs from conversation
        conversation_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in messages
        )
        extract_prompt = data_guide.SPEC_EXTRACTION_PROMPT.format(
            conversation=conversation_text,
        )

        from pydantic import BaseModel, Field

        class _CsvSpecList(BaseModel):
            specs: list[CsvFileSpec] = Field(default_factory=list)

        spec_result = llm.structured_output(
            [
                {"role": "system", "content": "Extract CSV file specifications."},
                {"role": "user", "content": extract_prompt},
            ],
            _CsvSpecList,
        )

        spec_dicts = [s.model_dump() for s in spec_result.specs]
        updates["csv_specs"] = spec_dicts

        # Clean marker from displayed message
        clean = response.replace("[DATA_SPEC_READY]", "").strip()
        ready_msg = (
            f"{clean}\n\n"
            f"Please place the CSV files in: **{data_dir}**\n"
            "Type **ready** when the files are in place."
        )
        messages[-1] = {"role": "assistant", "content": ready_msg}
        updates["messages"] = messages
        updates["needs_user_input"] = True
    else:
        updates["needs_user_input"] = True

    return {**state, **updates}


def _phase_load(
    state: WorkflowState,
    csv_spec_dicts: list[dict[str, Any]],
) -> WorkflowState:
    """Phase 2: Load CSV files from the data directory."""
    messages = list(state.get("messages", []))
    data_dir = state.get("data_dir", "./data")

    specs = [CsvFileSpec.model_validate(d) for d in csv_spec_dicts]

    try:
        user_data = UserData.load_from_csv_dir(data_dir, specs)
        problem = state.get("problem")
        if problem:
            csv_paths = {
                Path(spec.filename).stem: str((Path(data_dir) / spec.filename).resolve())
                for spec in specs
            }
            problem = problem.model_copy(update={"csv_file_paths": csv_paths})
    except FileNotFoundError as exc:
        messages.append({
            "role": "assistant",
            "content": (
                f"{exc}\n\n"
                f"Please place the missing file(s) in **{data_dir}** and type **ready**."
            ),
        })
        return {
            **state,
            "messages": messages,
            "current_node": "data_collection",
            "needs_user_input": True,
        }
    except ValueError as exc:
        messages.append({
            "role": "assistant",
            "content": (
                f"{exc}\n\n"
                "Please fix the issue(s) in your CSV file(s) and type **ready** when done."
            ),
        })
        return {
            **state,
            "messages": messages,
            "current_node": "data_collection",
            "needs_user_input": True,
        }

    table_names = ", ".join(user_data.raw_tables.keys())
    messages.append({
        "role": "assistant",
        "content": f"All CSV files loaded successfully (tables: {table_names}). Proceeding to build the model.",
    })

    result = {
        **state,
        "messages": messages,
        "user_data": user_data,
        "current_node": "data_collection",
        "needs_user_input": False,
    }
    if problem:
        result["problem"] = problem
    return result
