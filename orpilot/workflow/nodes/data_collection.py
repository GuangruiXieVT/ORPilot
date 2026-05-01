"""Data collection node — guide user to provide CSV data files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from orpilot.codegen.data_validator import validate_collected_data
from orpilot.llm.base import BaseLLM
from orpilot.models.data import CsvFileSpec, UserData
from orpilot.paths import DATA_DIR
from orpilot.prompts import data_guide
from orpilot.workflow.state import WorkflowState


def _compress_data_ctx(
    interview_ctx: list[dict],
    user_data: UserData,
) -> list[dict]:
    """Reduce data-collection turns to 2 messages appended to the interview summary."""
    table_names = list(user_data.raw_tables.keys())
    return list(interview_ctx) + [
        {"role": "user", "content": "My data files are ready."},
        {
            "role": "assistant",
            "content": f"[Data collection complete. Tables loaded: {table_names}]",
        },
    ]


def _get_ctx(state: WorkflowState) -> list[dict]:
    """Return messages_ctx if set, otherwise fall back to the full messages list."""
    ctx = state.get("messages_ctx")
    return list(ctx) if ctx is not None else list(state.get("messages", []))


def data_collection_node(state: WorkflowState, llm: BaseLLM) -> WorkflowState:
    """Guide the user to provide CSV data for the OR model.

    Phase 1 (spec): LLM defines CSV specs via conversation.  When the LLM
    signals ``[DATA_SPEC_READY]``, extract ``CsvFileSpec`` list and ask the
    user to place files in ``data_dir``.

    Phase 2 (confirm): LLM stays in the conversation so the user can ask
    questions or request changes.  The LLM signals ``[LOAD_DATA]`` when
    the user confirms files are ready, or emits a new ``[DATA_SPEC_READY]``
    if the requirements change.
    """
    # Already loaded (session resume past this node) — pass through like interview_node does.
    if state.get("user_data") is not None:
        return state

    csv_specs: list[dict[str, Any]] = state.get("csv_specs", [])

    if csv_specs:
        return _phase_confirm(state, csv_specs, llm)
    return _phase_spec(state, llm)


def _phase_spec(state: WorkflowState, llm: BaseLLM) -> WorkflowState:
    """Phase 1: LLM defines the CSV file specifications."""
    messages = list(state.get("messages", []))
    messages_ctx = _get_ctx(state)
    problem = state.get("problem")
    data_dir = state.get("data_dir", str(DATA_DIR))

    problem_json = problem.model_dump_json(indent=2) if problem else "{}"
    system_prompt = data_guide.SYSTEM_PROMPT.format(problem_json=problem_json)

    llm_messages = [{"role": "system", "content": system_prompt}]
    llm_messages.extend(messages_ctx)
    if llm_messages[-1]["role"] == "assistant":
        llm_messages.append({"role": "user", "content": "Please proceed."})

    response = llm.chat(llm_messages)

    messages.append({"role": "assistant", "content": response})
    messages_ctx.append({"role": "assistant", "content": response})

    updates: dict[str, Any] = {
        "messages": messages,
        "messages_ctx": messages_ctx,
        "current_node": "data_collection",
    }

    from pydantic import BaseModel, Field

    class _CsvSpecList(BaseModel):
        specs: list[CsvFileSpec] = Field(default_factory=list)

    signal_present = "[DATA_SPEC_READY]" in response

    # Fallback: if the LLM generated a schema but forgot the signal, detect it
    # by running spec extraction regardless. Only treat as ready if specs are found.
    if not signal_present:
        try:
            conversation_text = "\n".join(
                f"{m['role']}: {m['content']}" for m in messages
            )
            fallback_extract = data_guide.SPEC_EXTRACTION_PROMPT.format(
                conversation=conversation_text,
            )
            fallback_result = llm.structured_output(
                [
                    {"role": "system", "content": "Extract CSV file specifications."},
                    {"role": "user", "content": fallback_extract},
                ],
                _CsvSpecList,
            )
            if fallback_result.specs:
                signal_present = True
                spec_dicts = [s.model_dump() for s in fallback_result.specs]
        except Exception:
            spec_dicts = []
    else:
        spec_dicts = None  # extracted below

    if signal_present:
        if spec_dicts is None:
            conversation_text = "\n".join(
                f"{m['role']}: {m['content']}" for m in messages
            )
            extract_prompt = data_guide.SPEC_EXTRACTION_PROMPT.format(
                conversation=conversation_text,
            )
            spec_result = llm.structured_output(
                [
                    {"role": "system", "content": "Extract CSV file specifications."},
                    {"role": "user", "content": extract_prompt},
                ],
                _CsvSpecList,
            )
            spec_dicts = [s.model_dump() for s in spec_result.specs]

        updates["csv_specs"] = spec_dicts

        clean = response.replace("[DATA_SPEC_READY]", "").strip()
        ready_msg = (
            f"{clean}\n\n"
            f"Please place the CSV files in: `{data_dir}`\n"
            "Type **ready** when the files are in place."
        )
        messages[-1] = {"role": "assistant", "content": ready_msg}
        messages_ctx[-1] = {"role": "assistant", "content": ready_msg}
        updates["messages"] = messages
        updates["messages_ctx"] = messages_ctx
        updates["needs_user_input"] = True
    else:
        updates["needs_user_input"] = True

    return {**state, **updates}


def _phase_confirm(
    state: WorkflowState,
    csv_spec_dicts: list[dict[str, Any]],
    llm: BaseLLM,
) -> WorkflowState:
    """Phase 2: LLM-mediated confirmation loop."""
    messages = list(state.get("messages", []))
    messages_ctx = _get_ctx(state)
    problem = state.get("problem")
    data_dir = state.get("data_dir", str(DATA_DIR))

    problem_json = problem.model_dump_json(indent=2) if problem else "{}"
    system_prompt = data_guide.SYSTEM_PROMPT.format(problem_json=problem_json)

    llm_messages = [{"role": "system", "content": system_prompt}]
    llm_messages.extend(messages_ctx)
    if llm_messages[-1]["role"] == "assistant":
        llm_messages.append({"role": "user", "content": "Please proceed."})

    response = llm.chat(llm_messages)

    messages.append({"role": "assistant", "content": response})
    messages_ctx.append({"role": "assistant", "content": response})

    updates: dict[str, Any] = {
        "messages": messages,
        "messages_ctx": messages_ctx,
        "current_node": "data_collection",
    }

    # ------------------------------------------------------------------ #
    # 1. User confirmed — try to load files                               #
    # ------------------------------------------------------------------ #
    if "[LOAD_DATA]" in response:
        specs = [CsvFileSpec.model_validate(d) for d in csv_spec_dicts]
        clean = response.replace("[LOAD_DATA]", "").strip()

        try:
            user_data = UserData.load_from_csv_dir(data_dir, specs)
            if problem:
                csv_paths = {
                    Path(spec.filename).stem: str((Path(data_dir) / spec.filename).resolve())
                    for spec in specs
                }
                problem = problem.model_copy(update={"csv_file_paths": csv_paths})
        except FileNotFoundError as exc:
            msg = (
                f"{clean}\n\n"
                f"⚠️ {exc}\n\n"
                f"Please place the missing file(s) in `{data_dir}` and type **ready**."
            )
            messages[-1] = {"role": "assistant", "content": msg}
            messages_ctx[-1] = {"role": "assistant", "content": msg}
            updates["messages"] = messages
            updates["messages_ctx"] = messages_ctx
            updates["needs_user_input"] = True
            return {**state, **updates}
        except ValueError as exc:
            msg = (
                f"{clean}\n\n"
                f"⚠️ {exc}\n\n"
                "Please fix the issue(s) in your CSV file(s) and type **ready** when done."
            )
            messages[-1] = {"role": "assistant", "content": msg}
            messages_ctx[-1] = {"role": "assistant", "content": msg}
            updates["messages"] = messages
            updates["messages_ctx"] = messages_ctx
            updates["needs_user_input"] = True
            return {**state, **updates}

        data_errors = validate_collected_data(user_data)
        if data_errors:
            error_list = "\n".join(f"  - {e}" for e in data_errors)
            msg = (
                f"{clean}\n\n"
                f"⚠️ Data validation found the following issue(s):\n{error_list}\n\n"
                "Please fix your CSV file(s) and type **ready** when done."
            )
            messages[-1] = {"role": "assistant", "content": msg}
            messages_ctx[-1] = {"role": "assistant", "content": msg}
            updates["messages"] = messages
            updates["messages_ctx"] = messages_ctx
            updates["needs_user_input"] = True
            return {**state, **updates}

        table_names = ", ".join(user_data.raw_tables.keys())
        success_msg = (
            f"{clean}\n\n"
            f"All CSV files loaded successfully (tables: {table_names}). "
            "Proceeding to build the model."
        )
        messages[-1] = {"role": "assistant", "content": success_msg}
        updates["messages"] = messages

        # Compress messages_ctx: keep the interview summary (first 2 msgs) + data summary
        interview_ctx = messages_ctx[:2]
        updates["messages_ctx"] = _compress_data_ctx(interview_ctx, user_data)
        updates["user_data"] = user_data
        updates["needs_user_input"] = False
        if problem:
            updates["problem"] = problem
        return {**state, **updates}

    # ------------------------------------------------------------------ #
    # 2. User requested spec changes — re-extract and ask to place files  #
    # ------------------------------------------------------------------ #
    if "[DATA_SPEC_READY]" in response:
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

        new_spec_dicts = [s.model_dump() for s in spec_result.specs]
        clean = response.replace("[DATA_SPEC_READY]", "").strip()
        updated_msg = (
            f"{clean}\n\n"
            f"Please place the updated CSV files in: `{data_dir}`\n"
            "Type **ready** when the files are in place."
        )
        messages[-1] = {"role": "assistant", "content": updated_msg}
        messages_ctx[-1] = {"role": "assistant", "content": updated_msg}
        updates["messages"] = messages
        updates["messages_ctx"] = messages_ctx
        updates["csv_specs"] = new_spec_dicts
        updates["needs_user_input"] = True
        return {**state, **updates}

    # ------------------------------------------------------------------ #
    # 3. Conversational reply — keep talking                              #
    # ------------------------------------------------------------------ #
    updates["needs_user_input"] = True
    return {**state, **updates}
