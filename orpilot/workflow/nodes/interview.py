"""Interview node — elicit the user's business optimization need."""

from __future__ import annotations

from orpilot.llm.base import BaseLLM
from orpilot.models.problem import ProblemDefinition
from orpilot.prompts import interview as interview_prompts
from orpilot.workflow.state import WorkflowState


def _compress_interview_ctx(problem: ProblemDefinition) -> list[dict]:
    """Reduce N interview turns to 2 messages using the extracted problem as the summary."""
    return [
        {"role": "user", "content": "I need help with an optimization problem."},
        {
            "role": "assistant",
            "content": f"[Interview complete. Problem extracted:]\n{problem.model_dump_json(indent=2)}",
        },
    ]


def interview_node(state: WorkflowState, llm: BaseLLM) -> WorkflowState:
    """Run one turn of the interview conversation.

    Uses messages_ctx (compressed) for LLM calls to keep token usage bounded.
    The full verbatim history is preserved in messages for display and session resume.
    """
    if state.get("problem") is not None:
        return state

    messages = list(state.get("messages", []))
    # Fall back to full messages if messages_ctx has not been initialised yet
    messages_ctx = list(state["messages_ctx"]) if state.get("messages_ctx") is not None else list(messages)

    llm_messages = [{"role": "system", "content": interview_prompts.SYSTEM_PROMPT}]
    llm_messages.extend(messages_ctx)
    if len(llm_messages) == 1:
        llm_messages.append({"role": "user", "content": "Hello, I need help with an optimization problem."})

    response = llm.chat(llm_messages)

    messages.append({"role": "assistant", "content": response})
    messages_ctx.append({"role": "assistant", "content": response})

    updates: dict = {
        "messages": messages,
        "messages_ctx": messages_ctx,
        "current_node": "interview",
    }

    if "[INTERVIEW_COMPLETE]" in response:
        conversation_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in messages
        )
        summary_prompt = interview_prompts.SUMMARIZE_PROMPT.format(
            conversation=conversation_text
        )
        problem = llm.structured_output(
            [
                {"role": "system", "content": "Extract structured problem definition."},
                {"role": "user", "content": summary_prompt},
            ],
            ProblemDefinition,
        )

        messages[-1] = {
            "role": "assistant",
            "content": response.replace("[INTERVIEW_COMPLETE]", "").strip(),
        }
        updates["messages"] = messages
        updates["messages_ctx"] = _compress_interview_ctx(problem)
        updates["problem"] = problem
        updates["needs_user_input"] = False
    else:
        updates["needs_user_input"] = True

    return {**state, **updates}
