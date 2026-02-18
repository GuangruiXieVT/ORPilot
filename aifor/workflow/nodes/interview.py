"""Interview node — elicit the user's business optimization need."""

from __future__ import annotations

from aifor.llm.base import BaseLLM
from aifor.models.problem import ProblemDefinition
from aifor.prompts import interview as interview_prompts
from aifor.workflow.state import WorkflowState


def interview_node(state: WorkflowState, llm: BaseLLM) -> WorkflowState:
    """Run one turn of the interview conversation.

    If the LLM determines the interview is complete, extract a structured
    ProblemDefinition. Otherwise, set needs_user_input=True to get more info.
    """
    messages = list(state.get("messages", []))

    # Build conversation with the interview system prompt
    llm_messages = [{"role": "system", "content": interview_prompts.SYSTEM_PROMPT}]
    llm_messages.extend(messages)

    response = llm.chat(llm_messages)

    messages.append({"role": "assistant", "content": response})

    updates: dict = {
        "messages": messages,
        "current_node": "interview",
    }

    if "[INTERVIEW_COMPLETE]" in response:
        # Extract structured problem definition
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
        # Clean the marker from the last message
        messages[-1] = {
            "role": "assistant",
            "content": response.replace("[INTERVIEW_COMPLETE]", "").strip(),
        }
        updates["messages"] = messages
        updates["problem"] = problem
        updates["needs_user_input"] = False
    else:
        updates["needs_user_input"] = True

    return {**state, **updates}
