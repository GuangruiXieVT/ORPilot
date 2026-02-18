"""Main LangGraph graph definition for the ORPilot workflow."""

from __future__ import annotations

from typing import Any

from langgraph.graph import StateGraph, END

from orpilot.llm.base import BaseLLM
from orpilot.llm.config import LLMConfig, get_llm
from orpilot.workflow.state import WorkflowState
from orpilot.workflow.nodes.interview import interview_node
from orpilot.workflow.nodes.data_collection import data_collection_node
from orpilot.workflow.nodes.model_builder import model_builder_node
from orpilot.workflow.nodes.solver_runner import solver_runner_node
from orpilot.workflow.nodes.reporter import reporter_node
from orpilot.workflow import edges


def build_graph(
    llm: BaseLLM | None = None,
    llm_config: LLMConfig | None = None,
) -> StateGraph:
    """Build the ORPilot workflow graph.

    Args:
        llm: Pre-configured LLM instance. If None, created from llm_config.
        llm_config: LLM configuration. Used only if llm is None.

    Returns:
        A compiled LangGraph StateGraph ready for execution.
    """
    if llm is None:
        llm = get_llm(llm_config)

    graph = StateGraph(WorkflowState)

    # Add nodes — bind the LLM where needed
    graph.add_node("interview", lambda state: interview_node(state, llm))
    graph.add_node("data_collection", lambda state: data_collection_node(state, llm))
    graph.add_node("model_builder", lambda state: model_builder_node(state, llm))
    graph.add_node("solver_runner", lambda state: solver_runner_node(state))
    graph.add_node("reporter", lambda state: reporter_node(state, llm))

    # A no-op node that signals "waiting for user input"
    graph.add_node("wait_for_input", lambda state: {**state, "needs_user_input": True})

    # Set entry point
    graph.set_entry_point("interview")

    # Add conditional edges
    graph.add_conditional_edges("interview", edges.after_interview)
    graph.add_conditional_edges("data_collection", edges.after_data_collection)

    # model_builder always goes to solver_runner
    graph.add_edge("model_builder", "solver_runner")

    # solver_runner routes based on result
    graph.add_conditional_edges("solver_runner", edges.after_solver_runner)

    # reporter is terminal
    graph.add_edge("reporter", END)

    # wait_for_input is terminal (caller feeds input and re-invokes)
    graph.add_edge("wait_for_input", END)

    return graph.compile()
