"""ORPilot — AI Operations Research Agent Library.

Usage::

    from orpilot import Agent

    agent = Agent(llm_provider="openai", solver="pulp")
    result = agent.run()  # interactive
    # or
    result = agent.run(problem="minimize cost", data={"sources": ...})
"""

from __future__ import annotations

from typing import Any

from orpilot.llm.config import LLMConfig, get_llm
from orpilot.models.problem import ProblemDefinition
from orpilot.models.data import UserData, DataParameter
from orpilot.models.solution import SolutionResult
from orpilot.workflow.graph import build_graph


class Agent:
    """High-level Python API for the ORPilot agent."""

    def __init__(
        self,
        llm_provider: str = "openai",
        model: str | None = None,
        solver: str = "pulp",
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 3,
        data_dir: str = "./data",
        output_dir: str | None = None,
    ):
        self._llm_config = LLMConfig(
            provider=llm_provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
        )
        self._llm = get_llm(self._llm_config)
        self._solver = solver
        self._max_retries = max_retries
        self._data_dir = data_dir
        self._output_dir = output_dir
        self._graph = build_graph(llm=self._llm)

    def run(
        self,
        problem: str | ProblemDefinition | None = None,
        data: dict[str, Any] | UserData | None = None,
        interactive: bool = True,
        data_dir: str | None = None,
        output_dir: str | None = None,
    ) -> SolutionResult | None:
        """Run the ORPilot agent workflow.

        Args:
            problem: Problem description string or ProblemDefinition.
                If string, starts with an interview seeded by this description.
                If ProblemDefinition, skips interview.
            data: Data dict or UserData. If provided with a ProblemDefinition,
                skips data collection too.
            interactive: If True (default), prompts for user input at stdin.
                If False with string problem, runs interview non-interactively.

        Returns:
            SolutionResult if the workflow completes, None if aborted.
        """
        from pathlib import Path
        from orpilot.workflow.state import WorkflowState

        resolved_dir = data_dir or self._data_dir
        Path(resolved_dir).mkdir(parents=True, exist_ok=True)

        resolved_output = output_dir or self._output_dir or ""
        if resolved_output:
            Path(resolved_output).mkdir(parents=True, exist_ok=True)

        state: WorkflowState = {
            "messages": [],
            "problem": None,
            "user_data": None,
            "ir_model": None,
            "generated_code": "",
            "solution": None,
            "report": "",
            "current_node": "interview",
            "solver_name": self._solver,
            "retry_count": 0,
            "max_retries": self._max_retries,
            "error_context": "",
            "needs_user_input": False,
            "user_input": "",
            "llm_config": self._llm_config.__dict__,
            "data_dir": resolved_dir,
            "csv_specs": [],
            "output_dir": resolved_output,
        }

        # Pre-populate problem
        if isinstance(problem, ProblemDefinition):
            state["problem"] = problem
            state["current_node"] = "data_collection"
        elif isinstance(problem, str):
            state["messages"] = [{"role": "user", "content": problem}]

        # Pre-populate data
        if isinstance(data, UserData):
            state["user_data"] = data
            if state.get("problem"):
                state["current_node"] = "ir_builder"
        elif isinstance(data, dict):
            params = [
                DataParameter(name=k, value=v)
                for k, v in data.items()
            ]
            state["user_data"] = UserData(parameters=params)
            if state.get("problem"):
                state["current_node"] = "ir_builder"

        while True:
            result = self._graph.invoke(state)
            state = result

            if state.get("report"):
                return state.get("solution")

            if state.get("needs_user_input"):
                if not interactive:
                    return None

                messages = state.get("messages", [])
                if messages and messages[-1]["role"] == "assistant":
                    print(f"\nAssistant: {messages[-1]['content']}\n")

                user_input = input("You: ")
                if user_input.strip().lower() in ("quit", "exit", "q"):
                    return None

                state["messages"].append({"role": "user", "content": user_input})
                state["needs_user_input"] = False


__all__ = ["Agent", "SolutionResult"]
