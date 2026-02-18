"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod

from pydantic import BaseModel


class BaseLLM(ABC):
    """Unified interface for LLM providers."""

    @abstractmethod
    def chat(self, messages: list[dict]) -> str:
        """Send messages and return the assistant's text reply."""
        ...

    @abstractmethod
    def structured_output(
        self, messages: list[dict], schema: type[BaseModel]
    ) -> BaseModel:
        """Send messages and parse the response into a Pydantic model."""
        ...
