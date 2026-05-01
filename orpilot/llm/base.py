"""Abstract base class for LLM providers."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


@dataclass
class ToolDefinition:
    """Provider-agnostic tool (function) specification."""
    name: str
    description: str
    parameters: dict  # JSON Schema {type: object, properties: {...}, required: [...]}


@dataclass
class ToolCall:
    """A single tool invocation requested by the LLM."""
    id: str
    name: str
    arguments: dict


@dataclass
class ChatResponse:
    """Result from chat_with_tools. Either text or tool_calls will be populated."""
    text: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    # Opaque provider-specific payload used by extend_messages to reconstruct the
    # assistant turn for the next API request. Nodes must not inspect this directly.
    _payload: Any = field(default=None, repr=False)

    @property
    def is_tool_use(self) -> bool:
        return bool(self.tool_calls)

    def find(self, name: str) -> ToolCall | None:
        """Return the first tool call with the given name, or None."""
        for tc in self.tool_calls:
            if tc.name == name:
                return tc
        return None


class BaseLLM(ABC):
    """Unified interface for LLM providers."""

    def __init__(self) -> None:
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0

    def _add_usage(self, input_tokens: int, output_tokens: int) -> None:
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens

    def get_usage(self) -> dict:
        """Return cumulative token usage since last reset_usage() call."""
        return {"input_tokens": self._total_input_tokens, "output_tokens": self._total_output_tokens}

    def reset_usage(self) -> None:
        """Reset cumulative token counters to zero."""
        self._total_input_tokens = 0
        self._total_output_tokens = 0

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

    @abstractmethod
    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[ToolDefinition],
    ) -> ChatResponse:
        """Send messages with tool definitions.

        Returns a ChatResponse whose tool_calls are populated when the LLM wants
        to call tools, or whose text is populated when the LLM gives a final reply.
        """
        ...

    @abstractmethod
    def extend_messages(
        self,
        messages: list[dict],
        response: ChatResponse,
        results: dict[str, str],
    ) -> list[dict]:
        """Append the assistant tool-use turn and its results to the message list.

        results maps tool_call.id → result string. Returns the extended list.
        Only call this when response.is_tool_use is True.
        """
        ...

    @staticmethod
    def _strip_json_fences(text: str) -> str:
        """Strip markdown code fences from a JSON response."""
        if "```" in text:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                return text[start:end]
        return text.strip()

    @staticmethod
    def _sanitize_messages(messages: list[dict]) -> list[dict]:
        """Merge consecutive same-role text messages to satisfy API alternation rules.

        Skips messages whose content is a list (tool-use blocks, tool results) so
        provider-specific structured content is never flattened into a string.
        """
        if not messages:
            return messages
        result: list[dict] = [messages[0]]
        for msg in messages[1:]:
            last = result[-1]
            if (
                msg.get("role") == last.get("role")
                and msg.get("role") not in ("system", "tool")
                and isinstance(msg.get("content"), str)
                and isinstance(last.get("content"), str)
            ):
                result[-1] = {**last, "content": last["content"] + "\n\n" + msg["content"]}
            else:
                result.append(msg)
        return result

    def _retry(
        self,
        fn: Callable[[], T],
        max_retries: int,
        timeout_exceptions: tuple,
        rate_limit_exceptions: tuple = (),
        rate_limit_wait: int = 60,
    ) -> T:
        """Call fn, retrying up to max_retries times on timeout or rate-limit errors."""
        for attempt in range(max_retries + 1):
            try:
                return fn()
            except rate_limit_exceptions as exc:
                if attempt == max_retries:
                    raise
                print(f"[ORPilot] Rate limit hit (attempt {attempt + 1}/{max_retries + 1}), "
                      f"retrying in {rate_limit_wait}s…")
                time.sleep(rate_limit_wait)
            except timeout_exceptions as exc:
                if attempt == max_retries:
                    raise
                wait = 5 * (2 ** attempt)
                print(f"[ORPilot] API timeout (attempt {attempt + 1}/{max_retries + 1}), "
                      f"retrying in {wait}s…")
                time.sleep(wait)
        raise RuntimeError("unreachable")
