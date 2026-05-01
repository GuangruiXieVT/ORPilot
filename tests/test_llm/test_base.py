"""Tests for LLM abstraction."""

from pydantic import BaseModel

from orpilot.llm.base import BaseLLM, ChatResponse, ToolDefinition
from orpilot.llm.config import LLMConfig


class DummyResponse(BaseModel):
    name: str
    value: int


class MockLLM(BaseLLM):
    """Minimal mock LLM for unit testing."""

    def __init__(self, responses: list[str] | None = None):
        super().__init__()
        self._responses = responses or ["Hello"]
        self._call_count = 0

    def chat(self, messages: list[dict]) -> str:
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return self._responses[idx]

    def structured_output(self, messages: list[dict], schema: type[BaseModel]) -> BaseModel:
        self._call_count += 1
        if schema is DummyResponse:
            return DummyResponse(name="test", value=42)
        return schema()

    def chat_with_tools(self, messages: list[dict], tools: list[ToolDefinition]) -> ChatResponse:
        return ChatResponse(text="done", tool_calls=[])

    def extend_messages(self, messages: list[dict], response: ChatResponse, results: dict[str, str]) -> list[dict]:
        return messages


def test_mock_llm_chat():
    llm = MockLLM(responses=["Hi there", "How can I help?"])
    assert llm.chat([{"role": "user", "content": "hello"}]) == "Hi there"
    assert llm.chat([{"role": "user", "content": "help"}]) == "How can I help?"


def test_mock_llm_structured():
    llm = MockLLM()
    result = llm.structured_output([], DummyResponse)
    assert result.name == "test"
    assert result.value == 42


def test_mock_llm_usage_tracking():
    llm = MockLLM()
    assert llm.get_usage() == {"input_tokens": 0, "output_tokens": 0}
    llm._add_usage(100, 50)
    assert llm.get_usage() == {"input_tokens": 100, "output_tokens": 50}
    llm._add_usage(20, 10)
    assert llm.get_usage() == {"input_tokens": 120, "output_tokens": 60}
    llm.reset_usage()
    assert llm.get_usage() == {"input_tokens": 0, "output_tokens": 0}


def test_sanitize_messages_merges_consecutive_user():
    """Consecutive user messages are merged into one."""
    msgs = [
        {"role": "user", "content": "Hello"},
        {"role": "user", "content": "World"},
    ]
    result = MockLLM._sanitize_messages(msgs)
    assert len(result) == 1
    assert "Hello" in result[0]["content"]
    assert "World" in result[0]["content"]


def test_sanitize_messages_does_not_merge_tool_messages():
    """Tool messages must never be merged — each has a distinct tool_call_id."""
    msgs = [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "tc1", "type": "function", "function": {"name": "f", "arguments": "{}"}},
            {"id": "tc2", "type": "function", "function": {"name": "g", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "tc1", "content": "result_1"},
        {"role": "tool", "tool_call_id": "tc2", "content": "result_2"},
    ]
    result = MockLLM._sanitize_messages(msgs)
    # Must remain three separate messages; merging tool messages drops tool_call_id
    assert len(result) == 3
    tool_msgs = [m for m in result if m.get("role") == "tool"]
    assert len(tool_msgs) == 2
    ids = {m["tool_call_id"] for m in tool_msgs}
    assert ids == {"tc1", "tc2"}


def test_sanitize_messages_does_not_merge_system():
    """System messages are never merged (there should only be one, but guard anyway)."""
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "system", "content": "Be concise."},
    ]
    result = MockLLM._sanitize_messages(msgs)
    assert len(result) == 2


def test_llm_config_defaults():
    config = LLMConfig()
    assert config.provider == "openai"
    assert config.model is None
    assert config.api_key is None


def test_llm_config_google_provider():
    config = LLMConfig(provider="google", model="gemini-2.0-flash")
    assert config.provider == "google"
    assert config.model == "gemini-2.0-flash"
