"""Tests for LLM abstraction."""

from pydantic import BaseModel

from orpilot.llm.base import BaseLLM
from orpilot.llm.config import LLMConfig


class DummyResponse(BaseModel):
    name: str
    value: int


class MockLLM(BaseLLM):
    """Mock LLM for testing."""

    def __init__(self, responses: list[str] | None = None):
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


def test_mock_llm_chat():
    llm = MockLLM(responses=["Hi there", "How can I help?"])
    assert llm.chat([{"role": "user", "content": "hello"}]) == "Hi there"
    assert llm.chat([{"role": "user", "content": "help"}]) == "How can I help?"


def test_mock_llm_structured():
    llm = MockLLM()
    result = llm.structured_output([], DummyResponse)
    assert result.name == "test"
    assert result.value == 42


def test_llm_config_defaults():
    config = LLMConfig()
    assert config.provider == "openai"
    assert config.model is None
    assert config.api_key is None
