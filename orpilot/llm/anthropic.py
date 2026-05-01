"""Anthropic LLM provider."""

from __future__ import annotations

from pydantic import BaseModel

from .base import BaseLLM, ChatResponse, ToolCall, ToolDefinition


class AnthropicLLM(BaseLLM):
    """Anthropic Claude provider."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
        max_tokens: int = 4096,
        timeout: float = 120.0,
        max_retries: int = 2,
        temperature: float = 0.0,
    ):
        import anthropic

        kwargs: dict = {"timeout": timeout}
        if api_key:
            kwargs["api_key"] = api_key
        self._client = anthropic.Anthropic(**kwargs)
        self._model = model
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._temperature = temperature
        self._timeout_exceptions = self._resolve_timeout_exc()
        self._rate_limit_exceptions = self._resolve_rate_limit_exc()
        super().__init__()

    @staticmethod
    def _resolve_timeout_exc() -> tuple:
        import anthropic
        return (anthropic.APITimeoutError, anthropic.APIConnectionError)

    @staticmethod
    def _resolve_rate_limit_exc() -> tuple:
        import anthropic
        return (anthropic.RateLimitError,)

    def chat(self, messages: list[dict]) -> str:
        messages = self._sanitize_messages(messages)
        system = None
        chat_messages = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                chat_messages.append(m)

        # Anthropic requires at least one user message; add a placeholder when
        # the caller only passed a system prompt (e.g. first interview turn).
        if not chat_messages:
            chat_messages = [{"role": "user", "content": "Please begin."}]

        kwargs: dict = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": chat_messages,
            "temperature": self._temperature,
        }
        if system:
            kwargs["system"] = system

        def _call():
            response = self._client.messages.create(**kwargs)
            self._add_usage(response.usage.input_tokens, response.usage.output_tokens)
            return response.content[0].text

        return self._retry(_call, self._max_retries, self._timeout_exceptions, self._rate_limit_exceptions)

    def structured_output(
        self, messages: list[dict], schema: type[BaseModel]
    ) -> BaseModel:
        messages = self._sanitize_messages(messages)
        system = None
        chat_messages = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                chat_messages.append(m)
        if not chat_messages:
            chat_messages = [{"role": "user", "content": "Please proceed."}]

        tool_name = schema.__name__
        tool = {
            "name": tool_name,
            "description": f"Return the structured {tool_name} result.",
            "input_schema": schema.model_json_schema(),
        }

        def _call():
            kwargs: dict = {
                "model": self._model,
                "max_tokens": self._max_tokens,
                "temperature": self._temperature,
                "tools": [tool],
                "tool_choice": {"type": "tool", "name": tool_name},
                "messages": chat_messages,
            }
            if system:
                kwargs["system"] = system
            response = self._client.messages.create(**kwargs)
            self._add_usage(response.usage.input_tokens, response.usage.output_tokens)
            for block in response.content:
                if block.type == "tool_use":
                    return schema.model_validate(block.input)
            raise ValueError("No tool_use block in response")

        return self._retry(_call, self._max_retries, self._timeout_exceptions, self._rate_limit_exceptions)

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[ToolDefinition],
    ) -> ChatResponse:
        messages = self._sanitize_messages(messages)
        system = None
        chat_messages = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                chat_messages.append(m)
        if not chat_messages:
            chat_messages = [{"role": "user", "content": "Please proceed."}]

        anthropic_tools = [
            {"name": t.name, "description": t.description, "input_schema": t.parameters}
            for t in tools
        ]

        def _call():
            kwargs: dict = {
                "model": self._model,
                "max_tokens": self._max_tokens,
                "temperature": self._temperature,
                "tools": anthropic_tools,
                "tool_choice": {"type": "auto"},
                "messages": chat_messages,
            }
            if system:
                kwargs["system"] = system
            response = self._client.messages.create(**kwargs)
            self._add_usage(response.usage.input_tokens, response.usage.output_tokens)

            text = None
            tool_calls = []
            for block in response.content:
                if block.type == "text":
                    text = (text or "") + block.text
                elif block.type == "tool_use":
                    tool_calls.append(
                        ToolCall(id=block.id, name=block.name, arguments=block.input)
                    )
            return ChatResponse(text=text, tool_calls=tool_calls, _payload=response.content)

        return self._retry(_call, self._max_retries, self._timeout_exceptions, self._rate_limit_exceptions)

    def extend_messages(
        self,
        messages: list[dict],
        response: ChatResponse,
        results: dict[str, str],
    ) -> list[dict]:
        new_messages = list(messages)
        new_messages.append({"role": "assistant", "content": response._payload})
        tool_results = [
            {"type": "tool_result", "tool_use_id": tc.id, "content": results.get(tc.id, "")}
            for tc in response.tool_calls
        ]
        new_messages.append({"role": "user", "content": tool_results})
        return new_messages
