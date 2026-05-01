"""OpenAI LLM provider."""

from __future__ import annotations

import json
import re

from pydantic import BaseModel

from .base import BaseLLM, ChatResponse, ToolCall, ToolDefinition

# Matches <think>...</think> blocks emitted by reasoning models (e.g. deepseek-reasoner)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# Model name prefixes that are reasoning models
_REASONING_MODEL_PREFIXES = ("deepseek-r", "o1", "o3")

# Minimum max_tokens for reasoning models (thinking tokens compete with output tokens)
_REASONING_MIN_TOKENS = 16384


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from reasoning model output."""
    return _THINK_RE.sub("", text).strip()


def _is_reasoning_model(model: str) -> bool:
    model_lower = model.lower()
    return any(model_lower.startswith(p) for p in _REASONING_MODEL_PREFIXES)


class OpenAILLM(BaseLLM):
    """OpenAI chat completion provider."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 4096,
        timeout: float = 120.0,
        max_retries: int = 2,
        temperature: float = 0.0,
    ):
        import openai

        kwargs: dict = {"timeout": timeout}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._client = openai.OpenAI(**kwargs)
        self._model = model
        # Reasoning models need more tokens since thinking competes with output tokens
        if _is_reasoning_model(model):
            self._max_tokens = max(max_tokens, _REASONING_MIN_TOKENS)
        else:
            self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._is_reasoning = _is_reasoning_model(model)
        # Reasoning models don't support temperature — they use fixed sampling.
        self._temperature = None if self._is_reasoning else temperature
        # Newer OpenAI models require max_completion_tokens instead of max_tokens.
        # Third-party compatible APIs (e.g. DeepSeek) always set a base_url and
        # still expect max_tokens, so only switch when using the real OpenAI endpoint.
        self._max_tokens_param = "max_tokens" if base_url else "max_completion_tokens"
        # Track whether a custom base_url is set — third-party providers only support
        # json_object mode, not the full structured outputs (json_schema) API.
        self._has_base_url = base_url is not None
        self._timeout_exceptions = self._resolve_timeout_exc()
        self._rate_limit_exceptions = self._resolve_rate_limit_exc()
        super().__init__()

    @staticmethod
    def _resolve_timeout_exc() -> tuple:
        import openai
        return (openai.APITimeoutError, openai.APIConnectionError)

    @staticmethod
    def _resolve_rate_limit_exc() -> tuple:
        import openai
        return (openai.RateLimitError,)

    def chat(self, messages: list[dict]) -> str:
        messages = self._sanitize_messages(messages)

        def _call():
            kwargs: dict = {
                "model": self._model,
                "messages": messages,
                self._max_tokens_param: self._max_tokens,
            }
            if self._temperature is not None:
                kwargs["temperature"] = self._temperature
            response = self._client.chat.completions.create(**kwargs)
            if response.usage:
                self._add_usage(response.usage.prompt_tokens, response.usage.completion_tokens)
            content = response.choices[0].message.content or ""
            return _strip_thinking(content) if self._is_reasoning else content

        return self._retry(_call, self._max_retries, self._timeout_exceptions, self._rate_limit_exceptions)

    def structured_output(
        self, messages: list[dict], schema: type[BaseModel]
    ) -> BaseModel:
        messages = self._sanitize_messages(messages)

        # Native structured outputs (Level 3): real OpenAI endpoints, non-reasoning models.
        # base_url providers (DeepSeek, etc.) and reasoning models fall back to Level 2.
        if not self._has_base_url and not self._is_reasoning:
            return self._structured_output_native(messages, schema)
        return self._structured_output_fallback(messages, schema)

    def _structured_output_native(
        self, messages: list[dict], schema: type[BaseModel]
    ) -> BaseModel:
        """Level 3: schema-constrained output via beta.chat.completions.parse()."""
        def _call():
            kwargs: dict = {
                "model": self._model,
                "messages": messages,
                self._max_tokens_param: self._max_tokens,
                "response_format": schema,
            }
            if self._temperature is not None:
                kwargs["temperature"] = self._temperature
            response = self._client.beta.chat.completions.parse(**kwargs)
            if response.usage:
                self._add_usage(response.usage.prompt_tokens, response.usage.completion_tokens)
            parsed = response.choices[0].message.parsed
            if parsed is None:
                raise ValueError("Structured output parsing returned None")
            return parsed

        return self._retry(_call, self._max_retries, self._timeout_exceptions, self._rate_limit_exceptions)

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[ToolDefinition],
    ) -> ChatResponse:
        messages = self._sanitize_messages(messages)
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]

        def _call():
            kwargs: dict = {
                "model": self._model,
                "messages": messages,
                self._max_tokens_param: self._max_tokens,
                "tools": openai_tools,
                "tool_choice": "auto",
            }
            if self._temperature is not None:
                kwargs["temperature"] = self._temperature
            response = self._client.chat.completions.create(**kwargs)
            if response.usage:
                self._add_usage(response.usage.prompt_tokens, response.usage.completion_tokens)
            message = response.choices[0].message

            text = message.content
            tool_calls = []
            if message.tool_calls:
                for tc in message.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        # Arguments were truncated or malformed — surface as an error
                        # so the tool loop can feed it back to the LLM for a retry.
                        args = {"_parse_error": (
                            "Tool arguments could not be parsed as JSON (likely truncated). "
                            "Please resubmit with valid, complete JSON."
                        )}
                    tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
            return ChatResponse(text=text, tool_calls=tool_calls, _payload=message)

        return self._retry(_call, self._max_retries, self._timeout_exceptions, self._rate_limit_exceptions)

    def extend_messages(
        self,
        messages: list[dict],
        response: ChatResponse,
        results: dict[str, str],
    ) -> list[dict]:
        new_messages = list(messages)
        message = response._payload  # ChatCompletionMessage

        assistant_dict: dict = {"role": "assistant", "content": message.content}
        # DeepSeek thinking-mode models return reasoning_content that must be echoed back
        reasoning_content = getattr(message, "reasoning_content", None)
        if reasoning_content:
            assistant_dict["reasoning_content"] = reasoning_content
        if message.tool_calls:
            assistant_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in message.tool_calls
            ]
        new_messages.append(assistant_dict)

        for tc in response.tool_calls:
            new_messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": results.get(tc.id, ""),
            })
        return new_messages

    def _structured_output_fallback(
        self, messages: list[dict], schema: type[BaseModel]
    ) -> BaseModel:
        """Level 2 fallback: json_object mode + prompt engineering for base_url / reasoning models."""
        schema_json = schema.model_json_schema()
        system_suffix = (
            f"\n\nRespond ONLY with valid JSON matching this schema:\n"
            f"{json.dumps(schema_json, indent=2)}"
        )

        augmented = list(messages)
        if augmented and augmented[0]["role"] == "system":
            augmented[0] = {
                **augmented[0],
                "content": augmented[0]["content"] + system_suffix,
            }
        else:
            augmented.insert(0, {"role": "system", "content": system_suffix.strip()})

        def _call():
            kwargs: dict = {
                "model": self._model,
                "messages": augmented,
                self._max_tokens_param: self._max_tokens,
            }
            # response_format=json_object is not supported by reasoning model endpoints
            if not self._is_reasoning:
                kwargs["response_format"] = {"type": "json_object"}
            if self._temperature is not None:
                kwargs["temperature"] = self._temperature
            response = self._client.chat.completions.create(**kwargs)
            if response.usage:
                self._add_usage(response.usage.prompt_tokens, response.usage.completion_tokens)
            raw = response.choices[0].message.content or "{}"
            if self._is_reasoning:
                raw = _strip_thinking(raw)
            return schema.model_validate_json(self._strip_json_fences(raw))

        return self._retry(_call, self._max_retries, self._timeout_exceptions, self._rate_limit_exceptions)
