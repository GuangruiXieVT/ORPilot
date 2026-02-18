"""Anthropic LLM provider."""

from __future__ import annotations

import json

from pydantic import BaseModel

from .base import BaseLLM


class AnthropicLLM(BaseLLM):
    """Anthropic Claude provider."""

    def __init__(self, model: str = "claude-sonnet-4-5-20250929", api_key: str | None = None):
        import anthropic

        kwargs: dict = {}
        if api_key:
            kwargs["api_key"] = api_key
        self._client = anthropic.Anthropic(**kwargs)
        self._model = model

    def chat(self, messages: list[dict]) -> str:
        system = None
        chat_messages = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                chat_messages.append(m)

        kwargs: dict = {
            "model": self._model,
            "max_tokens": 4096,
            "messages": chat_messages,
        }
        if system:
            kwargs["system"] = system

        response = self._client.messages.create(**kwargs)
        return response.content[0].text

    def structured_output(
        self, messages: list[dict], schema: type[BaseModel]
    ) -> BaseModel:
        schema_json = schema.model_json_schema()
        suffix = (
            f"\n\nRespond ONLY with valid JSON matching this schema:\n"
            f"{json.dumps(schema_json, indent=2)}"
        )

        system = None
        chat_messages = []
        for m in messages:
            if m["role"] == "system":
                system = (m["content"] or "") + suffix
            else:
                chat_messages.append(m)
        if system is None:
            system = suffix.strip()

        response = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=system,
            messages=chat_messages,
        )
        raw = response.content[0].text

        # Extract JSON from possible markdown fencing
        if "```" in raw:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start != -1 and end > start:
                raw = raw[start:end]

        return schema.model_validate_json(raw)
