"""OpenAI LLM provider."""

from __future__ import annotations

import json

from pydantic import BaseModel

from .base import BaseLLM


class OpenAILLM(BaseLLM):
    """OpenAI chat completion provider."""

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None, base_url: str | None = None):
        import openai

        kwargs: dict = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._client = openai.OpenAI(**kwargs)
        self._model = model

    def chat(self, messages: list[dict]) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
        )
        return response.choices[0].message.content or ""

    def structured_output(
        self, messages: list[dict], schema: type[BaseModel]
    ) -> BaseModel:
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

        response = self._client.chat.completions.create(
            model=self._model,
            messages=augmented,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        return schema.model_validate_json(raw)
