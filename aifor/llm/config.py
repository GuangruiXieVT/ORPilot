"""LLM configuration and provider registry."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .base import BaseLLM


@dataclass
class LLMConfig:
    """Configuration for LLM provider selection."""

    provider: str = "openai"
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    extra: dict = field(default_factory=dict)


_DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-5-20250929",
}


def get_llm(config: LLMConfig | None = None) -> BaseLLM:
    """Create an LLM instance from configuration."""
    if config is None:
        config = LLMConfig(
            provider=os.getenv("AIFOR_LLM_PROVIDER", "openai"),
            model=os.getenv("AIFOR_MODEL"),
        )

    provider = config.provider.lower()
    model = config.model or _DEFAULT_MODELS.get(provider, "gpt-4o")

    if provider == "openai":
        from .openai import OpenAILLM

        return OpenAILLM(model=model, api_key=config.api_key, base_url=config.base_url)
    elif provider == "anthropic":
        from .anthropic import AnthropicLLM

        return AnthropicLLM(model=model, api_key=config.api_key)
    else:
        raise ValueError(f"Unknown LLM provider: {provider!r}. Supported: openai, anthropic")
