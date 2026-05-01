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
    # Hard cap on generated tokens — keeps responses fast and bounded.
    max_tokens: int = 8192
    # Per-request timeout in seconds.
    timeout: float = 120.0
    # How many times to retry after a timeout before giving up.
    max_retries: int = 2
    # Sampling temperature: 0.0 = fully deterministic, 1.0 = default creative.
    temperature: float = 0.0
    extra: dict = field(default_factory=dict)


_DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-5-20250929",
    "google": "gemini-2.0-flash",
}


def get_llm(config: LLMConfig | None = None) -> BaseLLM:
    """Create an LLM instance from configuration."""
    if config is None:
        config = LLMConfig(
            provider=os.getenv("ORPILOT_LLM_PROVIDER", "openai"),
            model=os.getenv("ORPILOT_MODEL"),
        )

    provider = config.provider.lower()
    model = config.model or _DEFAULT_MODELS.get(provider, "gpt-4o")

    if provider == "openai":
        from .openai import OpenAILLM

        return OpenAILLM(
            model=model,
            api_key=config.api_key,
            base_url=config.base_url,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
            max_retries=config.max_retries,
            temperature=config.temperature,
        )
    elif provider == "anthropic":
        from .anthropic import AnthropicLLM

        return AnthropicLLM(
            model=model,
            api_key=config.api_key,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
            max_retries=config.max_retries,
            temperature=config.temperature,
        )
    elif provider == "google":
        from .gemini import GeminiLLM

        return GeminiLLM(
            model=model,
            api_key=config.api_key,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
            max_retries=config.max_retries,
            temperature=config.temperature,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider!r}. Supported: openai, anthropic, google")
