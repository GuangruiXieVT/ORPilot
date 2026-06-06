"""OpenAI embeddings wrapper with batching."""

from __future__ import annotations

import os

_DEFAULT_MODEL = "text-embedding-3-small"


class Embedder:
    def __init__(self, model: str = _DEFAULT_MODEL, api_key: str | None = None) -> None:
        from openai import OpenAI
        self.model = model
        self._client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def embed(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(input=[text], model=self.model)
        return resp.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts; returns embeddings in the same order."""
        if not texts:
            return []
        resp = self._client.embeddings.create(input=texts, model=self.model)
        ordered = sorted(resp.data, key=lambda d: d.index)
        return [d.embedding for d in ordered]


def get_embedder(model: str = _DEFAULT_MODEL, api_key: str | None = None) -> Embedder | None:
    """Return an Embedder if an API key is available, else None.

    Priority: explicit api_key → ORPILOT_EMBED_API_KEY env var → OPENAI_API_KEY env var.
    When called from the CLI, api_key is already resolved (CLI flag > env var > config file)
    so the env var fallbacks here only apply to standalone/programmatic use.
    """
    resolved = api_key or os.getenv("ORPILOT_EMBED_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not resolved:
        return None
    try:
        return Embedder(model=model, api_key=resolved)
    except Exception:
        return None
