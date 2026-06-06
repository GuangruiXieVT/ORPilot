"""RAG (Retrieval-Augmented Generation) support for ORPilot IR building."""

from __future__ import annotations

from .indexer import INDEX_PATH, CORPUS_DIR, build_index, load_index
from .retriever import Retriever

_retriever_cache: Retriever | None = None


def get_retriever(embedder=None) -> Retriever | None:  # noqa: ANN001
    """Return a cached Retriever, building from disk index if available.

    Returns None if no index exists (run `orpilot rag build` first).
    """
    global _retriever_cache
    if _retriever_cache is not None:
        return _retriever_cache

    index = load_index()
    if index is None:
        return None

    _retriever_cache = Retriever(index, embedder=embedder)
    return _retriever_cache


def reset_retriever_cache() -> None:
    """Clear the in-process retriever cache (useful after index rebuild)."""
    global _retriever_cache
    _retriever_cache = None


__all__ = [
    "get_retriever",
    "reset_retriever_cache",
    "build_index",
    "load_index",
    "Retriever",
    "INDEX_PATH",
    "CORPUS_DIR",
]
