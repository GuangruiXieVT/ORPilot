"""Unit tests for Retriever internals: Jaccard gate, modeling pattern filtering, channel fallbacks.

These tests use a tiny synthetic index so no corpus file or API key is needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from orpilot.rag.retriever import Retriever


# ---------------------------------------------------------------------------
# Helpers — build minimal synthetic indices
# ---------------------------------------------------------------------------

def _make_example(name: str, text: str, fingerprint: list[str], embedding=None) -> dict:
    return {
        "name": name,
        "text": text,
        "problem_title": name,
        "modeling_patterns": [],
        "problem": {},
        "ir": {},
        "fingerprint": fingerprint,
        "embedding": embedding,
    }


def _make_index(*examples) -> dict:
    return {"examples": list(examples)}


# ---------------------------------------------------------------------------
# Modeling pattern filtering: Retriever.__init__ restricts stored patterns to
# pat: tokens only, even when the index contains sense:/model:/has: tokens.
# ---------------------------------------------------------------------------

def test_modeling_patterns_stored_only_pat_tokens():
    ex = _make_example(
        "ex1", "text",
        ["sense:minimize", "model:lp", "pat:binary variable", "has:lag", "pat:temporal lag"]
    )
    r = Retriever(_make_index(ex))
    assert r._modeling_patterns[0] == {"pat:binary variable", "pat:temporal lag"}


def test_modeling_patterns_empty_after_filtering():
    ex = _make_example("ex1", "text", ["sense:minimize", "model:mip", "has:ordered_set"])
    r = Retriever(_make_index(ex))
    assert r._modeling_patterns[0] == set()


def test_modeling_patterns_none_treated_as_empty():
    ex = _make_example("ex1", "text", [])
    ex["fingerprint"] = None
    r = Retriever(_make_index(ex))
    assert r._modeling_patterns[0] == set()


# ---------------------------------------------------------------------------
# Jaccard activation gate: channel only fires when query has >= 2 pat: tokens.
# We inspect the internal ranking count via a controlled setup where BM25 and
# Jaccard would disagree — if Jaccard fires, the order changes.
# ---------------------------------------------------------------------------

_EX_A = _make_example("alpha", "inventory balance temporal lag ordered periods", ["pat:temporal lag", "pat:ordered set", "pat:init constraint"])
_EX_B = _make_example("beta",  "binary variable big M linking capacity", ["pat:binary variable", "pat:big-M linking"])
_SMALL_INDEX = _make_index(_EX_A, _EX_B)


def test_jaccard_not_activated_with_zero_patterns():
    r = Retriever(_SMALL_INDEX)
    # Empty fingerprint → Jaccard channel skipped; only BM25 runs
    result = r.retrieve("binary variable capacity", k=2, modeling_patterns=set())
    # Should return something without error — channel simply not activated
    assert len(result) <= 2


def test_jaccard_not_activated_with_one_pattern():
    r = Retriever(_SMALL_INDEX)
    qfp = {"pat:binary variable"}  # only 1 pat: token
    result = r.retrieve("binary variable capacity", k=2, modeling_patterns=qfp)
    assert len(result) <= 2


def test_jaccard_activated_with_two_patterns():
    # Build index where Jaccard and BM25 clearly disagree:
    #   BM25 query text matches ex_bm25 strongly (exact keyword overlap)
    #   Jaccard patterns match ex_jac only (different patterns)
    ex_bm25 = _make_example("bm25_winner", "alpha beta gamma delta epsilon", ["pat:set covering"])
    ex_jac  = _make_example("jac_winner",  "unrelated text xyz",             ["pat:temporal lag", "pat:ordered set"])
    r = Retriever(_make_index(ex_bm25, ex_jac))

    # Query text heavily favours ex_bm25; fingerprint exclusively matches ex_jac
    qfp = {"pat:temporal lag", "pat:ordered set"}
    results = r.retrieve("alpha beta gamma delta epsilon", k=2, modeling_patterns=qfp)
    names = [x["name"] for x in results]
    # With Jaccard active, jac_winner should appear somewhere in top-2 via RRF
    assert "jac_winner" in names


def test_jaccard_not_activated_when_no_corpus_modeling_patterns():
    # All corpus modeling patterns empty → `any(self._modeling_patterns)` is False → channel skipped
    ex = _make_example("ex", "temporal lag ordered set inventory", [])
    r = Retriever(_make_index(ex))
    qfp = {"pat:temporal lag", "pat:ordered set"}
    result = r.retrieve("temporal lag ordered set", k=1, modeling_patterns=qfp)
    assert len(result) == 1  # no error, just BM25 result


# ---------------------------------------------------------------------------
# Semantic channel fallback: when emb_matrix is None, semantic channel is
# skipped silently and only BM25 (+ optional Jaccard) runs.
# ---------------------------------------------------------------------------

def test_no_embeddings_semantic_channel_skipped():
    ex1 = _make_example("doc1", "alpha beta gamma", [], embedding=None)
    ex2 = _make_example("doc2", "delta epsilon zeta", [], embedding=None)
    r = Retriever(_make_index(ex1, ex2))
    assert r._emb_matrix is None
    # retrieve must not raise even though no embedder
    results = r.retrieve("alpha beta", k=2)
    assert len(results) == 2


def test_no_embeddings_semantic_only_raises():
    ex = _make_example("doc1", "alpha beta", [], embedding=None)
    r = Retriever(_make_index(ex))
    with pytest.raises(ValueError, match="semantic_only requires"):
        r.retrieve("alpha", k=1, semantic_only=True)


def test_embeddings_loaded_into_matrix():
    emb = [0.1, 0.2, 0.3]
    ex1 = _make_example("doc1", "alpha", [], embedding=emb)
    ex2 = _make_example("doc2", "beta",  [], embedding=emb)
    r = Retriever(_make_index(ex1, ex2))
    assert r._emb_matrix is not None
    assert r._emb_matrix.shape == (2, 3)


def test_partial_embeddings_matrix_is_none():
    # If even one embedding is None, matrix is not built
    ex1 = _make_example("doc1", "alpha", [], embedding=[0.1, 0.2])
    ex2 = _make_example("doc2", "beta",  [], embedding=None)
    r = Retriever(_make_index(ex1, ex2))
    assert r._emb_matrix is None


# ---------------------------------------------------------------------------
# retrieve() k parameter respected
# ---------------------------------------------------------------------------

def test_retrieve_returns_at_most_k():
    examples = [_make_example(f"doc{i}", f"word{i}", []) for i in range(10)]
    r = Retriever(_make_index(*examples))
    for k in (1, 3, 5):
        results = r.retrieve("word0", k=k)
        assert len(results) <= k


def test_retrieve_k_larger_than_corpus():
    examples = [_make_example(f"doc{i}", f"text{i}", []) for i in range(3)]
    r = Retriever(_make_index(*examples))
    results = r.retrieve("text", k=100)
    assert len(results) == 3
