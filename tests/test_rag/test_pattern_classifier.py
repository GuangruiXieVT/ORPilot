"""Unit tests for orpilot.rag.pattern_classifier.

All tests use a mock BaseLLM so no API key is needed.
"""

from __future__ import annotations

import json
import pytest

from orpilot.llm.base import BaseLLM, ChatResponse, ToolDefinition
from orpilot.rag.pattern_classifier import classify_patterns
from orpilot.rag.patterns import VOCAB_SET


# ---------------------------------------------------------------------------
# Minimal BaseLLM stub — only chat() is exercised by classify_patterns
# ---------------------------------------------------------------------------

class _MockLLM(BaseLLM):
    def __init__(self, response: str):
        super().__init__()
        self._response = response

    def chat(self, messages):
        return self._response

    def structured_output(self, messages, schema):
        raise NotImplementedError

    def chat_with_tools(self, messages, tools):
        raise NotImplementedError

    def extend_messages(self, messages, response, results):
        raise NotImplementedError


def _llm(patterns: list[str]) -> _MockLLM:
    return _MockLLM(json.dumps({"patterns": patterns}))


_PROBLEM = {
    "title": "Make-or-Buy Sourcing",
    "description": "A manufacturer decides for each component whether to produce in-house or buy from a vendor. "
                   "Production is only allowed when the make decision is 1. Machine capacity is limited.",
    "objective": "minimize",
    "objective_description": "Total production and procurement cost",
    "decision_variables": ["make[i]: 1 if component i is produced in-house", "produce[i]: units made", "buy[i]: units purchased"],
    "parameters": ["machine_hours[i]", "machine_capacity", "demand[i]"],
    "constraints": ["Demand met via make or buy", "Machine capacity limit",
                    "Production only allowed when make is 1"],
}


# ---------------------------------------------------------------------------
# Core behaviour
# ---------------------------------------------------------------------------

def test_valid_patterns_returned_as_pat_tokens():
    result = classify_patterns(_PROBLEM, _llm(["binary variable", "big-M linking"]))
    assert result == {"pat:binary variable", "pat:big-M linking"}


def test_non_vocab_patterns_silently_dropped():
    result = classify_patterns(_PROBLEM, _llm(["binary variable", "invented pattern", "not-a-real-one"]))
    assert result == {"pat:binary variable"}
    # invalid names never appear
    assert not any("invented" in t or "real" in t for t in result)


def test_empty_pattern_list_returns_empty_set():
    result = classify_patterns(_PROBLEM, _llm([]))
    assert result == set()


def test_all_valid_vocab_patterns_accepted():
    # Every VOCAB entry should round-trip through the classifier
    all_vocab = list(VOCAB_SET)
    result = classify_patterns(_PROBLEM, _llm(all_vocab))
    assert result == {f"pat:{p}" for p in VOCAB_SET}


# ---------------------------------------------------------------------------
# Error handling: falls back to regex detect_modeling_patterns()
# ---------------------------------------------------------------------------

class _BrokenLLM(_MockLLM):
    def chat(self, messages):
        raise RuntimeError("API unavailable")


def test_llm_error_falls_back_to_regex():
    result = classify_patterns(_PROBLEM, _BrokenLLM(""))
    # Falls back to regex — result is a set of pat: tokens (possibly empty for this problem)
    assert isinstance(result, set)
    assert all(t.startswith("pat:") for t in result)


class _MalformedLLM(_MockLLM):
    def chat(self, messages):
        return "not json at all"


def test_malformed_json_falls_back_to_regex():
    result = classify_patterns(_PROBLEM, _MalformedLLM(""))
    assert isinstance(result, set)
    assert all(t.startswith("pat:") for t in result)


class _WrongSchemaLLM(_MockLLM):
    def chat(self, messages):
        return json.dumps({"result": ["binary variable"]})  # wrong key


def test_wrong_json_schema_returns_empty_set():
    # data.get("patterns", []) returns [] when key is absent — no fallback, empty set
    result = classify_patterns(_PROBLEM, _WrongSchemaLLM(""))
    assert result == set()


# ---------------------------------------------------------------------------
# csv_schemas integration: content reaches the LLM prompt
# ---------------------------------------------------------------------------

class _CapturingLLM(_MockLLM):
    """Records the last user message content for inspection."""
    def __init__(self, patterns):
        super().__init__(json.dumps({"patterns": patterns}))
        self.last_user_content = ""

    def chat(self, messages):
        user_msgs = [m for m in messages if m.get("role") == "user"]
        if user_msgs:
            self.last_user_content = user_msgs[-1]["content"]
        return self._response


def test_csv_schemas_included_in_prompt():
    csv_schemas = {
        "arcs": {
            "columns": {"from": "str", "to": "str", "cost": "float"},
            "row_count": 8,
            "distinct_values": {},
        }
    }
    llm = _CapturingLLM([])
    classify_patterns(_PROBLEM, llm, csv_schemas=csv_schemas)
    assert "arcs" in llm.last_user_content
    assert "8" in llm.last_user_content


def test_no_csv_schemas_shows_no_data_files():
    llm = _CapturingLLM([])
    classify_patterns(_PROBLEM, llm, csv_schemas=None)
    assert "no CSV data files" in llm.last_user_content


def test_empty_csv_schemas_shows_no_data_files():
    llm = _CapturingLLM([])
    classify_patterns(_PROBLEM, llm, csv_schemas={})
    assert "no CSV data files" in llm.last_user_content


# ---------------------------------------------------------------------------
# Sparse filter: LLM sees row_count and can reason about sparsity
# ---------------------------------------------------------------------------

def test_sparse_arc_table_llm_can_return_sparse_filter():
    # The LLM sees a 6-row arc table for a 4×4 potential grid (16 pairs) and
    # returns sparse_filter. The classifier accepts it as a valid VOCAB pattern.
    sparse_schemas = {
        "routes": {
            "columns": {"origin": "str", "destination": "str", "cost": "float"},
            "row_count": 6,
            "distinct_values": {
                "origin": ["A", "B", "C", "D"],
                "destination": ["A", "B", "C", "D"],
            },
        }
    }
    result = classify_patterns(_PROBLEM, _llm(["sparse filter"]), csv_schemas=sparse_schemas)
    assert "pat:sparse filter" in result
