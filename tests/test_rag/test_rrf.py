"""Unit tests for _rrf_fusion in orpilot.rag.retriever."""

from __future__ import annotations

import pytest

from orpilot.rag.retriever import _rrf_fusion


def _scores(fused: list[tuple[int, float]]) -> dict[int, float]:
    return dict(fused)


def _order(fused: list[tuple[int, float]]) -> list[int]:
    return [idx for idx, _ in fused]


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------

def test_single_list_preserves_order():
    ranking = [3, 1, 0, 2]
    result = _order(_rrf_fusion([ranking]))
    assert result == [3, 1, 0, 2]


def test_two_agreeing_lists_same_order():
    a = [0, 1, 2, 3]
    b = [0, 1, 2, 3]
    result = _order(_rrf_fusion([a, b]))
    assert result == [0, 1, 2, 3]


def test_consensus_beats_single_channel():
    # idx=5 is rank-0 in both lists; idx=0 is rank-0 in only one
    a = [5, 0, 1, 2]
    b = [5, 3, 4, 6]
    result = _order(_rrf_fusion([a, b]))
    assert result[0] == 5


def test_high_rank_in_two_lists_beats_rank0_in_one():
    # idx=99: rank-0 in list A only → score = 1/61 ≈ 0.0164
    # idx=7:  rank-1 in both A and B → score = 2 * 1/62 ≈ 0.0323
    a = [99, 7, 0, 1]
    b = [0,  7, 1, 2]
    result = _order(_rrf_fusion([a, b]))
    # idx=7 should beat idx=99 (appears once at rank-0)
    assert result.index(7) < result.index(99)


def test_score_formula():
    # Single list, one item at rank 0: score should be 1/(60+0+1) = 1/61
    fused = _rrf_fusion([[42]])
    s = _scores(fused)
    assert abs(s[42] - 1.0 / 61) < 1e-9


def test_two_lists_score_accumulates():
    # idx=0 at rank-0 in two lists → score = 2 * 1/61
    fused = _rrf_fusion([[0], [0]])
    s = _scores(fused)
    assert abs(s[0] - 2.0 / 61) < 1e-9


def test_custom_k_changes_scores():
    fused_60 = _rrf_fusion([[0]], k=60)
    fused_10 = _rrf_fusion([[0]], k=10)
    s60 = _scores(fused_60)[0]
    s10 = _scores(fused_10)[0]
    assert abs(s60 - 1.0 / 61) < 1e-9
    assert abs(s10 - 1.0 / 11) < 1e-9
    assert s10 > s60  # smaller k → higher score for rank-0


def test_empty_input():
    result = _rrf_fusion([])
    assert result == []


def test_three_lists_all_agree():
    a = b = c = [0, 1, 2]
    result = _order(_rrf_fusion([a, b, c]))
    assert result == [0, 1, 2]


def test_result_sorted_descending():
    a = [2, 0, 1]
    b = [0, 1, 2]
    fused = _rrf_fusion([a, b])
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)


def test_all_candidates_present():
    # Every index that appears in any input list must appear in the output
    a = [0, 1, 2]
    b = [3, 4, 5]
    result = _order(_rrf_fusion([a, b]))
    assert set(result) == {0, 1, 2, 3, 4, 5}


def test_later_rank_lower_score():
    fused = _rrf_fusion([[10, 20, 30]])
    s = _scores(fused)
    # rank-0 (idx=10) must score higher than rank-1 (idx=20)
    assert s[10] > s[20] > s[30]
