"""Specification for hybrid_rag.retrieval.fusion. Expected values are computed by hand."""

import pytest

from hybrid_rag.retrieval.fusion import reciprocal_rank_fusion as rrf


def ids(results):
    return [doc_id for doc_id, _ in results]


class TestScores:
    def test_two_rankings(self):
        results = rrf([["a", "b", "c"], ["b", "c", "d"]], k=60)
        assert ids(results) == ["b", "c", "a", "d"]
        scores = dict(results)
        assert scores["a"] == pytest.approx(1 / 61)
        assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
        assert scores["c"] == pytest.approx(1 / 63 + 1 / 62)
        assert scores["d"] == pytest.approx(1 / 63)

    def test_single_ranking_keeps_its_order(self):
        assert ids(rrf([["c", "a", "b"]])) == ["c", "a", "b"]

    def test_empty_input(self):
        assert rrf([]) == []
        assert rrf([[], []]) == []


class TestOrdering:
    def test_ties_broken_by_doc_id(self):
        assert ids(rrf([["y", "x"], ["x", "y"]])) == ["x", "y"]

    def test_top_k(self):
        results = rrf([["a", "b", "c"], ["b", "c", "d"]], top_k=2)
        assert ids(results) == ["b", "c"]


class TestEffectOfK:
    # "a" is ranked 1st by one ranking only; "b" is ranked 3rd by both
    RANKINGS = [["a", "x", "b"], ["y", "z", "b"]]

    def test_small_k_rewards_a_top_position(self):
        assert ids(rrf(self.RANKINGS, k=0))[0] == "a"

    def test_large_k_rewards_agreement(self):
        assert ids(rrf(self.RANKINGS, k=60))[0] == "b"


class TestWeights:
    def test_weight_scales_a_ranking(self):
        results = rrf([["a"], ["b"]], k=60, weights=[1.0, 2.0])
        assert ids(results) == ["b", "a"]
        assert dict(results)["b"] == pytest.approx(2 / 61)

    def test_equal_weights_match_unweighted(self):
        rankings = [["a", "b", "c"], ["b", "c", "d"]]
        weighted = rrf(rankings, weights=[1.0, 1.0])
        assert ids(weighted) == ids(rrf(rankings))

    def test_wrong_number_of_weights(self):
        with pytest.raises(ValueError):
            rrf([["a"], ["b"]], weights=[1.0])
