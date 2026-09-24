"""Specification for hybrid_rag.eval.metrics."""

import math

import pytest

from hybrid_rag.eval.metrics import evaluate_run, ndcg_at_k, recall_at_k, reciprocal_rank


class TestRecall:
    def test_counts_only_top_k(self):
        assert recall_at_k(["a", "b", "c", "d"], {"b", "d"}, k=2) == 0.5

    def test_all_found(self):
        assert recall_at_k(["a", "b", "c"], {"a", "c"}, k=3) == 1.0

    def test_k_larger_than_list(self):
        assert recall_at_k(["a"], {"a", "z"}, k=10) == 0.5

    def test_empty_relevant_is_an_error(self):
        with pytest.raises(ValueError):
            recall_at_k(["a"], set(), k=1)


class TestReciprocalRank:
    def test_first_relevant_rank(self):
        assert reciprocal_rank(["a", "b", "c"], {"c", "b"}) == 0.5

    def test_no_relevant(self):
        assert reciprocal_rank(["a", "b"], {"z"}) == 0.0

    def test_cutoff(self):
        assert reciprocal_rank(["a", "b", "c"], {"c"}, k=2) == 0.0
        assert reciprocal_rank(["a", "b", "c"], {"c"}, k=3) == pytest.approx(1 / 3)


class TestNdcg:
    def test_perfect_ranking(self):
        assert ndcg_at_k(["a", "b"], {"a": 2, "b": 1}, k=2) == pytest.approx(1.0)

    def test_graded_example(self):
        # ranking a(0) b(2) c(1); ideal grades are 3, 2, 1 (d is relevant but not retrieved)
        dcg = 0 / math.log2(2) + 2 / math.log2(3) + 1 / math.log2(4)
        idcg = 3 / math.log2(2) + 2 / math.log2(3) + 1 / math.log2(4)
        got = ndcg_at_k(["a", "b", "c"], {"b": 2, "c": 1, "d": 3}, k=3)
        assert got == pytest.approx(dcg / idcg)

    def test_no_relevant_gives_zero(self):
        assert ndcg_at_k(["a"], {}, k=1) == 0.0


class TestEvaluateRun:
    def test_averages_and_penalizes_missing_queries(self):
        qrels = {"q1": {"a": 1}, "q2": {"b": 1}}
        # q2 missing, q_extra ignored
        run = {"q1": ["a", "x"], "q_extra": ["b"]}
        scores = evaluate_run(run, qrels, k_values=(1,))
        assert scores["recall@1"] == pytest.approx(0.5)
        assert scores["mrr@1"] == pytest.approx(0.5)
        assert scores["ndcg@1"] == pytest.approx(0.5)

    def test_keys_for_each_k(self):
        scores = evaluate_run({"q": ["a"]}, {"q": {"a": 1}}, k_values=(1, 10))
        assert set(scores) == {f"{m}@{k}" for m in ("recall", "mrr", "ndcg") for k in (1, 10)}

    def test_zero_grade_is_not_relevant(self):
        scores = evaluate_run({"q1": ["a", "b"]}, {"q1": {"a": 0, "b": 1}}, k_values=(1,))
        assert scores["recall@1"] == 0.0
        assert scores["mrr@1"] == 0.0

    def test_query_without_relevant_docs_is_skipped(self):
        qrels = {"q1": {"b": 1}, "q2": {"x": 0}}
        scores = evaluate_run({"q1": ["b"], "q2": ["x"]}, qrels, k_values=(1,))
        assert scores["recall@1"] == 1.0

    def test_no_evaluable_queries_returns_zero(self):
        scores = evaluate_run({"q": ["x"]}, {"q": {"x": 0}}, k_values=(1,))
        assert scores == {"recall@1": 0.0, "mrr@1": 0.0, "ndcg@1": 0.0}
