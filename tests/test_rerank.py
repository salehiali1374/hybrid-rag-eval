"""Specification for hybrid_rag.retrieval.rerank. No real model is loaded."""

import numpy as np
import pytest

from hybrid_rag.retrieval.rerank import Reranker


class FakeCrossEncoder:
    """Scores a pair by looking its document text up in a dict; records every call."""

    def __init__(self, scores_by_text):
        self.scores_by_text = scores_by_text
        self.calls = []

    def predict(self, sentences, **kwargs):
        self.calls.append((list(sentences), kwargs))
        return np.array([self.scores_by_text[text] for _, text in sentences], dtype=np.float32)


def ids(results):
    return [doc_id for doc_id, _ in results]


class TestRerank:
    def test_orders_by_score_and_passes_pairs(self):
        model = FakeCrossEncoder({"low": 0.1, "high": 0.9, "mid": 0.5})
        candidates = [("d1", "low"), ("d2", "high"), ("d3", "mid")]
        results = Reranker(model, batch_size=4).rerank("پرسش", candidates)

        assert ids(results) == ["d2", "d3", "d1"]
        assert [score for _, score in results] == pytest.approx([0.9, 0.5, 0.1])
        ((pairs, kwargs),) = model.calls
        assert pairs == [("پرسش", "low"), ("پرسش", "high"), ("پرسش", "mid")]
        assert kwargs["batch_size"] == 4

    def test_ties_broken_by_doc_id(self):
        model = FakeCrossEncoder({"x": 0.5, "y": 0.5, "z": 0.5})
        results = Reranker(model).rerank("q", [("c", "x"), ("a", "y"), ("b", "z")])
        assert ids(results) == ["a", "b", "c"]

    def test_top_k(self):
        model = FakeCrossEncoder({"x": 1.0, "y": 2.0, "z": 3.0})
        results = Reranker(model).rerank("q", [("a", "x"), ("b", "y"), ("c", "z")], top_k=2)
        assert ids(results) == ["c", "b"]

    def test_empty_candidates_do_not_call_the_model(self):
        model = FakeCrossEncoder({})
        assert Reranker(model).rerank("q", []) == []
        assert model.calls == []

    def test_wrong_number_of_scores(self):
        class Broken:
            def predict(self, sentences, **kwargs):
                return np.array([1.0])

        with pytest.raises(ValueError):
            Reranker(Broken()).rerank("q", [("a", "x"), ("b", "y")])
