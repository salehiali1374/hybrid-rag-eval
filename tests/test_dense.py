"""Specification for hybrid_rag.retrieval.dense. No real model is loaded."""

import numpy as np
import pytest

from hybrid_rag.retrieval.dense import DenseIndex, Encoder


class FakeModel:
    """Records every call and returns one fixed 3-d vector per input text."""

    def __init__(self):
        self.calls = []

    def encode(self, sentences, **kwargs):
        self.calls.append((list(sentences), kwargs))
        return np.array([[1.0, 2.0, 2.0]] * len(sentences), dtype=np.float64)


class TestEncoder:
    def test_query_prefix_and_kwargs(self):
        model = FakeModel()
        out = Encoder(model, batch_size=8).encode_queries(["سلام", "دنیا"])
        ((texts, kwargs),) = model.calls
        assert texts == ["query: سلام", "query: دنیا"]
        assert kwargs["batch_size"] == 8
        assert kwargs["normalize_embeddings"] is True
        assert out.shape == (2, 3)
        assert out.dtype == np.float32

    def test_passage_prefix(self):
        model = FakeModel()
        Encoder(model).encode_passages(["متن"])
        assert model.calls[0][0] == ["passage: متن"]


class TestDenseIndex:
    def test_ranks_by_cosine_not_dot_product(self):
        # "big" has the larger dot product with the query, "aligned" the larger cosine
        index = DenseIndex(["big", "aligned"], np.array([[10.0, 10.0], [1.0, 0.0]]))
        results = index.search(np.array([1.0, 0.0]))
        assert [doc_id for doc_id, _ in results] == ["aligned", "big"]
        assert results[0][1] == pytest.approx(1.0)
        assert results[1][1] == pytest.approx(1 / np.sqrt(2))

    def test_query_does_not_need_to_be_normalized(self):
        index = DenseIndex(["a", "b"], np.array([[1.0, 0.0], [0.0, 1.0]]))
        assert index.search(np.array([0.0, 50.0]))[0] == ("b", pytest.approx(1.0))

    def test_top_k_and_small_index(self):
        index = DenseIndex(["a", "b", "c"], np.eye(3))
        assert len(index.search(np.array([1.0, 0.0, 0.0]), top_k=2)) == 2
        assert len(index.search(np.array([1.0, 0.0, 0.0]), top_k=50)) == 3

    def test_ties_broken_by_doc_id(self):
        index = DenseIndex(["z", "a", "m"], np.ones((3, 2)))
        assert [doc_id for doc_id, _ in index.search(np.array([1.0, 1.0]))] == ["a", "m", "z"]

    def test_batch_matches_single(self):
        rng = np.random.default_rng(0)
        docs, queries = rng.normal(size=(20, 8)), rng.normal(size=(5, 8))
        index = DenseIndex([f"d{i}" for i in range(20)], docs)
        batch = index.search_batch(queries, top_k=4)
        for row, query in zip(batch, queries, strict=True):
            single = index.search(query, top_k=4)
            assert [d for d, _ in row] == [d for d, _ in single]
            assert [s for _, s in row] == pytest.approx([s for _, s in single])

    def test_rejects_bad_shapes(self):
        with pytest.raises(ValueError):
            DenseIndex(["a", "b"], np.ones((3, 4)))
        with pytest.raises(ValueError):
            DenseIndex(["a"], np.ones(4))
