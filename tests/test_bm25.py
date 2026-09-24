"""Specification for hybrid_rag.retrieval.bm25. Expected values are computed by hand."""

import math

import pytest

from hybrid_rag.retrieval.bm25 import BM25

DOCS = {
    "d1": ["a", "b"],
    "d2": ["a", "c", "c"],
    "d3": ["d"],
}
N, AVGDL = 3, 2.0


def term_score(idf, tf, dl, k1=1.2, b=0.75):
    return idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / AVGDL))


def idf(df):
    return math.log(1 + (N - df + 0.5) / (df + 0.5))


class TestIdf:
    def test_values(self):
        bm25 = BM25(DOCS)
        assert bm25.idf("c") == pytest.approx(idf(1))
        assert bm25.idf("a") == pytest.approx(idf(2))

    def test_rarer_terms_weigh_more(self):
        bm25 = BM25(DOCS)
        assert bm25.idf("c") > bm25.idf("a") > 0

    def test_unknown_term(self):
        assert BM25(DOCS).idf("zzz") == 0.0


class TestScores:
    def test_single_term(self):
        scores = BM25(DOCS).scores(["c"])
        assert scores == {"d2": pytest.approx(term_score(idf(1), tf=2, dl=3))}

    def test_sums_over_terms(self):
        scores = BM25(DOCS).scores(["a", "c"])
        assert scores["d1"] == pytest.approx(term_score(idf(2), tf=1, dl=2))
        assert scores["d2"] == pytest.approx(
            term_score(idf(2), tf=1, dl=3) + term_score(idf(1), tf=2, dl=3)
        )
        assert "d3" not in scores

    def test_repeated_query_term_counts_twice(self):
        bm25 = BM25(DOCS)
        assert bm25.scores(["c", "c"])["d2"] == pytest.approx(2 * bm25.scores(["c"])["d2"])

    def test_b_zero_disables_length_normalization(self):
        docs = {"short": ["x"], "long": ["x", "y", "y", "y"]}
        scores = BM25(docs, b=0.0).scores(["x"])
        assert scores["short"] == pytest.approx(scores["long"])

    def test_length_normalization_favours_short_docs(self):
        docs = {"short": ["x"], "long": ["x", "y", "y", "y"]}
        scores = BM25(docs).scores(["x"])
        assert scores["short"] > scores["long"]

    def test_unknown_and_empty_queries(self):
        bm25 = BM25(DOCS)
        assert bm25.scores(["zzz"]) == {}
        assert bm25.scores([]) == {}


class TestSearch:
    def test_ranking_and_top_k(self):
        results = BM25(DOCS).search(["a", "c"], top_k=1)
        assert [doc_id for doc_id, _ in results] == ["d2"]

    def test_ties_broken_by_doc_id(self):
        docs = {"z": ["x"], "a": ["x"], "m": ["x"]}
        assert [doc_id for doc_id, _ in BM25(docs).search(["x"])] == ["a", "m", "z"]

    def test_no_match(self):
        assert BM25(DOCS).search(["zzz"]) == []
