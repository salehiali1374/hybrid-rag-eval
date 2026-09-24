"""Okapi BM25, implemented from scratch.

For a query q = (t_1, ..., t_n) and a document d:

    score(q, d) = sum over query terms t that occur in d of
                  idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))

    idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))      (Lucene variant, always > 0)

where tf(t, d) is how often t occurs in d, |d| is the number of tokens in d,
avgdl is the average document length, N the number of documents and df(t) the
number of documents containing t. A term that appears twice in the query adds
its contribution twice.

Design hint: build an inverted index in `__init__` (term -> {doc_id: tf}), so a
query only touches the documents that contain its terms instead of scanning
the whole corpus.
"""


class BM25:
    def __init__(self, docs: dict[str, list[str]], k1: float = 1.2, b: float = 0.75) -> None:
        """Index `docs`, a mapping of doc id -> list of tokens (already tokenized)."""
        raise NotImplementedError

    def idf(self, term: str) -> float:
        """idf of `term` as defined above; 0.0 for a term that is in no document."""
        raise NotImplementedError

    def scores(self, query: list[str]) -> dict[str, float]:
        """Score of every document that contains at least one query term.

        Documents without any query term are left out (their score would be 0).
        """
        raise NotImplementedError

    def search(self, query: list[str], top_k: int = 10) -> list[tuple[str, float]]:
        """The `top_k` best (doc_id, score) pairs, highest score first.

        Ties are broken by doc_id in ascending order, so results are deterministic.
        """
        raise NotImplementedError
