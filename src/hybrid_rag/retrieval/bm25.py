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

import math
from collections import Counter, defaultdict


class BM25:
    def __init__(self, docs: dict[str, list[str]], k1: float = 1.2, b: float = 0.75) -> None:
        """Index `docs`, a mapping of doc id -> list of tokens (already tokenized)."""
        self.k1 = k1
        self.b = b

        self.N = len(docs)

        # doc_id -> document length
        self.doc_lengths = {doc_id: len(tokens) for doc_id, tokens in docs.items()}

        # average document length
        self.avgdl = sum(self.doc_lengths.values()) / self.N if self.N else 0.0

        # term -> {doc_id: term frequency}
        index: dict[str, dict[str, int]] = defaultdict(dict)

        for doc_id, tokens in docs.items():
            term_counts = Counter(tokens)

            for term, tf in term_counts.items():
                index[term][doc_id] = tf

        # plain dict, so a lookup of an unknown term can never insert an empty entry
        self.index = dict(index)

    def idf(self, term: str) -> float:
        """idf of `term` as defined above; 0.0 for a term that is in no document."""
        postings = self.index.get(term)

        if not postings:
            return 0.0

        df = len(postings)

        return math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))

    def scores(self, query: list[str]) -> dict[str, float]:
        """Score of every document that contains at least one query term.

        Documents without any query term are left out (their score would be 0).
        """
        scores = defaultdict(float)

        for term in query:
            postings = self.index.get(term)

            if not postings:
                continue

            idf = self.idf(term)

            for doc_id, tf in postings.items():
                doc_len = self.doc_lengths[doc_id]

                denominator = tf + self.k1 * (1.0 - self.b + self.b * doc_len / self.avgdl)

                contribution = idf * tf * (self.k1 + 1.0) / denominator

                scores[doc_id] += contribution

        return dict(scores)

    def search(self, query: list[str], top_k: int = 10) -> list[tuple[str, float]]:
        """The `top_k` best (doc_id, score) pairs, highest score first.

        Ties are broken by doc_id in ascending order, so results are deterministic.
        """
        scores = self.scores(query)

        return sorted(
            scores.items(),
            key=lambda item: (-item[1], item[0]),
        )[:top_k]
