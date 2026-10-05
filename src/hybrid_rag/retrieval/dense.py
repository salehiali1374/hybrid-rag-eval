"""Dense retrieval with sentence embeddings.

Only numpy is used here; torch and sentence-transformers are never imported,
so this module is fast to import and easy to test.

`Encoder` receives the model object from outside ("dependency injection"):
any object with an `encode(texts, **kwargs) -> numpy array` method works. In
production that is a `SentenceTransformer`; in the tests it is a tiny fake, so
the tests run in milliseconds and never download a model.
"""

from typing import Protocol

import numpy as np


class EmbeddingModel(Protocol):
    def encode(self, sentences: list[str], **kwargs) -> np.ndarray: ...


class Encoder:
    """Turns texts into embeddings for E5-style models.

    E5 models were trained with a prefix that tells the model what the text is:
    queries must start with "query: " and documents with "passage: ". Without
    the prefixes retrieval quality drops noticeably.
    """

    def __init__(self, model: EmbeddingModel, batch_size: int = 32) -> None:
        self.model = model
        self.batch_size = batch_size

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        """Embed queries: add the "query: " prefix, call `model.encode` once with
        `batch_size=self.batch_size` and `normalize_embeddings=True`, and return a
        float32 array of shape (len(texts), dim)."""
        prefixed = [f"query: {text}" for text in texts]

        embeddings = self.model.encode(
            prefixed,
            batch_size=self.batch_size,
            normalize_embeddings=True,
        )

        return np.asarray(embeddings, dtype=np.float32)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        """Same as `encode_queries`, with the "passage: " prefix."""
        prefixed = [f"passage: {text}" for text in texts]

        embeddings = self.model.encode(
            prefixed,
            batch_size=self.batch_size,
            normalize_embeddings=True,
        )

        return np.asarray(embeddings, dtype=np.float32)


class DenseIndex:
    """Exact nearest-neighbour search by cosine similarity.

    Cosine similarity of two vectors is the dot product of their unit-length
    versions. Normalize the document matrix once in `__init__`, and each query
    in `search`, so that every search is a single matrix-vector product.
    Do not assume the caller passes normalized vectors.
    """

    def __init__(self, doc_ids: list[str], embeddings: np.ndarray) -> None:
        """Raise ValueError if `embeddings` is not 2-D or its number of rows
        differs from `len(doc_ids)`."""
        embeddings = np.asarray(embeddings, dtype=np.float32)

        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2-D array")

        if embeddings.shape[0] != len(doc_ids):
            raise ValueError("number of embeddings must match number of doc_ids")

        self.doc_ids = doc_ids

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Only the normalized matrix is kept; the raw one is not needed for search.
        # Avoid division by zero for zero vectors.
        self._docs = np.divide(
            embeddings,
            norms,
            out=np.zeros_like(embeddings),
            where=norms != 0,
        )

    def search(self, query: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        """The `top_k` (doc_id, cosine similarity) pairs, highest first.

        Ties are broken by doc_id in ascending order. If `top_k` is larger than
        the index, return every document.
        """
        query = np.asarray(query, dtype=np.float32)

        if query.ndim != 1:
            raise ValueError("query must be a 1-D array")

        norm = np.linalg.norm(query)

        if norm == 0:
            similarities = np.zeros(len(self.doc_ids), dtype=np.float32)
        else:
            normalized_query = query / norm
            similarities = self._docs @ normalized_query

        results = [
            (doc_id, float(score)) for doc_id, score in zip(self.doc_ids, similarities, strict=True)
        ]

        # Highest similarity first, doc_id ascending for ties.
        results.sort(key=lambda x: (-x[1], x[0]))

        return results[:top_k]

    def search_batch(self, queries: np.ndarray, top_k: int = 10) -> list[list[tuple[str, float]]]:
        """`search` for every row of `queries` (shape (n_queries, dim)).

        Compute all similarities with ONE matrix product (queries @ docs.T)
        instead of calling `search` in a Python loop.
        """
        queries = np.asarray(queries, dtype=np.float32)

        if queries.ndim != 2:
            raise ValueError("queries must be a 2-D array")

        norms = np.linalg.norm(queries, axis=1, keepdims=True)

        normalized_queries = np.divide(
            queries,
            norms,
            out=np.zeros_like(queries),
            where=norms != 0,
        )

        # ONE matrix product.
        similarities = normalized_queries @ self._docs.T

        results = []

        for scores in similarities:
            row = [
                (doc_id, float(score)) for doc_id, score in zip(self.doc_ids, scores, strict=True)
            ]

            row.sort(key=lambda x: (-x[1], x[0]))
            results.append(row[:top_k])

        return results
