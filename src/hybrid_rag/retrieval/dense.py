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
        raise NotImplementedError

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        """Embed queries: add the "query: " prefix, call `model.encode` once with
        `batch_size=self.batch_size` and `normalize_embeddings=True`, and return a
        float32 array of shape (len(texts), dim)."""
        raise NotImplementedError

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        """Same as `encode_queries`, with the "passage: " prefix."""
        raise NotImplementedError


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
        raise NotImplementedError

    def search(self, query: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        """The `top_k` (doc_id, cosine similarity) pairs, highest first.

        Ties are broken by doc_id in ascending order. If `top_k` is larger than
        the index, return every document.
        """
        raise NotImplementedError

    def search_batch(self, queries: np.ndarray, top_k: int = 10) -> list[list[tuple[str, float]]]:
        """`search` for every row of `queries` (shape (n_queries, dim)).

        Compute all similarities with ONE matrix product (queries @ docs.T)
        instead of calling `search` in a Python loop.
        """
        raise NotImplementedError
