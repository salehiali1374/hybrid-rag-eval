"""Reranking candidates with a cross-encoder.

A bi-encoder (the dense retriever) embeds the query and each document
separately, so documents can be embedded once and searched quickly, but the
query and the document never "see" each other. A cross-encoder reads the pair
`(query, document)` together and outputs one relevance score. That is more
accurate and far slower, so it is only run on the few candidates that the
first stage (BM25 + dense + RRF) already found.

As with `Encoder`, the model is injected: any object with a
`predict(pairs, **kwargs) -> numpy array` method works. In production that is
a `sentence_transformers.CrossEncoder`; in the tests it is a tiny fake.
"""

from typing import Protocol

import numpy as np


class CrossEncoderModel(Protocol):
    def predict(self, sentences: list[tuple[str, str]], **kwargs) -> np.ndarray: ...


class Reranker:
    def __init__(self, model: CrossEncoderModel, batch_size: int = 16) -> None:
        self.model = model
        self.batch_size = batch_size

    def rerank(
        self, query: str, candidates: list[tuple[str, str]], top_k: int | None = None
    ) -> list[tuple[str, float]]:
        """Reorder `candidates`, a list of (doc_id, text), by relevance to `query`.

        Returns (doc_id, score) pairs, highest score first; ties are broken by
        doc_id in ascending order. If `top_k` is given, only the best `top_k`
        are returned. An empty candidate list returns [] without calling the model.
        """
        if not candidates:
            return []
        scores = np.asarray(
            self.model.predict(
                [(query, text) for _, text in candidates],
                batch_size=self.batch_size,
                show_progress_bar=False,
            ),
            dtype=np.float64,
        )
        if scores.shape != (len(candidates),):
            raise ValueError(f"expected {len(candidates)} scores, got shape {scores.shape}")

        ranked = sorted(
            ((doc_id, float(score)) for (doc_id, _), score in zip(candidates, scores, strict=True)),
            key=lambda item: (-item[1], item[0]),
        )
        return ranked if top_k is None else ranked[:top_k]
