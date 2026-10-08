"""A tiny three-document pipeline with fake models, shared by the pipeline and API tests."""

import numpy as np
import pytest

from hybrid_rag.pipeline import Pipeline
from hybrid_rag.retrieval.bm25 import BM25
from hybrid_rag.retrieval.dense import DenseIndex, Encoder
from hybrid_rag.retrieval.rerank import Reranker
from hybrid_rag.schema import Document
from hybrid_rag.text.normalize import tokenize

DOCS = [
    Document("d-madrid", "مادرید پایتخت اسپانیا است.", title="اسپانیا"),
    Document("d-paris", "پاریس پایتخت فرانسه است.", title="فرانسه"),
    Document("d-berlin", "برلین پایتخت آلمان است.", title="آلمان"),
]
# the fake query encoder always returns [1, 0], so dense order is madrid, berlin, paris
DOC_VECTORS = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])


class FakeEncoderModel:
    def encode(self, sentences, **kwargs):
        return np.array([[1.0, 0.0]] * len(sentences))


class FakeCrossEncoder:
    """Prefers the passage about Paris, whatever the question is."""

    def predict(self, sentences, **kwargs):
        return np.array([1.0 if "پاریس" in text else 0.0 for _, text in sentences])


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def complete(self, messages):
        self.calls.append(messages)
        return self.reply


@pytest.fixture
def make_pipeline():
    def make(reply="مادرید [1].", *, llm=True, reranker=False, **kwargs):
        fake_llm = FakeLLM(reply)
        pipeline = Pipeline(
            {doc.id: doc for doc in DOCS},
            BM25({doc.id: tokenize(doc.text) for doc in DOCS}),
            DenseIndex([doc.id for doc in DOCS], DOC_VECTORS),
            Encoder(FakeEncoderModel()),
            llm=fake_llm if llm else None,
            reranker=Reranker(FakeCrossEncoder()) if reranker else None,
            context_k=2,
            rerank_depth=3,
            **kwargs,
        )
        pipeline.fake_llm = fake_llm  # for assertions on what the LLM was shown
        return pipeline

    return make
