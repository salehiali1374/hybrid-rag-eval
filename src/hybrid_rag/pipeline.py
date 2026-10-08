"""The retrieval-augmented answering pipeline behind the API.

    question -> BM25 + dense -> RRF -> (optional cross-encoder rerank) -> LLM -> answer + citations

Everything is injected, so tests use a tiny corpus and fake models. Every stage is timed,
which is the "tracing" returned with each response.
"""

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from hybrid_rag.generation.answer import Answer, LLMClient, generate_answer
from hybrid_rag.retrieval.bm25 import BM25
from hybrid_rag.retrieval.dense import DenseIndex, Encoder
from hybrid_rag.retrieval.fusion import reciprocal_rank_fusion
from hybrid_rag.retrieval.rerank import Reranker
from hybrid_rag.schema import Document
from hybrid_rag.text.normalize import tokenize


class RerankerUnavailable(RuntimeError):
    """A rerank was requested but no cross-encoder was loaded."""


class LLMUnavailable(RuntimeError):
    """An answer was requested but no LLM client is configured."""


@dataclass
class Trace:
    """Wall-clock milliseconds per stage, in the order the stages ran."""

    stages_ms: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stages_ms[name] = round(1000 * (time.perf_counter() - start), 1)

    @property
    def total_ms(self) -> float:
        return round(sum(self.stages_ms.values()), 1)


class Pipeline:
    def __init__(
        self,
        corpus: dict[str, Document],
        bm25: BM25,
        dense_index: DenseIndex,
        encoder: Encoder,
        *,
        llm: LLMClient | None = None,
        reranker: Reranker | None = None,
        rrf_k: int = 1,
        dense_weight: float = 1.5,
        retrieval_depth: int = 100,
        context_k: int = 5,
        rerank_depth: int = 10,
    ) -> None:
        self.corpus = corpus
        self.bm25 = bm25
        self.dense_index = dense_index
        self.encoder = encoder
        self.llm = llm
        self.reranker = reranker
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.retrieval_depth = retrieval_depth
        self.context_k = context_k
        self.rerank_depth = rerank_depth
        self._encode_lock = threading.Lock()  # one torch forward pass at a time

    def search(
        self, question: str, rerank: bool = False, trace: Trace | None = None
    ) -> list[Document]:
        """The `context_k` best documents for `question`, best first.

        With `rerank`, the cross-encoder reorders the best `rerank_depth` fused candidates.
        Raise RerankerUnavailable if `rerank` is set but no cross-encoder was loaded.
        """
        if rerank and self.reranker is None:
            raise RerankerUnavailable("the reranker is not loaded (set HRE_LOAD_RERANKER=true)")
        trace = trace if trace is not None else Trace()

        with trace.stage("bm25"):
            sparse = [
                doc_id
                for doc_id, _ in self.bm25.search(tokenize(question), top_k=self.retrieval_depth)
            ]
        with trace.stage("dense"):
            with self._encode_lock:
                query_vector = self.encoder.encode_queries([question])[0]
            dense = [
                doc_id
                for doc_id, _ in self.dense_index.search(query_vector, top_k=self.retrieval_depth)
            ]
        with trace.stage("fusion"):
            fused = [
                doc_id
                for doc_id, _ in reciprocal_rank_fusion(
                    [sparse, dense],
                    k=self.rrf_k,
                    weights=[1.0, self.dense_weight],
                    top_k=max(self.context_k, self.rerank_depth if rerank else 0),
                )
            ]
        if rerank:
            with trace.stage("rerank"):
                candidates = [(doc_id, self.corpus[doc_id].text) for doc_id in fused]
                assert self.reranker is not None
                fused = [
                    doc_id
                    for doc_id, _ in self.reranker.rerank(
                        question, candidates[: self.rerank_depth], top_k=self.context_k
                    )
                ]
        return [self.corpus[doc_id] for doc_id in fused[: self.context_k]]

    def ask(self, question: str, rerank: bool = False) -> tuple[Answer, list[Document], Trace]:
        """Answer `question` from the retrieved documents.

        Returns the checked answer, the documents the LLM was shown (in the order of its
        [1], [2], ... numbers) and the per-stage trace. Raise LLMUnavailable if no LLM is
        configured and RerankerUnavailable as in `search`.
        """
        if self.llm is None:
            raise LLMUnavailable("no LLM configured (set HRE_LLM_BASE_URL and HRE_LLM_MODEL)")
        trace = Trace()
        documents = self.search(question, rerank=rerank, trace=trace)
        with trace.stage("generation"):
            answer = generate_answer(self.llm, question, [(d.id, d.text) for d in documents])
        return answer, documents, trace
