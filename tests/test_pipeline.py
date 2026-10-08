"""Specification for hybrid_rag.pipeline, using the fake models of conftest.py."""

import pytest

from hybrid_rag.pipeline import LLMUnavailable, RerankerUnavailable, Trace

QUESTION = "پایتخت اسپانیا کجاست؟"


class TestTrace:
    def test_records_stages_in_order_and_sums_them(self):
        trace = Trace()
        with trace.stage("a"):
            pass
        with trace.stage("b"):
            pass
        assert list(trace.stages_ms) == ["a", "b"]
        assert trace.total_ms == pytest.approx(sum(trace.stages_ms.values()), abs=0.2)

    def test_stage_is_recorded_even_if_it_raises(self):
        trace = Trace()
        with pytest.raises(ValueError), trace.stage("boom"):
            raise ValueError
        assert "boom" in trace.stages_ms


class TestSearch:
    def test_returns_context_k_best_documents(self, make_pipeline):
        documents = make_pipeline().search(QUESTION)
        assert [d.id for d in documents][0] == "d-madrid"
        assert len(documents) == 2  # context_k

    def test_stages_are_traced(self, make_pipeline):
        trace = Trace()
        make_pipeline().search(QUESTION, trace=trace)
        assert list(trace.stages_ms) == ["bm25", "dense", "fusion"]

    def test_rerank_reorders_candidates(self, make_pipeline):
        trace = Trace()
        documents = make_pipeline(reranker=True).search(QUESTION, rerank=True, trace=trace)
        assert documents[0].id == "d-paris"
        assert list(trace.stages_ms) == ["bm25", "dense", "fusion", "rerank"]

    def test_rerank_without_a_reranker(self, make_pipeline):
        with pytest.raises(RerankerUnavailable):
            make_pipeline().search(QUESTION, rerank=True)


class TestAsk:
    def test_answer_is_checked_against_the_documents_shown(self, make_pipeline):
        pipeline = make_pipeline("مادرید [1] و [9].")
        answer, documents, trace = pipeline.ask(QUESTION)
        assert answer.citations == ["d-madrid"]
        assert answer.invalid_citations == [9]
        assert list(trace.stages_ms) == ["bm25", "dense", "fusion", "generation"]
        prompt = pipeline.fake_llm.calls[0][1]["content"]
        assert prompt.index("[1] مادرید") < prompt.index("[2]")  # best document is [1]

    def test_refusal(self, make_pipeline):
        answer, _, _ = make_pipeline("NO_ANSWER").ask(QUESTION)
        assert answer.refused and answer.citations == []

    def test_without_an_llm(self, make_pipeline):
        with pytest.raises(LLMUnavailable):
            make_pipeline(llm=False).ask(QUESTION)
