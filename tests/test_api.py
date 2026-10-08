import httpx
from fastapi.testclient import TestClient

from hybrid_rag.api.app import app, create_app

QUESTION = "پایتخت اسپانیا کجاست؟"


def test_health():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_before_the_pipeline_is_loaded():
    response = TestClient(create_app()).post("/query", json={"question": QUESTION})
    assert response.status_code == 503


class TestQuery:
    def test_answer_with_citations_and_trace(self, make_pipeline):
        client = TestClient(create_app(make_pipeline("پایتخت اسپانیا مادرید است [1].")))
        response = client.post("/query", json={"question": QUESTION})
        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "پایتخت اسپانیا مادرید است [1]."
        assert body["refused"] is False
        assert body["citations"] == [
            {"doc_id": "d-madrid", "title": "اسپانیا", "text": "مادرید پایتخت اسپانیا است."}
        ]
        assert body["retrieved"][0] == "d-madrid"
        assert set(body["trace"]["stages_ms"]) == {"bm25", "dense", "fusion", "generation"}
        assert body["trace"]["total_ms"] >= 0

    def test_refusal(self, make_pipeline):
        body = (
            TestClient(create_app(make_pipeline("NO_ANSWER")))
            .post("/query", json={"question": QUESTION})
            .json()
        )
        assert body["refused"] is True and body["answer"] == "" and body["citations"] == []

    def test_rerank_is_used_when_asked(self, make_pipeline):
        client = TestClient(create_app(make_pipeline("پاریس [1].", reranker=True)))
        body = client.post("/query", json={"question": QUESTION, "rerank": True}).json()
        assert body["citations"][0]["doc_id"] == "d-paris"
        assert "rerank" in body["trace"]["stages_ms"]

    def test_rerank_without_a_reranker_is_a_client_error(self, make_pipeline):
        client = TestClient(create_app(make_pipeline()))
        response = client.post("/query", json={"question": QUESTION, "rerank": True})
        assert response.status_code == 400

    def test_no_llm_is_service_unavailable(self, make_pipeline):
        client = TestClient(create_app(make_pipeline(llm=False)))
        assert client.post("/query", json={"question": QUESTION}).status_code == 503

    def test_llm_failure_is_bad_gateway_and_hides_details(self, make_pipeline):
        pipeline = make_pipeline()

        def broken(messages):
            raise httpx.ConnectError("http://secret-host:1234 refused")

        pipeline.llm.complete = broken
        response = TestClient(create_app(pipeline)).post("/query", json={"question": QUESTION})
        assert response.status_code == 502
        assert "secret-host" not in response.text

    def test_invalid_questions_are_rejected(self, make_pipeline):
        client = TestClient(create_app(make_pipeline()))
        assert client.post("/query", json={"question": "   "}).status_code == 422
        assert client.post("/query", json={"question": "x" * 1001}).status_code == 422
        assert client.post("/query", json={}).status_code == 422

    def test_one_json_log_line_per_query_without_the_question_text(self, make_pipeline, caplog):
        client = TestClient(create_app(make_pipeline("مادرید [1].")))
        with caplog.at_level("INFO", logger="hybrid_rag.query"):
            client.post("/query", json={"question": QUESTION})
        (record,) = [r for r in caplog.records if r.name == "hybrid_rag.query"]
        assert (
            '"event": "query"' in record.message and '"citations": ["d-madrid"]' in record.message
        )
        assert QUESTION not in record.message


class TestSearch:
    def test_returns_passages_without_calling_the_llm(self, make_pipeline):
        pipeline = make_pipeline(llm=False)
        body = TestClient(create_app(pipeline)).post("/search", json={"question": QUESTION}).json()
        assert body["passages"][0]["doc_id"] == "d-madrid"
        assert len(body["passages"]) == 2
        assert set(body["trace"]["stages_ms"]) == {"bm25", "dense", "fusion"}

    def test_rerank_without_a_reranker_is_a_client_error(self, make_pipeline):
        client = TestClient(create_app(make_pipeline()))
        assert (
            client.post("/search", json={"question": QUESTION, "rerank": True}).status_code == 400
        )
