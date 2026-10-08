"""Specification for hybrid_rag.generation.answer. No real LLM is called."""

import httpx
import pytest

from hybrid_rag.generation.answer import (
    REFUSAL_MARKER,
    Answer,
    build_messages,
    generate_answer,
    parse_answer,
)
from hybrid_rag.generation.client import OpenAICompatibleClient

PASSAGES = [("doc-a", "تهران پایتخت ایران است."), ("doc-b", "اصفهان شهری تاریخی است.")]


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def complete(self, messages):
        self.calls.append(messages)
        return self.reply


class TestBuildMessages:
    def test_passages_are_numbered_from_one_in_ranked_order(self):
        system, user = build_messages("پایتخت کجاست؟", PASSAGES)
        assert system["role"] == "system" and user["role"] == "user"
        assert REFUSAL_MARKER in system["content"]
        assert "[1] تهران پایتخت ایران است." in user["content"]
        assert "[2] اصفهان شهری تاریخی است." in user["content"]
        assert user["content"].endswith("Question: پایتخت کجاست؟")

    def test_doc_ids_are_not_shown_to_the_model(self):
        _, user = build_messages("q", PASSAGES)
        assert "doc-a" not in user["content"]


class TestParseAnswer:
    IDS = ["doc-a", "doc-b", "doc-c"]

    def test_citations_are_mapped_to_doc_ids(self):
        answer = parse_answer("تهران [1] و اصفهان [3].", self.IDS)
        assert answer == Answer(text="تهران [1] و اصفهان [3].", citations=["doc-a", "doc-c"])

    def test_repeated_citation_is_listed_once_in_order_of_first_use(self):
        assert parse_answer("الف [2] ب [1] ج [2]", self.IDS).citations == ["doc-b", "doc-a"]

    def test_several_numbers_in_one_bracket(self):
        assert parse_answer("جواب [1, 3]", self.IDS).citations == ["doc-a", "doc-c"]
        assert parse_answer("جواب [1،2]", self.IDS).citations == ["doc-a", "doc-b"]

    def test_persian_digits(self):
        assert parse_answer("جواب [۲]", self.IDS).citations == ["doc-b"]

    def test_hallucinated_citation_is_reported_not_mapped(self):
        answer = parse_answer("جواب [1] و [7] و [0]", self.IDS)
        assert answer.citations == ["doc-a"]
        assert answer.invalid_citations == [7, 0]

    def test_refusal(self):
        answer = parse_answer(f"  {REFUSAL_MARKER}\n", self.IDS)
        assert answer.refused and answer.text == "" and answer.citations == []

    def test_answer_without_citation_has_none(self):
        answer = parse_answer("تهران است.", self.IDS)
        assert not answer.refused and answer.citations == [] and answer.invalid_citations == []


class TestGenerateAnswer:
    def test_sends_prompt_and_parses_reply(self):
        llm = FakeLLM("پایتخت ایران تهران است [1].")
        answer = generate_answer(llm, "پایتخت کجاست؟", PASSAGES)
        assert answer.citations == ["doc-a"]
        assert llm.calls[0] == build_messages("پایتخت کجاست؟", PASSAGES)


class TestOpenAICompatibleClient:
    def test_request_and_response(self, monkeypatch):
        seen = {}

        def fake_post(url, headers, json, timeout):
            seen.update(url=url, headers=headers, json=json)
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "سلام"}}]},
                request=httpx.Request("POST", url),
            )

        monkeypatch.setattr(httpx, "post", fake_post)
        client = OpenAICompatibleClient("https://example.test/v1/", "some-model", api_key="k")
        assert client.complete([{"role": "user", "content": "hi"}]) == "سلام"
        assert seen["url"] == "https://example.test/v1/chat/completions"
        assert seen["headers"] == {"Authorization": "Bearer k"}
        assert seen["json"]["model"] == "some-model"
        assert seen["json"]["temperature"] == 0

    def test_http_error_is_raised(self, monkeypatch):
        def fake_post(url, headers, json, timeout):
            return httpx.Response(401, request=httpx.Request("POST", url))

        monkeypatch.setattr(httpx, "post", fake_post)
        with pytest.raises(httpx.HTTPStatusError):
            OpenAICompatibleClient("https://example.test/v1", "m").complete([])
