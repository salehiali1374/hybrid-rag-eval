"""Minimal client for any OpenAI-compatible chat endpoint (OpenRouter, Ollama, vLLM, ...)."""

import httpx


class OpenAICompatibleClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.timeout = timeout

    def complete(self, messages: list[dict[str, str]]) -> str:
        """The assistant's reply. Temperature 0 so that evaluations are repeatable."""
        response = httpx.post(
            self.url,
            headers=self.headers,
            json={"model": self.model, "messages": messages, "temperature": 0},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
