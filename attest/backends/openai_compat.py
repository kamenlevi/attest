"""A real Generator backend that talks to any OpenAI-compatible chat endpoint.

This is how "use any model, local or cloud" works with one piece of code: the
OpenAI chat-completions format is spoken by
  - cloud providers: OpenAI, OpenRouter, Together, Anthropic (via a gateway), ...
  - local servers:    Ollama, LM Studio, vLLM, the MLX servers
So the user just supplies a base URL + model name (+ an API key for cloud).

It runs anywhere with internet — including your Linux ThinkPad, today. No MLX,
no GPU. Uses only the Python standard library (no extra dependencies).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from ..interfaces import Generator


def _default_post(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    """Send one JSON POST and return the decoded JSON response (stdlib only)."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={**headers, "Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class OpenAICompatibleGenerator(Generator):
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        timeout: float = 120.0,
        post_fn=_default_post,
    ) -> None:
        # base_url should be the API root, e.g. "https://api.openai.com/v1"
        # or "http://localhost:11434/v1" for a local Ollama server.
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        # Dependency injection: tests pass a fake post_fn so this is testable
        # with no network and no API key.
        self._post = post_fn

    def generate(self, prompt: str) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        response = self._post(url, payload, headers, self.timeout)
        return response["choices"][0]["message"]["content"]
