import os

import requests


class OllamaClient:

    def __init__(
        self,
        model=None,
        base_url=None,
        timeout=240,
    ):
        self.model = (
            model
            or os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        )

        self.base_url = (
            base_url
            or os.getenv(
                "OLLAMA_API_BASE",
                "http://127.0.0.1:11434",
            )
        ).rstrip("/")

        self.timeout = timeout

    def _post(self, endpoint, payload):

        url = self.base_url + endpoint

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Ollama connection failed: {exc}"
            ) from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(
                "Ollama returned invalid JSON: "
                + response.text[:500]
            ) from exc

        if not isinstance(data, dict):
            raise RuntimeError(
                "Unexpected Ollama response type."
            )

        if not response.ok or "error" in data:
            raise RuntimeError(
                "Ollama API error: "
                + str(
                    data.get(
                        "error",
                        response.text[:500],
                    )
                )
            )

        return data

    def generate(self, prompt):

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
            },
        }

        data = self._post(
            "/api/generate",
            payload,
        )

        content = data.get("response")

        if not isinstance(content, str):
            raise RuntimeError(
                "Ollama generate response missing text."
            )

        return content

    def chat(self, messages, tools=None):

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0,
            },
        }

        if tools:
            payload["tools"] = tools

        data = self._post(
            "/api/chat",
            payload,
        )

        message = data.get("message")

        if not isinstance(message, dict):
            raise RuntimeError(
                "Ollama chat response missing message."
            )

        return data
