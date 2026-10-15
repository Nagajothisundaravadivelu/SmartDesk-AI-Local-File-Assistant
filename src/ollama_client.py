from __future__ import annotations

from typing import Any

import requests

from .config import DEFAULT_MODEL, OLLAMA_BASE_URL, REQUEST_TIMEOUT


class OllamaError(RuntimeError):
    """Raised when Ollama cannot be reached or returns an unusable response."""


class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = DEFAULT_MODEL, timeout: float = REQUEST_TIMEOUT) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _request_json(self, endpoint: str, payload: dict[str, Any] | None = None, method: str = "POST") -> dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        try:
            if method.upper() == "GET":
                response = requests.get(url, timeout=self.timeout)
            else:
                response = requests.post(url, json=payload, timeout=self.timeout)
        except requests.exceptions.Timeout as exc:
            raise OllamaError("The local Ollama service took too long to respond.") from exc
        except requests.exceptions.ConnectionError as exc:
            raise OllamaError("Ollama is not reachable at the configured local endpoint.") from exc
        except requests.exceptions.RequestException as exc:
            raise OllamaError(f"A request to Ollama failed: {exc}") from exc

        if response.status_code == 404:
            detail = response.text.strip()
            if "model" in detail.lower() and "not found" in detail.lower():
                raise OllamaError(
                    f"The Ollama model '{self.model}' is not installed. Run 'ollama pull {self.model}' and try again."
                )
            raise OllamaError("The Ollama endpoint was not found. Check that the local service is running.")
        if response.status_code >= 400:
            detail = response.text.strip() or "unknown error"
            raise OllamaError(f"Ollama returned an error: {detail}")

        try:
            data = response.json()
        except ValueError as exc:
            raise OllamaError("Ollama returned an invalid response payload.") from exc

        if not isinstance(data, dict):
            raise OllamaError("Ollama returned an unexpected response format.")
        return data

    def check_connection(self) -> bool:
        try:
            self._request_json("/api/tags", method="GET")
            return True
        except OllamaError:
            return False

    def check_model_available(self) -> bool:
        try:
            payload = self._request_json("/api/tags", method="GET")
        except OllamaError:
            return False

        models = payload.get("models", [])
        available = []
        for model in models:
            if isinstance(model, dict):
                name = model.get("name")
                if name:
                    available.append(name)
            elif isinstance(model, str):
                available.append(model)

        return self.model in available or self.model.split(":", 1)[0] in {name.split(":", 1)[0] for name in available}

    def generate_chat_response(self, prompt: str, system_prompt: str | None = None) -> str:
        if not prompt or not prompt.strip():
            raise OllamaError("The prompt is empty and cannot be sent to Ollama.")

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.2},
        }

        response = self._request_json("/api/chat", payload=payload)
        content = response.get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise OllamaError("Ollama produced an empty or invalid response.")
        return content.strip()
