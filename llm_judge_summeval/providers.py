"""Minimal provider clients with injectable HTTP transport for deterministic tests."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable


class ProviderError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = False, status: int | None = None):
        super().__init__(message)
        self.retryable, self.status = retryable, status


Transport = Callable[[str, dict[str, str], dict, float], dict]


def urllib_transport(url: str, headers: dict[str, str], payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        body = error.read().decode(errors="replace")[:2000]
        raise ProviderError(f"HTTP {error.code}: {body}", retryable=error.code in {408, 409, 429} or error.code >= 500, status=error.code) from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise ProviderError(str(error), retryable=True) from error


@dataclass(frozen=True)
class ProviderResponse:
    parsed: dict
    response_id: str
    model: str
    stop_reason: str | None
    input_tokens: int
    output_tokens: int
    raw: dict


class AnthropicProvider:
    endpoint = "https://api.anthropic.com/v1/messages"

    def __init__(self, model: str, *, transport: Transport = urllib_transport, timeout: float = 120):
        self.model, self.transport, self.timeout = model, transport, timeout

    def request(self, system: str, user: str, schema: dict, *, max_tokens: int = 900) -> ProviderResponse:
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ProviderError("ANTHROPIC_API_KEY is missing")
        payload = {
            "model": self.model, "max_tokens": max_tokens, "temperature": 0,
            "system": system, "messages": [{"role": "user", "content": user}],
            "output_config": {"format": {"type": "json_schema", "schema": schema}},
        }
        raw = self.transport(self.endpoint, {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}, payload, self.timeout)
        text = next((block["text"] for block in raw.get("content", []) if block.get("type") == "text"), None)
        if text is None:
            raise ProviderError("Anthropic response contained no text block")
        usage = raw.get("usage", {})
        return ProviderResponse(json.loads(text), raw.get("id", ""), raw.get("model", self.model), raw.get("stop_reason"),
                                int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0)), raw)


class OpenAIProvider:
    endpoint = "https://api.openai.com/v1/responses"

    def __init__(self, model: str, *, transport: Transport = urllib_transport, timeout: float = 120):
        self.model, self.transport, self.timeout = model, transport, timeout

    def request(self, system: str, user: str, schema: dict, *, max_tokens: int = 900) -> ProviderResponse:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ProviderError("OPENAI_API_KEY is missing")
        payload = {
            "model": self.model, "instructions": system, "input": user, "temperature": 0,
            "max_output_tokens": max_tokens, "store": False,
            "text": {"format": {"type": "json_schema", "name": "summeval_judgment", "strict": True, "schema": schema}},
        }
        raw = self.transport(self.endpoint, {"authorization": f"Bearer {key}", "content-type": "application/json"}, payload, self.timeout)
        texts = []
        for output in raw.get("output", []):
            for content in output.get("content", []):
                if content.get("type") == "output_text":
                    texts.append(content["text"])
        if not texts:
            raise ProviderError("OpenAI response contained no output_text")
        usage = raw.get("usage", {})
        return ProviderResponse(json.loads("".join(texts)), raw.get("id", ""), raw.get("model", self.model), raw.get("status"),
                                int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0)), raw)


def build_provider(name: str, model: str, *, transport: Transport = urllib_transport):
    if name == "anthropic":
        return AnthropicProvider(model, transport=transport)
    if name == "openai":
        return OpenAIProvider(model, transport=transport)
    raise ValueError(f"Unsupported provider: {name}")
