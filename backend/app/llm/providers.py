"""LLM providers behind one interface: ``generate(system, user, schema) -> validated Pydantic object``.

* ``anthropic``         — Claude via the official SDK with native structured outputs (``messages.parse``).
* ``gemini``            — Google Gemini (free tier via Google AI Studio) with JSON-schema constrained output.
* ``openai_compatible`` — any Chat Completions endpoint (Groq, OpenRouter, Ollama, OpenAI, ...).
* ``mock``              — no network; services fall back to deterministic extractive answers.

Every provider's output is validated against the Pydantic schema; invalid JSON gets one corrective retry.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from ..config import Settings

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

DEFAULT_MODELS = {
    "anthropic": "claude-opus-5",
    "gemini": "gemini-2.5-flash",
    "openai_compatible": "llama-3.3-70b-versatile",
    "mock": "extractive-demo",
}


class LLMError(Exception):
    """Provider failure. The message is safe to show to users (no secrets, no stack traces)."""


class LLMService(ABC):
    provider: str
    model: str
    is_mock: bool = False

    @abstractmethod
    def generate(self, system: str, user: str, schema: type[T]) -> T: ...


def _extract_json(text: str) -> str:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        return fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start >= 0 and end > start else text


def _schema_instruction(schema: type[BaseModel]) -> str:
    return (
        "\n\nRespond with a single JSON object (no prose, no markdown) that validates against this JSON Schema:\n"
        + json.dumps(schema.model_json_schema())
    )


class MockLLM(LLMService):
    provider, model, is_mock = "mock", DEFAULT_MODELS["mock"], True

    def generate(self, system: str, user: str, schema: type[T]) -> T:  # pragma: no cover - never called
        raise LLMError("Mock mode does not call a language model.")


class AnthropicLLM(LLMService):
    provider = "anthropic"

    def __init__(self, settings: Settings):
        import anthropic

        self._anthropic = anthropic
        self.model = settings.llm_model or DEFAULT_MODELS["anthropic"]
        self.effort = settings.llm_effort
        kwargs = {"timeout": settings.llm_timeout_s, "max_retries": 2}
        if settings.llm_api_key:
            kwargs["api_key"] = settings.llm_api_key
        # Explicit base URL: the SDK would otherwise pick up a machine-wide ANTHROPIC_BASE_URL (e.g. a local proxy)
        # and silently route this app's traffic through it. LLM_BASE_URL is the only override.
        kwargs["base_url"] = settings.llm_base_url or "https://api.anthropic.com"
        self.client = anthropic.Anthropic(**kwargs)

    def generate(self, system: str, user: str, schema: type[T]) -> T:
        a = self._anthropic
        extra = {"output_config": {"effort": self.effort}} if self.effort else {}
        try:
            response = self.client.messages.parse(
                model=self.model,
                max_tokens=16000,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_format=schema,
                **extra,
            )
        except a.AuthenticationError as exc:
            raise LLMError("Anthropic rejected the API key (check LLM_API_KEY).") from exc
        except a.RateLimitError as exc:
            raise LLMError("Anthropic rate limit reached — retry shortly.") from exc
        except a.BadRequestError as exc:
            raise LLMError(f"Anthropic rejected the request: {getattr(exc, 'message', 'bad request')}") from exc
        except a.APIStatusError as exc:
            raise LLMError(f"Anthropic API error ({exc.status_code}).") from exc
        except a.APIConnectionError as exc:
            raise LLMError("Could not reach the Anthropic API.") from exc
        if response.stop_reason == "refusal":
            raise LLMError("The model declined to answer this request.")
        if response.stop_reason == "max_tokens":
            raise LLMError("The model response was truncated.")
        parsed = response.parsed_output
        if parsed is None:
            raise LLMError("The model returned no structured output.")
        return parsed


class _JSONHttpLLM(LLMService):
    """Shared validate-and-retry loop for HTTP providers that return JSON text."""

    timeout: float = 120

    @abstractmethod
    def _call(self, system: str, user: str, schema: type[BaseModel]) -> str: ...

    def generate(self, system: str, user: str, schema: type[T]) -> T:
        prompt = user
        last_error = ""
        for attempt in range(2):
            try:
                raw = self._call(system, prompt, schema)
            except httpx.HTTPStatusError as exc:
                code = exc.response.status_code
                body = exc.response.text[:2000].lower()
                # Gemini reports a bad key as HTTP 400 INVALID_ARGUMENT, not 401
                if code in (401, 403) or "api key not valid" in body or "api_key_invalid" in body:
                    raise LLMError(f"{self.provider} rejected the API key (check LLM_API_KEY).") from exc
                if code == 429:
                    raise LLMError(f"{self.provider} rate limit / free-tier quota reached — retry shortly.") from exc
                raise LLMError(f"{self.provider} API error ({code}).") from exc
            except httpx.HTTPError as exc:
                raise LLMError(f"Could not reach the {self.provider} API.") from exc
            try:
                return schema.model_validate_json(_extract_json(raw))
            except ValidationError as exc:
                last_error = str(exc)[:1500]
                log.warning("%s returned invalid JSON (attempt %d)", self.provider, attempt + 1)
                prompt = (f"{user}\n\nYour previous reply did not validate:\n{last_error}\n"
                          "Return ONLY the corrected JSON object.")
        raise LLMError(f"{self.provider} returned output that did not match the required schema.")


class GeminiLLM(_JSONHttpLLM):
    provider = "gemini"
    BASE = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, settings: Settings):
        if not settings.llm_api_key:
            raise LLMError("LLM_API_KEY is required for Gemini (free key: https://aistudio.google.com/apikey).")
        self.api_key = settings.llm_api_key
        self.model = settings.llm_model or DEFAULT_MODELS["gemini"]
        self.base = (settings.llm_base_url or self.BASE).rstrip("/")
        self.timeout = settings.llm_timeout_s
        self._schema_supported = True

    def _call(self, system: str, user: str, schema: type[BaseModel]) -> str:
        config: dict = {"temperature": 0.1, "responseMimeType": "application/json"}
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user + _schema_instruction(schema)}]}],
            "generationConfig": config,
        }
        with httpx.Client(timeout=self.timeout) as client:
            url = f"{self.base}/models/{self.model}:generateContent"
            headers = {"x-goog-api-key": self.api_key}
            if self._schema_supported:
                config["responseJsonSchema"] = schema.model_json_schema()
                resp = client.post(url, headers=headers, json=body)
                if resp.status_code == 400:  # older models/endpoints without JSON-schema support
                    self._schema_supported = False
                    config.pop("responseJsonSchema", None)
                    resp = client.post(url, headers=headers, json=body)
            else:
                resp = client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            raise LLMError("Gemini returned no candidates (the prompt may have been blocked).")
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts)


class OpenAICompatibleLLM(_JSONHttpLLM):
    provider = "openai_compatible"

    def __init__(self, settings: Settings):
        self.api_key = settings.llm_api_key
        self.model = settings.llm_model or DEFAULT_MODELS["openai_compatible"]
        self.base = (settings.llm_base_url or "https://api.groq.com/openai/v1").rstrip("/")
        self.timeout = settings.llm_timeout_s
        self._json_mode = True

    def _call(self, system: str, user: str, schema: type[BaseModel]) -> str:
        body: dict = {
            "model": self.model,
            "temperature": 0.1,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user + _schema_instruction(schema)}],
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        with httpx.Client(timeout=self.timeout) as client:
            if self._json_mode:
                body["response_format"] = {"type": "json_object"}
            resp = client.post(f"{self.base}/chat/completions", headers=headers, json=body)
            if resp.status_code == 400 and self._json_mode:
                self._json_mode = False
                body.pop("response_format", None)
                resp = client.post(f"{self.base}/chat/completions", headers=headers, json=body)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"] or ""


def build_llm(settings: Settings) -> tuple[LLMService, str | None]:
    """Return (service, warning). Misconfiguration falls back to mock mode instead of crashing."""
    provider = settings.llm_provider
    try:
        if provider == "anthropic":
            return AnthropicLLM(settings), None
        if provider == "gemini":
            return GeminiLLM(settings), None
        if provider in {"openai", "openai_compatible", "groq", "openrouter", "ollama"}:
            return OpenAICompatibleLLM(settings), None
        if provider != "mock":
            return MockLLM(), f"Unknown LLM_PROVIDER '{provider}'; running in extractive demo mode."
    except LLMError as exc:
        return MockLLM(), f"{exc} Running in extractive demo mode."
    except Exception as exc:  # e.g. SDK missing credentials
        return MockLLM(), f"LLM provider '{provider}' could not be initialised ({exc.__class__.__name__}); demo mode."
    return MockLLM(), None
