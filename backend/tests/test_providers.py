"""Provider adapters, tested offline with mocked transports."""

import dataclasses
import json
from types import SimpleNamespace

import httpx
import pytest

from app.config import get_settings
from app.llm import providers
from app.llm.providers import AnthropicLLM, GeminiLLM, LLMError, OpenAICompatibleLLM, build_llm
from app.models.schemas import LLMCrossExpertAnswer

VALID = {"insufficient_evidence": False, "answer": "ok", "expert_answers": [],
         "evidence": [{"evidence_id": "alpha_02_08", "highlight": None}]}


def settings(**kw):
    return dataclasses.replace(get_settings(), **kw)


def mock_httpx(monkeypatch, handler):
    real = httpx.Client
    monkeypatch.setattr(providers.httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))


def test_gemini_request_shape_and_parsing(monkeypatch):
    seen = {}

    def handler(req: httpx.Request):
        seen["url"], seen["key"] = str(req.url), req.headers.get("x-goog-api-key")
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": json.dumps(VALID)}]}}]})

    mock_httpx(monkeypatch, handler)
    llm = GeminiLLM(settings(llm_provider="gemini", llm_api_key="k-123", llm_model="gemini-2.5-flash"))
    out = llm.generate("sys", "user", LLMCrossExpertAnswer)
    assert out.evidence[0].evidence_id == "alpha_02_08"
    assert seen["url"].endswith("/models/gemini-2.5-flash:generateContent")
    assert "key=" not in seen["url"] and seen["key"] == "k-123"  # key in header, never in URL
    assert seen["body"]["generationConfig"]["responseMimeType"] == "application/json"
    assert "responseJsonSchema" in seen["body"]["generationConfig"]


def test_gemini_falls_back_when_schema_unsupported_and_retries_invalid_json(monkeypatch):
    calls = []

    def handler(req):
        body = json.loads(req.content)
        calls.append("schema" in json.dumps(body["generationConfig"]))
        if "responseJsonSchema" in body["generationConfig"]:
            return httpx.Response(400, json={"error": "unsupported"})
        text = "not json" if len(calls) == 2 else "```json\n" + json.dumps(VALID) + "\n```"
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": text}]}}]})

    mock_httpx(monkeypatch, handler)
    llm = GeminiLLM(settings(llm_api_key="k"))
    assert llm.generate("s", "u", LLMCrossExpertAnswer).answer == "ok"
    assert len(calls) == 3


def test_gemini_quota_error_is_friendly(monkeypatch):
    mock_httpx(monkeypatch, lambda req: httpx.Response(429, json={}))
    with pytest.raises(LLMError, match="quota"):
        GeminiLLM(settings(llm_api_key="k")).generate("s", "u", LLMCrossExpertAnswer)


def test_gemini_requires_key():
    llm, warning = build_llm(settings(llm_provider="gemini", llm_api_key=""))
    assert llm.is_mock and "LLM_API_KEY" in warning


def test_openai_compatible(monkeypatch):
    seen = {}

    def handler(req):
        seen["auth"] = req.headers.get("authorization")
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(VALID)}}]})

    mock_httpx(monkeypatch, handler)
    llm = OpenAICompatibleLLM(settings(llm_api_key="gsk", llm_model="llama-3.3-70b-versatile",
                                       llm_base_url="https://api.groq.com/openai/v1"))
    assert llm.generate("s", "u", LLMCrossExpertAnswer).answer == "ok"
    assert seen["auth"] == "Bearer gsk" and seen["body"]["response_format"] == {"type": "json_object"}


def test_openai_compatible_bad_key(monkeypatch):
    mock_httpx(monkeypatch, lambda req: httpx.Response(401, json={}))
    with pytest.raises(LLMError, match="API key"):
        OpenAICompatibleLLM(settings(llm_api_key="bad")).generate("s", "u", LLMCrossExpertAnswer)


class _FakeMessages:
    def __init__(self, response):
        self.response, self.kwargs = response, None

    def parse(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def _anthropic(response):
    llm = AnthropicLLM(settings(llm_provider="anthropic", llm_api_key="sk-test", llm_model="claude-opus-5"))
    llm.client = SimpleNamespace(messages=_FakeMessages(response))
    return llm


def test_anthropic_structured_output():
    parsed = LLMCrossExpertAnswer.model_validate(VALID)
    llm = _anthropic(SimpleNamespace(stop_reason="end_turn", parsed_output=parsed))
    assert llm.generate("sys", "user", LLMCrossExpertAnswer) is parsed
    kw = llm.client.messages.kwargs
    assert kw["model"] == "claude-opus-5" and kw["output_format"] is LLMCrossExpertAnswer and kw["system"] == "sys"


def test_anthropic_refusal_and_truncation_are_errors():
    with pytest.raises(LLMError, match="declined"):
        _anthropic(SimpleNamespace(stop_reason="refusal", parsed_output=None)).generate("s", "u", LLMCrossExpertAnswer)
    with pytest.raises(LLMError, match="truncated"):
        _anthropic(SimpleNamespace(stop_reason="max_tokens", parsed_output=None)).generate("s", "u", LLMCrossExpertAnswer)


def test_unknown_provider_falls_back_to_mock():
    llm, warning = build_llm(settings(llm_provider="nope"))
    assert llm.is_mock and "Unknown" in warning


def test_llm_failure_falls_back_to_extractive(app_state, with_fake_llm):
    def boom(user, schema):
        raise LLMError("Gemini rate limit / free-tier quota reached — retry shortly.")

    with_fake_llm(boom)
    from app.models.schemas import AskRequest

    r = app_state.research.ask(AskRequest(question="Which expert discusses training capacity as a barrier?"))
    assert r.sources and any("LLM unavailable" in n for n in r.meta.notices)


def test_gemini_invalid_key_reported_as_key_problem(monkeypatch):
    mock_httpx(monkeypatch, lambda req: httpx.Response(400, json={"error": {"message": "API key not valid. Please pass a valid API key.", "status": "INVALID_ARGUMENT"}}))
    with pytest.raises(LLMError, match="API key"):
        GeminiLLM(settings(llm_api_key="typo")).generate("s", "u", LLMCrossExpertAnswer)


def test_anthropic_ignores_machine_wide_base_url(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://localhost:9999/proxy")
    llm = AnthropicLLM(settings(llm_provider="anthropic", llm_api_key="sk-test", llm_base_url=""))
    assert str(llm.client.base_url).startswith("https://api.anthropic.com")
