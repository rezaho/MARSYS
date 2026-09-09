"""Minimal stays positive on documented models whose lowest effort is low."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from marsys.models.adapters.openai import OpenAIAdapter
from marsys.models.adapters.openai_oauth import OpenAIOAuthAdapter
from marsys.models.models import BaseAPIModel


MESSAGES = [{"role": "user", "content": "Continue."}]
RAW_RESPONSE = {
    "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}],
    "usage": {},
}


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "provider, model_name",
    [
        ("openai", "gpt-5.4-mini"), ("openai-oauth", "gpt-5.4-mini"),
        ("openai", "gpt-5.5"), ("openai-oauth", "gpt-5.5"),
        ("openai", "gpt-5"), ("openai-oauth", "gpt-5"),
        ("openai", "gpt-5.4-nano"),
    ],
)
@pytest.mark.parametrize(
    "configured, expected",
    [
        ({"reasoning_effort": "minimal"}, "minimal"),
        ({"thinking_budget": 512}, "minimal"),
        ({"reasoning_effort": "low"}, "low"),
        ({"reasoning_effort": "medium"}, "medium"),
        ({"reasoning_effort": "high"}, "high"),
        ({}, None),
        ({"thinking_budget": 0}, None),
    ],
)
async def test_configured_effort_reaches_transport_in_supported_form(
    monkeypatch, provider, asynchronous, model_name, configured, expected
):
    monkeypatch.setattr(
        OpenAIOAuthAdapter, "_load_codex_credentials",
        lambda self, path: {"access_token": "fake-token", "account_id": "fake-account"},
    )
    oauth = provider == "openai-oauth"
    model = BaseAPIModel(
        model_name=model_name, provider=provider, api_key="fake-key",
        base_url="https://example.invalid/v1",
        **({"credentials_path": "unused", "auto_refresh": False} if oauth else {}),
        **configured,
    )
    captured = []

    if oauth:
        def capture(endpoint, headers, payload):
            captured.append(payload)
            return RAW_RESPONSE

        async def async_capture(endpoint, headers, payload, **kwargs):
            return capture(endpoint, headers, payload)

        monkeypatch.setattr(model.adapter, "_sync_stream_response", capture)
        monkeypatch.setattr(model.async_adapter, "_async_stream_response", async_capture)
    else:
        def post(url, *, json, **kwargs):
            captured.append(json)
            response = MagicMock(status_code=200, status=200)
            response.raise_for_status.return_value = None
            response.json.return_value = RAW_RESPONSE
            async_response = MagicMock(status=200)
            async_response.raise_for_status.return_value = None
            async_response.json = AsyncMock(return_value=RAW_RESPONSE)
            response.__aenter__ = AsyncMock(return_value=async_response)
            response.__aexit__ = AsyncMock(return_value=False)
            return response

        monkeypatch.setattr("marsys.models.adapters.base.requests.post", post)
        session = MagicMock()
        session.post.side_effect = post
        monkeypatch.setattr(model.async_adapter, "_ensure_session", AsyncMock(return_value=session))

    if asynchronous:
        await model.arun(MESSAGES, prompt_cache_key="install:owner")
    else:
        model.run(MESSAGES, prompt_cache_key="install:owner")

    assert len(captured) == 1
    payload = captured[0]
    if expected == "minimal" and model_name != "gpt-5":
        expected = "low"
    reasoning = {"summary": "auto"} if oauth else {}
    if expected is not None:
        reasoning["effort"] = expected
    assert payload.get("reasoning", {}) == reasoning
    assert payload["prompt_cache_key"] == "install:owner"
    if oauth:
        assert payload["stream"] is True
        assert payload["store"] is False
        assert payload["include"] == ["reasoning.encrypted_content"]


@pytest.mark.parametrize("adapter_type", [OpenAIAdapter, OpenAIOAuthAdapter])
@pytest.mark.parametrize(
    "model_name, expected",
    [
        ("gpt-5.1", "low"),
        ("gpt-5.1-2025-11-13", "low"),
        ("gpt-5.2", "low"),
        ("gpt-5.2-2025-12-11", "low"),
        ("gpt-5.4", "low"),
        ("gpt-5.4-2026-03-05", "low"),
        ("gpt-5.4-mini", "low"),
        ("gpt-5.4-mini-2026-03-17", "low"),
        ("gpt-5.4-nano", "low"),
        ("gpt-5.4-nano-2026-03-17", "low"),
        ("gpt-5.5", "low"),
        ("gpt-5.5-2026-04-23", "low"),
        ("gpt-5.3-codex", "low"),
        ("gpt-5", "minimal"),
        ("gpt-5-mini", "minimal"),
        ("gpt-5-nano", "minimal"),
        ("gpt-5.5-pro", "minimal"),
        ("gpt-5.50", "minimal"),
        ("gpt-5.6", "minimal"),
    ],
)
def test_minimal_compatibility_is_bounded_to_verified_models(adapter_type, model_name, expected):
    if adapter_type is OpenAIOAuthAdapter:
        adapter = object.__new__(adapter_type)
        adapter.model_name = model_name
    else:
        adapter = adapter_type(
            model_name=model_name, api_key="fake-key", base_url="https://example.invalid/v1"
        )
    payload = adapter.format_request_payload(MESSAGES, reasoning_effort="minimal")
    assert payload["reasoning"]["effort"] == expected
