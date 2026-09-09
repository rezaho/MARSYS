"""Configured depth survives BaseAPIModel and the actual OAuth payload builder."""

import pytest

from marsys.models.adapters.openai_oauth import OpenAIOAuthAdapter
from marsys.models.models import BaseAPIModel


MESSAGES = [{"role": "user", "content": "Continue the work."}]


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "configured, overrides, expected",
    [
        ({"reasoning_effort": "low"}, {}, "low"),
        ({"reasoning_effort": "medium"}, {}, "medium"),
        ({"reasoning_effort": "high"}, {}, "high"),
        ({"reasoning_effort": "minimal"}, {}, "minimal"),
        ({"reasoning_effort": "LOW"}, {}, "low"),
        ({"reasoning_effort": "high"}, {"reasoning_effort": "low"}, "low"),
        ({"reasoning_effort": "high"}, {"reasoning_effort": None}, None),
        ({}, {}, None),
        ({"thinking_budget": 0}, {}, None),
        ({"thinking_budget": -1}, {}, None),
        ({"thinking_budget": 512}, {}, "minimal"),
        ({"thinking_budget": 1024}, {}, "low"),
        ({"thinking_budget": 8192}, {}, "medium"),
        ({"thinking_budget": 32768}, {}, "high"),
        ({"thinking_budget": 32768, "reasoning_effort": "low"}, {}, "low"),
        ({"thinking_budget": 0, "reasoning_effort": "low"}, {}, "low"),
    ],
)
async def test_model_depth_reaches_oauth_transport(
    monkeypatch, asynchronous, configured, overrides, expected
):
    monkeypatch.setattr(
        OpenAIOAuthAdapter,
        "_load_codex_credentials",
        lambda self, path: {"access_token": "fake-token", "account_id": "fake-account"},
    )
    model = BaseAPIModel(
        model_name="gpt-5", provider="openai-oauth", api_key="unused", base_url="",
        credentials_path="unused-fake-credentials", auto_refresh=False, **configured,
    )
    captured = []

    def capture(endpoint, headers, payload):
        captured.append((endpoint, headers, payload))
        return {
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}],
            "usage": {},
        }

    async def async_capture(endpoint, headers, payload, **kwargs):
        return capture(endpoint, headers, payload)

    monkeypatch.setattr(model.adapter, "_sync_stream_response", capture)
    monkeypatch.setattr(model.async_adapter, "_async_stream_response", async_capture)
    for _ in range(2):
        if asynchronous:
            await model.arun(MESSAGES, prompt_cache_key="install:owner", **overrides)
        else:
            model.run(MESSAGES, prompt_cache_key="install:owner", **overrides)

    assert len(captured) == 2
    assert captured[0] == captured[1]
    endpoint, headers, payload = captured[0]
    reasoning = {"summary": "auto"}
    if expected is not None:
        reasoning["effort"] = expected
    assert payload["reasoning"] == reasoning
    assert payload["prompt_cache_key"] == "install:owner"
    assert payload["stream"] is True
    assert payload["store"] is False
    assert payload["include"] == ["reasoning.encrypted_content"]
    assert "reasoning_effort" not in payload
    assert "thinking_budget" not in payload
    assert endpoint == OpenAIOAuthAdapter.RESPONSES_ENDPOINT
    assert headers["Authorization"] == "Bearer fake-token"
    assert headers["chatgpt-account-id"] == "fake-account"


@pytest.mark.parametrize("kwargs", [{"reasoning_effort": "minimal"}, {"thinking_budget": 512}])
def test_codex_smallest_effort_matches_existing_openai_mapping(kwargs):
    adapter = object.__new__(OpenAIOAuthAdapter)
    adapter.model_name = "gpt-5.3-codex"
    payload = adapter.format_request_payload(MESSAGES, **kwargs)
    assert payload["reasoning"] == {"summary": "auto", "effort": "low"}
