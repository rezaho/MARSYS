"""Caller-selected cache routing survives each OpenAI-family payload builder."""

import pytest

from marsys.models.adapters.azure import AsyncAzureOpenAIAdapter, AzureOpenAIAdapter
from marsys.models.adapters.openai import AsyncOpenAIAdapter, OpenAIAdapter
from marsys.models.adapters.openai_oauth import AsyncOpenAIOAuthAdapter, OpenAIOAuthAdapter


@pytest.fixture(params=[
    OpenAIAdapter, AsyncOpenAIAdapter, AzureOpenAIAdapter, AsyncAzureOpenAIAdapter,
    OpenAIOAuthAdapter, AsyncOpenAIOAuthAdapter,
])
def adapter(request):
    adapter_type = request.param
    if issubclass(adapter_type, OpenAIOAuthAdapter):
        # Payload construction needs no credential discovery or network client.
        result = object.__new__(adapter_type)
        result.model_name = "gpt-5"
        return result
    return adapter_type(
        model_name="gpt-5", api_key="not-a-real-key",
        base_url="https://example.invalid/openai/v1", max_tokens=1024,
    )


def test_caller_key_reaches_payload_and_stays_stable(adapter):
    messages = [{"role": "user", "content": "Continue the work."}]
    key = "installation:instance"
    first = adapter.format_request_payload(messages, prompt_cache_key=key)
    second = adapter.format_request_payload(messages, prompt_cache_key=key)
    assert first["prompt_cache_key"] == second["prompt_cache_key"] == key
    other = adapter.format_request_payload(messages, prompt_cache_key="installation:other")
    assert other["prompt_cache_key"] == "installation:other"


@pytest.mark.parametrize("kwargs", [{}, {"prompt_cache_key": None}])
def test_absent_key_is_omitted_without_minting_one(adapter, kwargs):
    messages = [{"role": "user", "content": "Continue the work."}]
    first = adapter.format_request_payload(messages, **kwargs)
    second = adapter.format_request_payload(messages, **kwargs)
    assert "prompt_cache_key" not in first
    assert "prompt_cache_key" not in second
    assert first == second
