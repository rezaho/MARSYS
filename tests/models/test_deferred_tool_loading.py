"""Deferred tool loading (session 17) — per-adapter request-payload shapes (no network).

A per-tool ``defer_loading: true`` rides the OpenAI-shaped tool dict top-level. Each adapter
maps it onto its provider-native tool + auto-adds that provider's tool-search built-in so deferred
tools are discovered on demand (their schemas stay out of the billed/cached prefix). The two
providers without the feature handle it explicitly: openrouter STRIPS the flag (it would otherwise
reach the wire), google WARNS (it already drops the flag). The load-bearing guarantees:

- **Additive / identity:** with nothing deferred, each adapter's tool payload is byte-identical to
  before (no ``defer_loading`` key, no search tool).
- **Cache preservation:** the ``tools=`` prefix is byte-identical across a discovery round-trip —
  the discovered tool rides the message tail, never the ``tools=`` prefix (so the prompt cache holds).
"""
import pytest

from marsys.models.adapters.anthropic import AnthropicAdapter
from marsys.models.adapters.anthropic_oauth import AnthropicOAuthAdapter
from marsys.models.adapters.google import GoogleAdapter
from marsys.models.adapters.openai import OpenAIAdapter
from marsys.models.adapters.openai_oauth import OpenAIOAuthAdapter
from marsys.models.adapters.openrouter import OpenRouterAdapter

MSGS = [{"role": "user", "content": "hi"}]


def _tool(name="get_weather", *, defer=False):
    t = {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} description",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    if defer:
        t["defer_loading"] = True
    return t


# --- adapter fixtures (OAuth ones get patched creds so __init__ touches no keychain/network) ---


@pytest.fixture
def anthropic():
    return AnthropicAdapter(model_name="claude-sonnet-4-6", api_key="x", base_url="https://api.anthropic.com/v1")


@pytest.fixture
def anthropic_oauth(monkeypatch):
    monkeypatch.setattr(AnthropicOAuthAdapter, "_load_claude_credentials",
                        lambda self, path=None: {"access_token": "fake"})
    return AnthropicOAuthAdapter(model_name="claude-sonnet-4-6", auto_refresh=False)


@pytest.fixture
def openai():
    return OpenAIAdapter(model_name="gpt-5.4", api_key="x", base_url="https://api.openai.com/v1")


@pytest.fixture
def openai_oauth(monkeypatch):
    monkeypatch.setattr(OpenAIOAuthAdapter, "_load_codex_credentials",
                        lambda self, path=None: {"access_token": "fake", "account_id": "acct"})
    return OpenAIOAuthAdapter(model_name="gpt-5.4", auto_refresh=False)


@pytest.fixture
def openrouter():
    return OpenRouterAdapter(model_name="anthropic/claude-sonnet-4-6", api_key="x", base_url="https://openrouter.ai/api/v1")


@pytest.fixture
def google():
    return GoogleAdapter(model_name="gemini-3.5-flash", api_key="x", base_url="https://generativelanguage.googleapis.com")


def _names(tools):
    return [t.get("name") for t in tools if t.get("type") != "tool_search" and not str(t.get("type", "")).startswith("tool_search_tool")]


def _by_name(tools, name):
    return next((t for t in tools if t.get("name") == name), None)


# --- Anthropic (api-key + OAuth): defer_loading maps + the regex search tool is added ---


def test_anthropic_apikey_maps_defer_loading_and_adds_search_tool(anthropic):
    tools = anthropic.format_request_payload(
        MSGS, tools=[_tool("get_weather", defer=True), _tool("core_tool")]
    )["tools"]
    assert _by_name(tools, "get_weather")["defer_loading"] is True
    assert "defer_loading" not in _by_name(tools, "core_tool")  # non-deferred tool unchanged
    assert any(t.get("type") == "tool_search_tool_regex_20251119" for t in tools)  # search tool added


def test_anthropic_apikey_nothing_deferred_is_identical(anthropic):
    tools = anthropic.format_request_payload(MSGS, tools=[_tool("a"), _tool("b")])["tools"]
    assert _names(tools) == ["a", "b"]
    assert all("defer_loading" not in t for t in tools)
    assert not any(str(t.get("type", "")).startswith("tool_search_tool") for t in tools)  # no search tool


def test_anthropic_oauth_maps_defer_loading_and_adds_search_tool(anthropic_oauth):
    tools = anthropic_oauth.format_request_payload(
        MSGS, tools=[_tool("get_weather", defer=True), _tool("core_tool")]
    )["tools"]
    assert _by_name(tools, "get_weather")["defer_loading"] is True
    assert "defer_loading" not in _by_name(tools, "core_tool")
    assert any(t.get("type") == "tool_search_tool_regex_20251119" for t in tools)


def test_anthropic_oauth_nothing_deferred_is_identical(anthropic_oauth):
    tools = anthropic_oauth.format_request_payload(MSGS, tools=[_tool("a"), _tool("b")])["tools"]
    assert not any(str(t.get("type", "")).startswith("tool_search_tool") for t in tools)
    assert all("defer_loading" not in t for t in tools)


# --- OpenAI Responses (api-key + OAuth): defer_loading maps + the tool_search built-in is added ---


def test_openai_apikey_maps_defer_loading_and_adds_tool_search(openai):
    tools = openai.format_request_payload(
        MSGS, tools=[_tool("get_weather", defer=True), _tool("core_tool")]
    )["tools"]
    assert _by_name(tools, "get_weather")["defer_loading"] is True
    assert "defer_loading" not in _by_name(tools, "core_tool")
    assert any(t.get("type") == "tool_search" for t in tools)


def test_openai_apikey_nothing_deferred_is_identical(openai):
    tools = openai.format_request_payload(MSGS, tools=[_tool("a"), _tool("b")])["tools"]
    assert not any(t.get("type") == "tool_search" for t in tools)
    assert all("defer_loading" not in t for t in tools)


def test_openai_oauth_maps_defer_loading_and_adds_tool_search(openai_oauth):
    tools = openai_oauth.format_request_payload(
        MSGS, tools=[_tool("get_weather", defer=True), _tool("core_tool")]
    )["tools"]
    assert _by_name(tools, "get_weather")["defer_loading"] is True
    assert any(t.get("type") == "tool_search" for t in tools)


def test_openai_oauth_nothing_deferred_is_identical(openai_oauth):
    tools = openai_oauth.format_request_payload(MSGS, tools=[_tool("a"), _tool("b")])["tools"]
    assert not any(t.get("type") == "tool_search" for t in tools)


# --- providers without the feature: openrouter STRIPS, google WARNS (no silent behavior change) ---


def test_openrouter_strips_defer_loading_and_warns(openrouter):
    with pytest.warns(Warning, match="defer_loading"):
        tools = openrouter.format_request_payload(MSGS, tools=[_tool("get_weather", defer=True)])["tools"]
    # the flag must NOT reach the wire (verbatim forward would 400 on some providers)
    assert all("defer_loading" not in t for t in tools)


def test_openrouter_nothing_deferred_forwards_unchanged(openrouter):
    src = [_tool("a"), _tool("b")]
    tools = openrouter.format_request_payload(MSGS, tools=src)["tools"]
    assert tools == src  # byte-identical verbatim forward


def test_google_warns_on_deferred(google):
    with pytest.warns(Warning, match="defer_loading"):
        google.format_request_payload(MSGS, tools=[_tool("get_weather", defer=True)])


# --- cache preservation: the tools= prefix is byte-stable across a discovery round-trip ---


def test_anthropic_cache_prefix_stable_across_discovery_roundtrip(anthropic):
    tools = [_tool("get_weather", defer=True), _tool("core_tool")]
    turn1 = anthropic.format_request_payload([{"role": "user", "content": "weather in Paris?"}], tools=tools)
    # The model searched + called the discovered tool; that discovery rides the MESSAGES, not tools=.
    turn2 = anthropic.format_request_payload(
        [
            {"role": "user", "content": "weather in Paris?"},
            {"role": "assistant", "content": "searching",
             "tool_calls": [{"id": "c1", "function": {"name": "get_weather", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "name": "get_weather", "content": "{}"},
        ],
        tools=tools,
    )
    assert turn1["tools"] == turn2["tools"]  # tools= prefix byte-identical → prompt cache holds
