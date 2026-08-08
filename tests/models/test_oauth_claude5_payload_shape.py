"""Payload shaping on the Anthropic OAuth leg for Claude 5 (no network).

The OAuth adapter builds its own payload, so the capability shaping proven for
the API-key adapter does not carry over for free. This leg previously set
``temperature`` unconditionally, which is a live 400 on every reasoning-capable
model (Opus 4.7/4.8 included, not just the Claude 5 pair).

Note the two legs are NOT identical by measurement: a fixed ``budget_tokens`` is
still accepted here for Claude 5 while Bedrock rejects it. Adaptive is sent
anyway — it is accepted on both and is the shape the models are tuned for — but
that is a deliberate choice, not a forced one.
"""

import json
import time

import pytest

from marsys.models.adapters.anthropic import AnthropicAdapter
from marsys.models.adapters.anthropic_oauth import AnthropicOAuthAdapter

MESSAGES = [{"role": "user", "content": "hi"}]
SCHEMA = {
    "type": "object",
    "properties": {"units": {"type": "array", "items": {"type": "string"}}},
    "required": ["units"],
}


@pytest.fixture(autouse=True)
def _dummy_credentials(tmp_path, monkeypatch):
    """A credentials file the real constructor can load, at the path it already reads from
    the environment. The alternative — building the adapter through ``__new__`` and hand-setting
    the five attributes the payload builder happens to read today — measures a hand-assembled
    object: anything ``__init__`` does that shapes a payload (alias resolution, defaulting, a
    capability read) is invisible to it, and the parity arms then compare that object against a
    real ``AnthropicAdapter``. Nothing here reaches the network."""
    path = tmp_path / ".credentials.json"
    path.write_text(json.dumps({"claudeAiOauth": {
        "accessToken": "dummy-access-token",
        "refreshToken": "dummy-refresh-token",
        "expiresAt": int((time.time() + 3600) * 1000),
        "subscriptionType": "max",
    }}))
    monkeypatch.setenv("CLAUDE_AUTH_PATH", str(path))


def _oauth(model_name: str, *, budget: int = 0, enable: bool = False) -> AnthropicOAuthAdapter:
    """The real constructor, on the dummy credentials above. ``auto_refresh`` is off because a
    refresh is an OAuth round-trip against a token this file invented — the only step in
    ``__init__`` a payload-shape test has to keep out of."""
    return AnthropicOAuthAdapter(
        model_name=model_name,
        max_tokens=8192,
        temperature=0.7,
        enable_thinking=enable,
        thinking_budget=budget,
        auto_refresh=False,
    )


def _api(model_name: str, *, max_tokens: int = 8192) -> AnthropicAdapter:
    """The api-key twin, for the parity arms: the two builders are separate code and
    only a test that drives BOTH can show they still agree."""
    return AnthropicAdapter(
        model_name=model_name,
        api_key="not-a-real-key",
        base_url="https://api.anthropic.com/v1",
        max_tokens=max_tokens,
    )


@pytest.mark.parametrize("model_name", ["claude-opus-5", "claude-sonnet-5"])
def test_claude5_gets_adaptive_thinking(model_name):
    payload = _oauth(model_name).format_request_payload(MESSAGES, thinking_budget=8192)
    assert payload["thinking"] == {"type": "adaptive"}


@pytest.mark.parametrize(
    "model_name", ["claude-opus-5", "claude-sonnet-5", "claude-opus-4-8", "claude-opus-4-7"]
)
def test_reasoning_models_never_get_temperature(model_name):
    """The regression that mattered: this leg used to always send temperature,
    so 4.7/4.8 were already 400ing before Claude 5 existed."""
    for budget in (8192, 0):
        payload = _oauth(model_name).format_request_payload(
            MESSAGES, thinking_budget=budget, temperature=0.7
        )
        assert "temperature" not in payload, (model_name, budget)


def test_legacy_model_keeps_budget_and_temperature():
    legacy = _oauth("claude-sonnet-4-6")
    thinking = legacy.format_request_payload(MESSAGES, thinking_budget=4096)
    assert thinking["thinking"] == {"type": "enabled", "budget_tokens": 4096}
    plain = legacy.format_request_payload(MESSAGES, thinking_budget=0, temperature=0.3)
    assert plain["temperature"] == 0.3
    assert "thinking" not in plain


def test_effort_rides_output_config_for_claude5():
    payload = _oauth("claude-opus-5").format_request_payload(
        MESSAGES, thinking_budget=8192, reasoning_effort="XHigh"
    )
    assert payload["output_config"]["effort"] == "xhigh"


def test_short_aliases_resolve_and_shape_as_claude5():
    """`opus`/`sonnet` now point at the Claude 5 generation, so alias callers
    must get the Claude 5 payload shape too."""
    for alias in ("opus", "sonnet"):
        payload = _oauth(alias).format_request_payload(
            MESSAGES, thinking_budget=8192, temperature=0.7
        )
        assert payload["thinking"] == {"type": "adaptive"}, alias
        assert "temperature" not in payload, alias


def test_fixed_budget_is_clamped_under_max_tokens():
    """Parity with the api-key twin: budget_tokens >= max_tokens is a live 400.
    The shape that mattered: a background model built at max_tokens=4096 with the
    default 8192 thinking budget — every call was an illegal payload on this leg."""
    adapter = _oauth("claude-haiku-4-5-20251001", budget=8192)
    adapter.max_tokens = 4096
    payload = adapter.format_request_payload(MESSAGES, thinking_budget=8192)
    assert payload["thinking"] == {"type": "enabled", "budget_tokens": 3072}


def test_budget_that_cannot_fit_disables_thinking():
    adapter = _oauth("claude-haiku-4-5-20251001", budget=8192)
    adapter.max_tokens = 1536  # headroom leaves less than the documented minimum budget
    payload = adapter.format_request_payload(MESSAGES, thinking_budget=8192)
    assert "thinking" not in payload


def test_thinking_flag_without_budget_sends_no_null_budget():
    """enable_thinking with no usable budget used to put budget_tokens=None/0 on the
    wire; thinking is dropped instead of sending an illegal payload."""
    adapter = _oauth("claude-haiku-4-5-20251001", enable=True, budget=0)
    payload = adapter.format_request_payload(MESSAGES)
    assert "thinking" not in payload


# --- structured output ------------------------------------------------------


def test_schema_rides_output_config_natively():
    """This endpoint IS the first-party Messages API, so the schema goes on the wire
    rather than into the prompt."""
    payload = _oauth("claude-opus-5").format_request_payload(
        MESSAGES, thinking_budget=8192, response_schema=SCHEMA
    )
    assert payload["output_config"]["format"]["type"] == "json_schema"
    assert payload["output_config"]["format"]["schema"]["additionalProperties"] is False
    assert payload["output_config"]["format"]["schema"]["required"] == ["units"]
    # …and the request text is untouched: no prompt fallback rides along with it.
    assert "JSON Schema" not in str(payload["messages"][-1]["content"])


def test_effort_and_schema_share_one_output_config():
    """The regression this leg shipped: `output_config` was ASSIGNED here, so a request
    carrying both a reasoning effort and a schema lost the effort silently. The API
    takes exactly one such object per request — merge, never assign."""
    payload = _oauth("claude-opus-5").format_request_payload(
        MESSAGES, thinking_budget=8192, reasoning_effort="low", response_schema=SCHEMA
    )
    assert payload["output_config"]["effort"] == "low"
    assert payload["output_config"]["format"]["type"] == "json_schema"


def test_a_leg_without_native_enforcement_puts_the_whole_schema_in_the_prompt(monkeypatch):
    """The house fallback (Bedrock's convention): an endpoint that cannot enforce
    schemas must not put the illegal field on the wire, and must not degrade the ask to
    a bare "please emit JSON" — the caller's parser would only be satisfied by luck."""
    adapter = _oauth("claude-opus-5")
    monkeypatch.setattr(adapter, "supports_structured_output", False, raising=False)
    payload = adapter.format_request_payload(
        MESSAGES, thinking_budget=8192, reasoning_effort="low", response_schema=SCHEMA
    )
    assert "format" not in payload.get("output_config", {})
    assert payload["output_config"]["effort"] == "low"  # the effort still survives
    text = str(payload["messages"][-1]["content"])
    assert "JSON Schema" in text
    assert '"properties"' in text and '"units"' in text


# --- parity between the two builders ----------------------------------------


@pytest.mark.parametrize("model_name", ["claude-opus-5", "claude-sonnet-4-6"])
@pytest.mark.parametrize("native", [True, False])
def test_both_anthropic_builders_emit_the_same_structured_output_shape(model_name, native, monkeypatch):
    """The two payload builders are separate code kept deliberately parallel, and this
    is the branch where the parallelism drifted twice. Identical inputs must produce an
    identical structured-output shape — native config and prompt fallback alike."""
    oauth, api = _oauth(model_name), _api(model_name)
    for adapter in (oauth, api):
        monkeypatch.setattr(adapter, "supports_structured_output", native, raising=False)
    kwargs = dict(thinking_budget=8192, reasoning_effort="low", response_schema=SCHEMA)
    oauth_payload = oauth.format_request_payload([dict(m) for m in MESSAGES], **kwargs)
    api_payload = api.format_request_payload([dict(m) for m in MESSAGES], **kwargs)

    assert oauth_payload.get("output_config") == api_payload.get("output_config")
    assert oauth_payload["messages"][-1]["content"] == api_payload["messages"][-1]["content"]


def test_both_builders_clamp_a_fixed_budget_to_the_same_number():
    """The repair path's landmine: a background model built at max_tokens=4096 with the
    default 8192 budget. The api-key leg clamped and the OAuth leg did not, so the same
    settings were legal on one leg and a 400 on the other."""
    oauth = _oauth("claude-haiku-4-5-20251001", budget=8192)
    oauth.max_tokens = 4096
    oauth_payload = oauth.format_request_payload(MESSAGES, thinking_budget=8192)
    api_payload = _api("claude-haiku-4-5-20251001", max_tokens=4096).format_request_payload(
        MESSAGES, thinking_budget=8192
    )
    assert oauth_payload["thinking"] == api_payload["thinking"]
    assert oauth_payload["thinking"] == {"type": "enabled", "budget_tokens": 3072}
