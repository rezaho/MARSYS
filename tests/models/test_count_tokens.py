"""Provider-counted input tokens on the two Anthropic legs.

No network. What is pinned here:

* **The counted body is the body that would be SENT.** Each leg builds the count
  payload through its own ``format_request_payload``, so the OAuth leg's required
  Claude-Code system block and its tool renaming are counted too. A count assembled
  any other way measures a request nobody makes.
* **Only the generation controls come off.** ``max_tokens``/``stream``/sampling are
  rejected by the count endpoint; ``system``/``messages``/``tools``/``thinking`` are
  exactly what decides the number.
* **A failure is ``None``, never an exception.** The caller is sizing something, not
  producing an answer. A missing route (404/405) is remembered for the process; a
  credential-shaped refusal is not — the OAuth token file has several writers and a
  refresh in flight looks exactly like a rejection for one request.
"""

import json

import httpx
import pytest

from marsys.models.adapters import anthropic as anthropic_mod
from marsys.models.adapters.anthropic import (
    AsyncAnthropicAdapter,
    count_tokens_url_for,
    strip_for_count_tokens,
)
from marsys.models.adapters.anthropic_oauth import AsyncAnthropicOAuthAdapter
from marsys.models.models import BaseAPIModel

MESSAGES = [{"role": "user", "content": "hi"}]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "read a file",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


@pytest.fixture(autouse=True)
def _forget_unsupported_endpoints():
    """The unavailable-endpoint memo is process-wide by design; tests must not
    inherit each other's."""
    anthropic_mod._COUNT_TOKENS_UNSUPPORTED.clear()
    yield
    anthropic_mod._COUNT_TOKENS_UNSUPPORTED.clear()


class _FakeResponse:
    def __init__(self, status: int, body):
        self.status = status
        self._body = body

    async def json(self, content_type=None):
        if self._body is None:
            raise ValueError("not json")
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


_DEFAULT_BODY = object()  # so ``body=None`` can mean "the response is not JSON"


class _FakeSession:
    """The aiohttp seam ``AsyncBaseAPIAdapter._ensure_session`` hands out."""

    # ``_ensure_session`` reuses a live session and rebuilds a closed one.
    closed = False

    def __init__(self, status: int = 200, body=_DEFAULT_BODY, raises=None):
        self.status = status
        self.body = {"input_tokens": 1234} if body is _DEFAULT_BODY else body
        self.raises = raises
        self.calls: list[dict] = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "payload": json})
        if self.raises is not None:
            raise self.raises
        return _FakeResponse(self.status, self.body)


def _api(session: _FakeSession) -> AsyncAnthropicAdapter:
    adapter = AsyncAnthropicAdapter(
        model_name="claude-opus-5",
        api_key="not-a-real-key",
        base_url="https://api.anthropic.com/v1",
        max_tokens=8192,
    )
    adapter._session = session
    return adapter


def _oauth() -> AsyncAnthropicOAuthAdapter:
    """An OAuth adapter that never touches the credentials file on disk."""
    adapter = object.__new__(AsyncAnthropicOAuthAdapter)
    adapter.model_name = "claude-sonnet-5"
    adapter.max_tokens = 8192
    adapter.temperature = 0.7
    adapter.enable_thinking = False
    adapter.thinking_budget = 0
    adapter.auto_refresh = False
    adapter.access_token = "not-a-real-token"
    adapter.credentials = {"access_token": "not-a-real-token"}
    adapter._credentials_path = "unused"
    adapter._session = None
    return adapter


def _oauth_transport(monkeypatch, handler):
    real_async_client = httpx.AsyncClient
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: real_async_client(transport=transport)
    )


# ── the URL and the payload ───────────────────────────────────────────────────


def test_the_count_url_sits_beside_the_messages_url_and_keeps_its_query():
    assert (
        count_tokens_url_for("https://api.anthropic.com/v1/messages")
        == "https://api.anthropic.com/v1/messages/count_tokens"
    )
    # The OAuth leg's endpoint carries a query string; dropping it would send the
    # count somewhere the messages call never goes.
    assert (
        count_tokens_url_for("https://api.anthropic.com/v1/messages?beta=true")
        == "https://api.anthropic.com/v1/messages/count_tokens?beta=true"
    )


def test_the_final_assistant_message_is_trimmed_of_trailing_whitespace():
    """The API rejects a final assistant message that ends in whitespace, and a count
    is the one request that routinely presents one: a settled conversation ends with
    an assistant reply, and models end replies with a newline. Measured live — the
    provider answers "final assistant content cannot end with trailing whitespace"."""
    payload = {
        "model": "m",
        "messages": [
            {"role": "user", "content": "ask"},
            {"role": "assistant", "content": [{"type": "text", "text": "answered.\n\n"}]},
        ],
    }
    stripped = strip_for_count_tokens(payload)
    assert stripped["messages"][-1]["content"][-1]["text"] == "answered."
    # …and the caller's own rows are untouched: the payload's blocks may be the
    # durable conversation's dicts.
    assert payload["messages"][-1]["content"][-1]["text"] == "answered.\n\n"


def test_a_plain_string_final_assistant_message_is_trimmed_too():
    stripped = strip_for_count_tokens(
        {"model": "m", "messages": [{"role": "assistant", "content": "done  "}]}
    )
    assert stripped["messages"][-1]["content"] == "done"


def test_a_whitespace_only_final_assistant_message_is_dropped():
    """Nothing is left of it to count, and an empty text block is itself rejected."""
    stripped = strip_for_count_tokens(
        {
            "model": "m",
            "messages": [
                {"role": "user", "content": "ask"},
                {"role": "assistant", "content": [{"type": "text", "text": "   "}]},
            ],
        }
    )
    assert stripped["messages"] == [{"role": "user", "content": "ask"}]


def test_a_final_user_message_is_left_exactly_as_it_is():
    """The rule is about assistant content. Trimming a user row would change what is
    being counted for no reason."""
    stripped = strip_for_count_tokens(
        {"model": "m", "messages": [{"role": "user", "content": "ask \n"}]}
    )
    assert stripped["messages"] == [{"role": "user", "content": "ask \n"}]


def test_only_the_generation_controls_are_stripped():
    payload = {
        "model": "claude-opus-5",
        "max_tokens": 8192,
        "stream": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "output_config": {"effort": "high"},
        "system": [{"type": "text", "text": "SYS"}],
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"name": "t", "input_schema": {}}],
        "thinking": {"type": "adaptive"},
    }
    stripped = strip_for_count_tokens(payload)
    assert set(stripped) == {"model", "system", "messages", "tools", "thinking"}
    # the survivors are untouched, not rebuilt
    assert stripped["system"] == payload["system"]
    assert stripped["tools"] == payload["tools"]


# ── the api-key leg ───────────────────────────────────────────────────────────


async def test_the_api_key_leg_counts_the_body_it_would_send():
    session = _FakeSession(body={"input_tokens": 4_242})
    adapter = _api(session)

    count = await adapter.acount_tokens(MESSAGES, tools=TOOLS, system="SYS")

    assert count == 4_242
    call = session.calls[0]
    assert call["url"] == "https://api.anthropic.com/v1/messages/count_tokens"
    # The system string rides as the payload builder's own rendered system field —
    # the block shape the messages call sends, not the raw string.
    assert call["payload"]["system"][0]["text"] == "SYS"
    # …the tools are the converted Anthropic shape the request would carry…
    assert call["payload"]["tools"][0]["name"] == "read_file"
    assert "input_schema" in call["payload"]["tools"][0]
    # …and the generation controls are gone.
    assert "max_tokens" not in call["payload"] and "stream" not in call["payload"]
    assert call["headers"]["accept"] == "application/json"
    assert call["headers"]["x-api-key"] == "not-a-real-key"


async def test_a_body_with_no_system_is_counted_as_is():
    session = _FakeSession(body={"input_tokens": 11})
    adapter = _api(session)
    assert await adapter.acount_tokens(MESSAGES) == 11
    assert session.calls[0]["payload"]["messages"][0]["role"] == "user"


async def test_a_transport_failure_answers_none_rather_than_raising():
    import aiohttp

    session = _FakeSession(raises=aiohttp.ClientError("connection reset"))
    adapter = _api(session)
    assert await adapter.acount_tokens(MESSAGES) is None


async def test_a_body_that_is_not_json_answers_none():
    session = _FakeSession(status=200, body=None)
    adapter = _api(session)
    assert await adapter.acount_tokens(MESSAGES) is None


async def test_a_200_without_input_tokens_answers_none():
    session = _FakeSession(status=200, body={"unexpected": True})
    adapter = _api(session)
    assert await adapter.acount_tokens(MESSAGES) is None


# ── availability: structural vs credential-shaped ─────────────────────────────


@pytest.mark.parametrize("status", [404, 405])
async def test_a_missing_route_is_remembered_for_the_process(status):
    """The endpoint is not there; asking again every turn buys nothing."""
    session = _FakeSession(status=status, body={"error": "not found"})
    adapter = _api(session)

    assert await adapter.acount_tokens(MESSAGES) is None
    assert await adapter.acount_tokens(MESSAGES) is None
    assert len(session.calls) == 1  # the second call never left the process


@pytest.mark.parametrize("status", [401, 403, 429, 500])
async def test_a_credential_or_transient_refusal_is_never_remembered(status):
    """The OAuth token file has several writers, so a refresh in flight looks like a
    rejection for one request. Memoising that would disable counting for the life of
    a daemon that never restarts."""
    session = _FakeSession(status=status, body={"error": "nope"})
    adapter = _api(session)

    assert await adapter.acount_tokens(MESSAGES) is None
    assert await adapter.acount_tokens(MESSAGES) is None
    assert len(session.calls) == 2  # asked again, as it must be


async def test_the_memo_is_per_endpoint_not_global():
    unavailable = _api(_FakeSession(status=404, body={}))
    assert await unavailable.acount_tokens(MESSAGES) is None

    other_session = _FakeSession(body={"input_tokens": 7})
    other = AsyncAnthropicAdapter(
        model_name="claude-opus-5",
        api_key="k",
        base_url="https://other.example.invalid/v1",
        max_tokens=1024,
    )
    other._session = other_session
    assert await other.acount_tokens(MESSAGES) == 7


# ── the OAuth leg ─────────────────────────────────────────────────────────────


async def test_the_oauth_leg_counts_what_its_own_payload_builder_renders(monkeypatch):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        seen["payload"] = json.loads(request.content)
        return httpx.Response(200, json={"input_tokens": 9_001})

    _oauth_transport(monkeypatch, handler)
    adapter = _oauth()
    monkeypatch.setattr(adapter, "_ensure_fresh_token", lambda: None, raising=False)

    count = await adapter.acount_tokens(MESSAGES, tools=TOOLS, system="SYS")

    assert count == 9_001
    assert seen["url"] == "https://api.anthropic.com/v1/messages/count_tokens?beta=true"
    # The required Claude-Code prefix block is COUNTED, because the messages call
    # always sends it — a count without it undercounts every request.
    assert seen["payload"]["system"][0]["text"] == adapter.CLAUDE_CODE_PREFIX
    assert seen["payload"]["system"][1]["text"] == "SYS"
    # …as is the reserved-name transform this leg applies to tools.
    assert seen["payload"]["tools"][0]["name"] == "Read"
    assert "max_tokens" not in seen["payload"] and "stream" not in seen["payload"]
    # JSON, not the messages call's SSE.
    assert seen["headers"]["accept"] == "application/json"
    assert seen["headers"]["authorization"].startswith("Bearer ")
    assert "oauth-2025-04-20" in seen["headers"]["anthropic-beta"]


async def test_the_oauth_leg_refreshes_its_token_before_counting(monkeypatch):
    """Same discipline as the messages call: the cached token is a per-request
    snapshot of a file other processes rewrite."""
    refreshed: list[bool] = []
    _oauth_transport(
        monkeypatch, lambda request: httpx.Response(200, json={"input_tokens": 5})
    )
    adapter = _oauth()
    monkeypatch.setattr(
        adapter, "_ensure_fresh_token", lambda: refreshed.append(True), raising=False
    )

    assert await adapter.acount_tokens(MESSAGES) == 5
    assert refreshed == [True]


async def test_an_unauthorised_oauth_leg_answers_none_and_is_asked_again(monkeypatch):
    """If the subscription credential is not accepted here, compaction reports "not
    measured" and the turn is untouched — and the next turn asks again. This leg's
    token file has several writers, so a refusal is as likely to be a refresh in
    flight as a verdict, and a daemon that never restarts would otherwise stop
    measuring for days on one blip."""
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(403, json={"error": {"type": "forbidden"}})

    _oauth_transport(monkeypatch, handler)
    adapter = _oauth()
    monkeypatch.setattr(adapter, "_ensure_fresh_token", lambda: None, raising=False)

    assert await adapter.acount_tokens(MESSAGES) is None
    assert await adapter.acount_tokens(MESSAGES) is None
    assert len(calls) == 2


async def test_an_oauth_transport_failure_answers_none(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    _oauth_transport(monkeypatch, handler)
    adapter = _oauth()
    monkeypatch.setattr(adapter, "_ensure_fresh_token", lambda: None, raising=False)

    assert await adapter.acount_tokens(MESSAGES) is None


# ── the model port ────────────────────────────────────────────────────────────


class _AdapterWithCount:
    def __init__(self):
        self.seen: dict = {}

    async def acount_tokens(self, messages, *, tools=None, system=None):
        self.seen = {"messages": messages, "tools": tools, "system": system}
        return 321


async def test_the_model_delegates_the_count_to_its_async_adapter():
    model = object.__new__(BaseAPIModel)
    adapter = _AdapterWithCount()
    model.async_adapter = adapter

    assert await model.acount_tokens(MESSAGES, tools=TOOLS, system="SYS") == 321
    assert adapter.seen == {"messages": MESSAGES, "tools": TOOLS, "system": "SYS"}


async def test_a_provider_that_cannot_count_answers_none_rather_than_guessing():
    """Every non-Anthropic leg takes this path. The caller must be able to tell an
    absent count from a small one."""
    model = object.__new__(BaseAPIModel)
    model.async_adapter = object()
    assert await model.acount_tokens(MESSAGES) is None

    model.async_adapter = None
    assert await model.acount_tokens(MESSAGES) is None
