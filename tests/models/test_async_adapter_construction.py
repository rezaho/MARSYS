"""Every model gets its provider's async adapter, and ``arun`` never passes through ``run``.

A model's async adapter is chosen by the provider, the same way its sync adapter is. It must not
depend on where the model's class happens to be defined: a subclass written in an application's
own module gets exactly the async adapter a plain ``BaseAPIModel`` gets.

``arun`` and ``run`` are two independent entry points. An async call goes through the async
adapter and never through ``run``, so a subclass that wraps ``run`` sees only the sync calls.

This module deliberately imports no async adapter class by name: the subclasses below are
defined here, and the async adapter they get must not be one this module put in scope. The
expected classes are read off the adapters package by name instead.

No network: the one test that completes a call answers it from a server on the loopback
interface. No credential on the machine is read: every test runs with the home directory, the
CLI login paths and the profile store moved under its own temporary folder.
"""

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import marsys.models.adapters as adapters_package
from marsys.models.adapters.anthropic_oauth import AnthropicOAuthAdapter
from marsys.models.adapters.factory import ProviderAdapterFactory
from marsys.models.adapters.openai_oauth import OpenAIOAuthAdapter
from marsys.models.credentials import OAuthCredentialStore
from marsys.models.models import BaseAPIModel

# Spelled out here rather than read from the factory, so a wrong row in the factory's table is
# a failing test instead of the test agreeing with it.
ASYNC_ADAPTER_BY_PROVIDER = {
    "openai": "AsyncOpenAIAdapter",
    "anthropic": "AsyncAnthropicAdapter",
    "bedrock": "AsyncBedrockAdapter",
    "azure": "AsyncAzureOpenAIAdapter",
    "google": "AsyncGoogleAdapter",
    "openrouter": "AsyncOpenRouterAdapter",
    "xai": "AsyncOpenRouterAdapter",
    "openai-oauth": "AsyncOpenAIOAuthAdapter",
    "anthropic-oauth": "AsyncAnthropicOAuthAdapter",
}
OAUTH_PROVIDERS = ("openai-oauth", "anthropic-oauth")
STREAMING_OPT_IN_PROVIDERS = ("anthropic", "azure", "bedrock", "openai")
FAKE_CREDENTIALS_PATH = "unused-fake-credentials"


class ApplicationModel(BaseAPIModel):
    """A subclass as an application would write one, outside ``marsys.models.models``."""


class RunCountingModel(BaseAPIModel):
    """A subclass that wraps ``run`` and counts the calls that reach it."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_calls = 0

    def run(self, *args, **kwargs):
        self.run_calls += 1
        return super().run(*args, **kwargs)


#: The real login reads, kept so the fence test below can prove where they look.
_READ_CLAUDE_LOGIN = AnthropicOAuthAdapter._load_claude_credentials
_READ_CODEX_LOGIN = OpenAIOAuthAdapter._load_codex_credentials


@pytest.fixture(autouse=True)
def fenced_home(monkeypatch, tmp_path):
    """No credential file on the machine is opened by any test here.

    The home directory, both CLI login paths and the profile store point under the test's own
    temporary folder, and the profile store is re-read from there. The OAuth adapters' login
    read also answers from memory, so the placeholder path most tests pass is never opened."""
    home = tmp_path / "home"
    home.mkdir()
    for name in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(name, str(home))
    monkeypatch.setenv("CLAUDE_AUTH_PATH", str(home / ".claude" / ".credentials.json"))
    monkeypatch.setenv("CODEX_AUTH_PATH", str(home / ".codex" / "auth.json"))
    monkeypatch.setenv("MARSYS_CREDENTIALS_PATH", str(home / ".marsys" / "credentials.json"))
    monkeypatch.setattr(OAuthCredentialStore, "_instance", None)
    monkeypatch.setattr(
        AnthropicOAuthAdapter, "_load_claude_credentials",
        lambda self, path=None: {"access_token": "fake-claude-token"},
    )
    monkeypatch.setattr(
        OpenAIOAuthAdapter, "_load_codex_credentials",
        lambda self, path=None: {"access_token": "fake-codex-token", "account_id": "fake-account"},
    )
    return home


def _write_logins(home):
    """A CLI login of each kind under ``home``, far from expiry and with no refresh token, so
    nothing tries to refresh it."""
    claude = home / ".claude" / ".credentials.json"
    codex = home / ".codex" / "auth.json"
    claude.parent.mkdir(parents=True)
    codex.parent.mkdir(parents=True)
    far_future_ms = int((time.time() + 365 * 86400) * 1000)
    claude.write_text(json.dumps({"claudeAiOauth": {
        "accessToken": "fenced-claude-token", "expiresAt": far_future_ms,
    }}), encoding="utf-8")
    codex.write_text(json.dumps({"tokens": {
        "access_token": "fenced-codex-token", "account_id": "fenced-account",
    }}), encoding="utf-8")


def test_every_credential_location_is_under_the_tests_own_home(fenced_home):
    """The fence above, proved location by location: the home directory, each CLI login the
    OAuth adapters read when given no path, and the profile store with the logins it discovers.
    The home directory is checked first and each login path before its read, so a fence that
    failed to move one fails here before a real file is opened."""
    home = fenced_home
    assert Path.home() == home
    _write_logins(home)

    claude = AnthropicOAuthAdapter("claude-haiku-4-5", auto_refresh=False)
    codex = OpenAIOAuthAdapter("gpt-5.5", auto_refresh=False)
    assert Path(claude._credentials_path).is_relative_to(home)
    assert Path(codex._credentials_path).is_relative_to(home)
    assert _READ_CLAUDE_LOGIN(claude)["access_token"] == "fenced-claude-token"
    assert _READ_CODEX_LOGIN(codex)["access_token"] == "fenced-codex-token"

    store = OAuthCredentialStore.get_instance()
    assert store._store_path.is_relative_to(home)
    discovered = store.list_profiles()
    assert {profile.provider for profile in discovered} == set(OAUTH_PROVIDERS)
    assert all(profile.resolved_path.is_relative_to(home) for profile in discovered)


def _config(provider, **overrides):
    config = dict(
        model_name="claude-haiku-4-5" if "anthropic" in provider or provider == "bedrock" else "gpt-5.5",
        api_key="not-a-real-key",
        base_url="https://example.invalid/v1",
        provider=provider,
        max_tokens=5000,
        temperature=0.3,
        top_p=0.9,
        thinking_budget=1234,
        reasoning_effort="low",
    )
    if provider in OAUTH_PROVIDERS:
        config.update(credentials_path=FAKE_CREDENTIALS_PATH, auto_refresh=False)
    config.update(overrides)
    return config


def _expected_async_class(provider):
    return getattr(adapters_package, ASYNC_ADAPTER_BY_PROVIDER[provider])


def test_this_module_puts_no_async_adapter_class_in_scope():
    """The premise the subclass tests rest on: nothing named like an async adapter is
    reachable in the module that defines them."""
    module = sys.modules[ApplicationModel.__module__]
    assert module.__name__ != "marsys.models.models"
    assert not [name for name in vars(module) if name.startswith("Async")]


def test_the_expected_table_covers_every_provider_the_factory_maps():
    assert set(ProviderAdapterFactory.ADAPTERS) == set(ASYNC_ADAPTER_BY_PROVIDER)


@pytest.mark.parametrize("provider", sorted(ASYNC_ADAPTER_BY_PROVIDER))
def test_a_model_gets_its_providers_async_adapter(provider):
    model = BaseAPIModel(**_config(provider))

    assert model.async_adapter is not None
    assert type(model.async_adapter) is _expected_async_class(provider)


@pytest.mark.parametrize("provider", sorted(ASYNC_ADAPTER_BY_PROVIDER))
def test_a_subclass_defined_elsewhere_gets_the_same_async_adapter(provider):
    plain = BaseAPIModel(**_config(provider))
    subclassed = ApplicationModel(**_config(provider))

    assert subclassed.async_adapter is not None
    assert type(subclassed.async_adapter) is type(plain.async_adapter)
    assert type(subclassed.async_adapter) is _expected_async_class(provider)


def test_an_unmapped_provider_gets_the_default_adapters_async_twin():
    """An OpenAI-compatible endpoint the factory does not name is served by the default
    adapter, and it gets that adapter's async twin, so no factory-built model is without one."""
    model = ApplicationModel(**_config("some-openai-compatible-endpoint"))

    assert type(model.adapter) is adapters_package.OpenAIAdapter
    assert type(model.async_adapter) is adapters_package.AsyncOpenAIAdapter
    assert model.async_adapter.provider == "some-openai-compatible-endpoint"


# --- the async adapter is configured from the same config as the sync one -----------------


def _configuration(adapter):
    """What an adapter was configured with: its instance attributes, minus the connection
    state only the async twin carries."""
    return {k: v for k, v in vars(adapter).items() if k != "_session"}


@pytest.mark.parametrize("provider", sorted(ASYNC_ADAPTER_BY_PROVIDER))
def test_the_async_adapter_is_configured_like_the_sync_adapter(provider):
    model = ApplicationModel(**_config(provider))
    sync, async_ = _configuration(model.adapter), _configuration(model.async_adapter)

    # Every setting the sync adapter holds, the async adapter holds with the same value, and
    # the only thing the async adapter adds is an opt-in the sync adapter does not read.
    assert {k: async_.get(k, "<missing>") for k in sync} == sync
    assert set(async_) - set(sync) <= {"streaming"}
    assert model.async_adapter.provider == model.adapter.provider == provider
    assert model.async_adapter.model_name == model.adapter.model_name


@pytest.mark.parametrize("provider", ["anthropic", "openai", "openrouter", "google", "azure", "bedrock"])
def test_the_key_endpoint_and_sampling_reach_the_async_adapter(provider):
    model = ApplicationModel(**_config(provider))
    async_ = model.async_adapter

    assert async_.api_key == model.adapter.api_key == "not-a-real-key"
    assert async_.base_url == model.adapter.base_url
    assert "example.invalid" in async_.base_url
    assert async_.max_tokens == 5000
    assert async_.temperature == 0.3


@pytest.mark.parametrize("provider", OAUTH_PROVIDERS)
def test_an_oauth_async_adapter_reads_the_credentials_its_sync_twin_reads(provider):
    model = ApplicationModel(**_config(provider))

    assert model.async_adapter._credentials_path == model.adapter._credentials_path
    assert model.async_adapter._credentials_path == FAKE_CREDENTIALS_PATH
    assert model.async_adapter.access_token == model.adapter.access_token
    assert model.async_adapter.auto_refresh is model.adapter.auto_refresh is False


@pytest.mark.parametrize("provider", STREAMING_OPT_IN_PROVIDERS)
def test_the_streaming_opt_in_reaches_the_async_adapter(provider):
    streamed = ApplicationModel(**_config(provider, streaming=True))
    unstreamed = ApplicationModel(**_config(provider))

    assert streamed.async_adapter.streaming is True
    assert unstreamed.async_adapter.streaming is False


# --- arun never passes through run ------------------------------------------------------


_ANTHROPIC_REPLY = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-4-5",
    "content": [{"type": "text", "text": "pong"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 7, "output_tokens": 2},
}


@pytest.fixture
def loopback_messages_api():
    """A Messages API stand-in on the loopback interface. It answers every POST with one
    reply in the Messages API's shape and counts what it was sent."""
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            received.append(json.loads(self.rfile.read(length) or b"{}"))
            body = json.dumps(_ANTHROPIC_REPLY).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1", received
    finally:
        server.shutdown()
        server.server_close()


async def test_arun_never_calls_a_subclasss_run(loopback_messages_api):
    base_url, received = loopback_messages_api
    model = RunCountingModel(
        model_name="claude-haiku-4-5", api_key="not-a-real-key", base_url=base_url,
        provider="anthropic", max_tokens=64,
    )
    try:
        response = await model.arun(messages=[{"role": "user", "content": "ping"}])
    finally:
        await model.cleanup()

    assert response.content == "pong"
    assert len(received) == 1
    assert model.run_calls == 0


def test_run_still_reaches_a_subclasss_run(loopback_messages_api):
    """The control: the sync entry point is untouched, so the wrapper above does count
    the calls that enter through ``run``."""
    base_url, received = loopback_messages_api
    model = RunCountingModel(
        model_name="claude-haiku-4-5", api_key="not-a-real-key", base_url=base_url,
        provider="anthropic", max_tokens=64,
    )

    response = model.run(messages=[{"role": "user", "content": "ping"}])

    assert response.content == "pong"
    assert len(received) == 1
    assert model.run_calls == 1
