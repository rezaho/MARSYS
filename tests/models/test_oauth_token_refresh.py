"""OAuth adapters refresh + reload the access token at request time.

Regression for three defects that made a long-lived daemon 401 once its token
expired:

1. The token was refreshed only at ``__init__``; nothing on the request path
   refreshed it (``_ensure_fresh_token`` must run per request, on the sync AND
   async streaming paths).
2. The credentials file is MULTI-WRITER (sibling adapter instances in the same
   process, the claude/codex CLI on the same machine). A reload gated on a
   self-performed refresh left the cached ``self.access_token`` stale forever
   once any OTHER writer refreshed the file — ``refresh_if_needed``
   short-circuits on the FILE's freshness, never this cache's. The reload is
   therefore unconditional: the file is the source of truth, the attribute a
   per-request snapshot.
3. ``OAuthTokenRefresher._update_credential_file`` used a POSIX-only
   ``fcntl`` lock, so on Windows every refresh write raised ``ImportError``
   (swallowed -> refresh silently never happened). It must write
   cross-platform (this test file runs on win32, exercising that path).
"""
import json
from pathlib import Path
from unittest.mock import patch

from marsys.models.adapters.anthropic_oauth import (
    AnthropicOAuthAdapter,
    AsyncAnthropicOAuthAdapter,
)
from marsys.models.adapters.openai_oauth import (
    AsyncOpenAIOAuthAdapter,
    OpenAIOAuthAdapter,
)
from marsys.models.credentials import OAuthTokenRefresher


def _anthropic(token: str) -> AnthropicOAuthAdapter:
    with patch.object(AnthropicOAuthAdapter, "_refresh_token_if_needed", lambda self: None), patch.object(
        AnthropicOAuthAdapter, "_load_claude_credentials", lambda self, p=None: {"access_token": token}
    ):
        return AnthropicOAuthAdapter(model_name="claude-haiku-4-5-20251001", auto_refresh=True)


def _openai(token: str) -> OpenAIOAuthAdapter:
    with patch.object(OpenAIOAuthAdapter, "_refresh_token_if_needed", lambda self: None), patch.object(
        OpenAIOAuthAdapter,
        "_load_codex_credentials",
        lambda self, p=None: {"access_token": token, "account_id": "acct"},
    ):
        return OpenAIOAuthAdapter(model_name="gpt-5.4-mini", auto_refresh=True)


def test_ensure_fresh_token_reloads_access_token_after_refresh():
    """A refresh rewrote the credentials file; the cached token must be reloaded
    (otherwise get_headers keeps sending the stale token and 401s)."""
    adapter = _anthropic("old-token")
    assert adapter.access_token == "old-token"
    with patch.object(AnthropicOAuthAdapter, "_refresh_token_if_needed", lambda self: True), patch.object(
        AnthropicOAuthAdapter, "_load_claude_credentials", lambda self, p=None: {"access_token": "new-token"}
    ):
        adapter._ensure_fresh_token()
    assert adapter.access_token == "new-token"


def test_ensure_fresh_token_reloads_even_without_a_self_refresh():
    """THE SECOND-WRITER REGRESSION (deliberate inversion of the former
    "noop when still valid" test, which pinned the bug): refresh returns False
    because the FILE is fresh — but another writer (sibling adapter, the CLI)
    made it so, and this instance's cache is stale. The reload must happen
    anyway; the file is the source of truth."""
    adapter = _anthropic("stale-boot-token")
    with patch.object(AnthropicOAuthAdapter, "_refresh_token_if_needed", lambda self: False), patch.object(
        AnthropicOAuthAdapter,
        "_load_claude_credentials",
        lambda self, p=None: {"access_token": "rotated-by-another-writer"},
    ):
        adapter._ensure_fresh_token()
    assert adapter.access_token == "rotated-by-another-writer"


def test_ensure_fresh_token_keeps_prior_cache_when_reload_fails():
    """Expired file + failed refresh: the reload raises, the prior cache is
    kept, and the request carries the provider's own error — no valid token
    exists in that state, so there is nothing better to send."""
    def _boom(self, p=None):
        raise RuntimeError("file expired and refresh failed")

    adapter = _anthropic("last-known-token")
    with patch.object(AnthropicOAuthAdapter, "_refresh_token_if_needed", lambda self: False), patch.object(
        AnthropicOAuthAdapter, "_load_claude_credentials", _boom
    ):
        adapter._ensure_fresh_token()  # must not raise
    assert adapter.access_token == "last-known-token"


def test_openai_ensure_fresh_token_reloads_even_without_a_self_refresh():
    """The OpenAI twin of the second-writer regression — including account_id,
    which get_headers also sends."""
    adapter = _openai("stale-boot-token")
    with patch.object(OpenAIOAuthAdapter, "_refresh_token_if_needed", lambda self: False), patch.object(
        OpenAIOAuthAdapter,
        "_load_codex_credentials",
        lambda self, p=None: {"access_token": "rotated", "account_id": "acct-2"},
    ):
        adapter._ensure_fresh_token()
    assert adapter.access_token == "rotated"
    assert adapter.account_id == "acct-2"


def test_ensure_fresh_token_disabled_when_auto_refresh_false():
    adapter = _anthropic("tok")
    adapter.auto_refresh = False
    with patch.object(AnthropicOAuthAdapter, "_refresh_token_if_needed", lambda self: True), patch.object(
        AnthropicOAuthAdapter, "_load_claude_credentials", lambda self, p=None: {"access_token": "new"}
    ):
        adapter._ensure_fresh_token()
    assert adapter.access_token == "tok"


def test_openai_ensure_fresh_token_reloads_after_refresh():
    adapter = _openai("old")
    with patch.object(OpenAIOAuthAdapter, "_refresh_token_if_needed", lambda self: True), patch.object(
        OpenAIOAuthAdapter,
        "_load_codex_credentials",
        lambda self, p=None: {"access_token": "new", "account_id": "acct"},
    ):
        adapter._ensure_fresh_token()
    assert adapter.access_token == "new"


async def test_anthropic_async_request_path_refreshes_token_per_request():
    """The named regression was INVOCATION, not mechanics: a long-lived
    process 401s because nothing on the request path refreshes the token.
    arun_streaming must call _ensure_fresh_token before building headers."""
    with patch.object(AnthropicOAuthAdapter, "_refresh_token_if_needed", lambda self: None), patch.object(
        AnthropicOAuthAdapter, "_load_claude_credentials", lambda self, p=None: {"access_token": "tok"}
    ):
        adapter = AsyncAnthropicOAuthAdapter(model_name="claude-haiku-4-5-20251001", auto_refresh=True)

    calls = []

    def spy():
        calls.append(1)
        raise RuntimeError("short-circuit before any network I/O")

    adapter._ensure_fresh_token = spy
    try:
        await adapter.arun_streaming([{"role": "user", "content": "hi"}])
    except Exception:
        pass  # the adapter may wrap or re-raise; only the invocation matters
    assert calls, "arun_streaming never invoked _ensure_fresh_token"


async def test_openai_async_request_path_refreshes_token_per_request():
    with patch.object(OpenAIOAuthAdapter, "_refresh_token_if_needed", lambda self: None), patch.object(
        OpenAIOAuthAdapter,
        "_load_codex_credentials",
        lambda self, p=None: {"access_token": "tok", "account_id": "acct"},
    ):
        adapter = AsyncOpenAIOAuthAdapter(model_name="gpt-5.4-mini", auto_refresh=True)

    calls = []

    def spy():
        calls.append(1)
        raise RuntimeError("short-circuit before any network I/O")

    adapter._ensure_fresh_token = spy
    try:
        await adapter.arun_streaming([{"role": "user", "content": "hi"}])
    except Exception:
        pass
    assert calls, "arun_streaming never invoked _ensure_fresh_token"


def test_anthropic_sync_request_path_refreshes_token_per_request():
    """The sync streaming path is live in production (StructuredLLM /
    consolidation route through BaseAPIModel.run -> run_streaming) and never
    refreshed — it 401'd unconditionally once the boot token expired."""
    adapter = _anthropic("tok")

    calls = []

    def spy():
        calls.append(1)
        raise RuntimeError("short-circuit before any network I/O")

    adapter._ensure_fresh_token = spy
    try:
        adapter.run_streaming([{"role": "user", "content": "hi"}])
    except Exception:
        pass  # only the invocation matters
    assert calls, "run_streaming never invoked _ensure_fresh_token"


def test_openai_sync_request_path_refreshes_token_per_request():
    adapter = _openai("tok")

    calls = []

    def spy():
        calls.append(1)
        raise RuntimeError("short-circuit before any network I/O")

    adapter._ensure_fresh_token = spy
    try:
        adapter.run_streaming([{"role": "user", "content": "hi"}])
    except Exception:
        pass
    assert calls, "run_streaming never invoked _ensure_fresh_token"


def test_update_credential_file_writes_without_posix_lock(tmp_path: Path):
    """The credential-file write must succeed cross-platform — on win32 there is
    no ``fcntl``; the atomic temp-file + ``os.replace`` is the durability
    guarantee. Before the fix this raised ``ImportError`` on Windows."""
    cred_path = tmp_path / ".credentials.json"
    cred_path.write_text(
        json.dumps({"claudeAiOauth": {"accessToken": "old", "expiresAt": 1, "refreshToken": "r"}})
    )

    OAuthTokenRefresher._update_credential_file(
        cred_path, "anthropic-oauth", {"access_token": "rotated", "expires_in": 3600}
    )

    written = json.loads(cred_path.read_text())
    assert written["claudeAiOauth"]["accessToken"] == "rotated"
    assert written["claudeAiOauth"]["refreshToken"] == "r"  # preserved
