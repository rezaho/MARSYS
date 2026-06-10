"""OAuth adapters refresh + reload the access token at request time.

Regression for two defects that made a long-lived daemon 401 once its token
expired:

1. The token was refreshed only at ``__init__``; ``_refresh_token_if_needed``
   rewrites the credentials file but does NOT touch the cached
   ``self.access_token`` that ``get_headers`` sends — so ``_ensure_fresh_token``
   must reload it after a refresh.
2. ``OAuthTokenRefresher._update_credential_file`` used a POSIX-only
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


def test_ensure_fresh_token_noop_when_token_still_valid():
    """Refresh returns False (still valid) -> no reload, cached token preserved."""
    adapter = _anthropic("valid-token")
    with patch.object(AnthropicOAuthAdapter, "_refresh_token_if_needed", lambda self: False), patch.object(
        AnthropicOAuthAdapter, "_load_claude_credentials", lambda self, p=None: {"access_token": "must-not-load"}
    ):
        adapter._ensure_fresh_token()
    assert adapter.access_token == "valid-token"


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
