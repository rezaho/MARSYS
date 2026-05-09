"""Unit tests for SecretRedactor.

Covers: default deny-list, case-insensitivity, nested dicts, lists, walking
span.events[*]['attributes'] and span.links[*]['attributes'], custom
extra_deny, replacement override, NoRedactionRedactor opt-out.
"""
from __future__ import annotations

import pytest

from marsys.coordination.tracing.redactor import NoRedactionRedactor, SecretRedactor
from marsys.coordination.tracing.types import create_span


# ── Default deny-list ──────────────────────────────────────────────────────


def test_default_deny_redacts_top_level_api_key():
    redactor = SecretRedactor()
    attrs = {"api_key": "sk-secret", "user_id": "alice"}
    redactor.redact(attrs)
    assert attrs["api_key"] == "[REDACTED]"
    assert attrs["user_id"] == "alice"


def test_default_deny_redacts_each_default_key():
    redactor = SecretRedactor()
    attrs = {
        "api_key": "v",
        "apikey": "v",
        "token": "v",
        "authorization": "v",
        "auth": "v",
        "secret": "v",
        "password": "v",
        "bearer": "v",
        "cookie": "v",
        "session": "v",
        "credential": "v",
        "harmless": "kept",
    }
    redactor.redact(attrs)
    for key in (
        "api_key", "apikey", "token", "authorization", "auth",
        "secret", "password", "bearer", "cookie", "session", "credential",
    ):
        assert attrs[key] == "[REDACTED]", f"{key} not redacted"
    assert attrs["harmless"] == "kept"


def test_redaction_is_case_insensitive():
    redactor = SecretRedactor()
    attrs = {"API_KEY": "v", "Token": "v", "PASSWORD": "v"}
    redactor.redact(attrs)
    assert attrs["API_KEY"] == "[REDACTED]"
    assert attrs["Token"] == "[REDACTED]"
    assert attrs["PASSWORD"] == "[REDACTED]"


def test_word_boundary_match_catches_compound_keys():
    """Compound keys redact when the deny entry sits at a word boundary (`_`/`-`)."""
    redactor = SecretRedactor()
    attrs = {"tool_api_key": "sk-...", "x_auth_token": "bearer-..."}
    redactor.redact(attrs)
    assert attrs["tool_api_key"] == "[REDACTED]"
    assert attrs["x_auth_token"] == "[REDACTED]"


def test_word_boundary_match_does_not_overmatch_metric_names():
    """`prompt_tokens` (LLM token count) must NOT redact even though `token` is in deny."""
    redactor = SecretRedactor()
    attrs = {
        "prompt_tokens": 200,
        "completion_tokens": 50,
        "reasoning_tokens": 10,
        "tokenizer_name": "cl100k",
        "authority": "example.com",
        "sessions_count": 3,
    }
    redactor.redact(attrs)
    assert attrs["prompt_tokens"] == 200
    assert attrs["completion_tokens"] == 50
    assert attrs["reasoning_tokens"] == 10
    assert attrs["tokenizer_name"] == "cl100k"
    assert attrs["authority"] == "example.com"
    assert attrs["sessions_count"] == 3


# ── Nested structures ──────────────────────────────────────────────────────


def test_redaction_recurses_into_nested_dicts():
    redactor = SecretRedactor()
    attrs = {
        "config": {
            "client": {"api_key": "sk-deep", "host": "example.com"},
        },
    }
    redactor.redact(attrs)
    assert attrs["config"]["client"]["api_key"] == "[REDACTED]"
    assert attrs["config"]["client"]["host"] == "example.com"


def test_redaction_walks_lists_of_dicts():
    redactor = SecretRedactor()
    attrs = {
        "calls": [
            {"api_key": "sk-1", "name": "foo"},
            {"password": "p", "id": 7},
        ],
    }
    redactor.redact(attrs)
    assert attrs["calls"][0]["api_key"] == "[REDACTED]"
    assert attrs["calls"][0]["name"] == "foo"
    assert attrs["calls"][1]["password"] == "[REDACTED]"
    assert attrs["calls"][1]["id"] == 7


# ── Custom configuration ───────────────────────────────────────────────────


def test_extra_deny_extends_default_list():
    redactor = SecretRedactor(extra_deny=("internal_id",))
    attrs = {"internal_id": "x", "api_key": "y", "name": "z"}
    redactor.redact(attrs)
    assert attrs["internal_id"] == "[REDACTED]"
    assert attrs["api_key"] == "[REDACTED]"  # still in default deny
    assert attrs["name"] == "z"


def test_custom_replacement_string():
    redactor = SecretRedactor(replacement="***")
    attrs = {"api_key": "secret"}
    redactor.redact(attrs)
    assert attrs["api_key"] == "***"


# ── Mutation in place ──────────────────────────────────────────────────────


def test_redaction_mutates_in_place():
    redactor = SecretRedactor()
    attrs = {"api_key": "secret"}
    original_id = id(attrs)
    redactor.redact(attrs)
    assert id(attrs) == original_id


# ── Span-level walk ────────────────────────────────────────────────────────


def test_redact_span_walks_attributes_events_and_links():
    redactor = SecretRedactor()
    span = create_span("TR", "step1", "step", attributes={"api_key": "sk"})
    span.add_event("validation", {"token": "v", "is_valid": True})
    span.add_link("LK", "convergence", {"password": "p", "group_id": "g"})

    redactor.redact_span(span)

    assert span.attributes["api_key"] == "[REDACTED]"
    assert span.events[0]["attributes"]["token"] == "[REDACTED]"
    assert span.events[0]["attributes"]["is_valid"] is True
    assert span.links[0]["attributes"]["password"] == "[REDACTED]"
    assert span.links[0]["attributes"]["group_id"] == "g"


def test_redact_span_handles_empty_attributes_events_links():
    redactor = SecretRedactor()
    span = create_span("TR", "step1", "step")
    redactor.redact_span(span)  # no raise; nothing to redact


# ── NoRedactionRedactor opt-out ────────────────────────────────────────────


def test_no_redaction_redactor_passes_secrets_through():
    redactor = NoRedactionRedactor()
    attrs = {"api_key": "secret", "name": "foo"}
    redactor.redact(attrs)
    assert attrs["api_key"] == "secret"
    assert attrs["name"] == "foo"


def test_no_redaction_redactor_span_walk_is_noop():
    redactor = NoRedactionRedactor()
    span = create_span("TR", "step1", "step", attributes={"api_key": "sk"})
    span.add_event("e", {"token": "t"})
    redactor.redact_span(span)
    assert span.attributes["api_key"] == "sk"
    assert span.events[0]["attributes"]["token"] == "t"
