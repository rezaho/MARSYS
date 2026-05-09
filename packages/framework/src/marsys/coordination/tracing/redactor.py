"""
SecretRedactor — scrubs known-secret keys from span attribute payloads.

Runs once at the fan-out boundary inside TraceCollector._stream_span,
mutating in place. All consumers (NDJSON writer, vendor sinks, in-memory
TraceTree) see the same redacted view.

Walks span.attributes recursively. Also walks every event['attributes']
dict in span.events and every link['attributes'] dict in span.links —
those sub-dicts can carry secrets too (e.g., a tool-call event whose
attributes include the tool's api_key argument).
"""

import re
from typing import TYPE_CHECKING, Any, List, Pattern, Tuple

if TYPE_CHECKING:
    from .types import Span


class SecretRedactor:
    """Redacts known-secret keys from span attribute payloads.

    Default deny-list (case-insensitive, word-boundary match against dict keys):
      api_key, apikey, token, authorization, auth, secret, password,
      bearer, cookie, session, credential

    Word boundaries treat ``_`` and ``-`` (and any non-alphanumeric character)
    as separators, so ``auth_token`` and ``x_api_key`` redact but ``prompt_tokens``
    (an LLM token-count metric) and ``authority`` do not. Matched keys at any
    nesting depth have their values replaced with the configured replacement
    string. Mutates in place.
    """

    DEFAULT_DENY: Tuple[str, ...] = (
        "api_key", "apikey", "token", "authorization", "auth",
        "secret", "password", "bearer", "cookie", "session", "credential",
    )

    def __init__(
        self,
        *,
        extra_deny: Tuple[str, ...] = (),
        replacement: str = "[REDACTED]",
    ):
        self._deny: Tuple[str, ...] = tuple(
            s.lower() for s in (*self.DEFAULT_DENY, *extra_deny)
        )
        self._replacement: str = replacement
        self._patterns: List[Pattern[str]] = [
            re.compile(rf"(?:^|[^a-z0-9]){re.escape(entry)}(?:$|[^a-z0-9])")
            for entry in self._deny
        ]

    def redact(self, attributes: dict) -> None:
        """Mutate the attributes dict in place. Recurses into nested dicts and lists."""
        for key in list(attributes.keys()):
            if isinstance(key, str) and self._is_denied(key):
                attributes[key] = self._replacement
                continue
            value = attributes[key]
            self._redact_value(value)

    def redact_span(self, span: 'Span') -> None:
        """Apply redact() to span.attributes, span.events[*]['attributes'], span.links[*]['attributes']."""
        if span.attributes:
            self.redact(span.attributes)
        for event in span.events:
            attrs = event.get('attributes')
            if isinstance(attrs, dict):
                self.redact(attrs)
        for link in span.links:
            attrs = link.get('attributes')
            if isinstance(attrs, dict):
                self.redact(attrs)

    def _is_denied(self, key: str) -> bool:
        lower = key.lower()
        return any(p.search(lower) for p in self._patterns)

    def _redact_value(self, value: Any) -> None:
        if isinstance(value, dict):
            self.redact(value)
        elif isinstance(value, list):
            for item in value:
                self._redact_value(item)


class NoRedactionRedactor(SecretRedactor):
    """Opt-in no-op redactor. Caller accepts the leak risk.

    Use when local debugging or tests need raw values. Production runs
    that ship traces to bug reports / external backends should keep
    the default SecretRedactor.
    """

    def redact(self, attributes: dict) -> None:
        return None

    def redact_span(self, span: 'Span') -> None:
        return None
