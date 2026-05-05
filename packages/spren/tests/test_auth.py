"""Unit tests for spren.auth."""
from __future__ import annotations

from spren.auth import generate_token, validate_bearer


class TestGenerateToken:
    def test_returns_url_safe_string(self) -> None:
        token = generate_token()
        assert isinstance(token, str)
        assert len(token) >= 40
        assert all(c.isalnum() or c in "-_" for c in token)

    def test_distinct_invocations(self) -> None:
        assert generate_token() != generate_token()


class TestValidateBearer:
    def test_accepts_correct_token(self) -> None:
        token = generate_token()
        assert validate_bearer(f"Bearer {token}", token) is True

    def test_rejects_wrong_token(self) -> None:
        assert validate_bearer("Bearer wrong", "right") is False

    def test_rejects_missing_header(self) -> None:
        assert validate_bearer(None, "right") is False

    def test_rejects_empty_header(self) -> None:
        assert validate_bearer("", "right") is False

    def test_rejects_missing_bearer_prefix(self) -> None:
        token = generate_token()
        assert validate_bearer(token, token) is False

    def test_rejects_when_expected_empty(self) -> None:
        assert validate_bearer("Bearer something", "") is False
