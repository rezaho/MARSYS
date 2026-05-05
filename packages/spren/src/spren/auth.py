"""Per-launch bearer-token authentication for the Spren FastAPI sidecar (SP-002).

The token is generated at process start, printed on stdout for the Tauri shell to
capture, and required on every request via `Authorization: Bearer <token>`.
Validation uses constant-time comparison to avoid timing oracles.
"""
from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, status

_BEARER_PREFIX = "Bearer "


def generate_token() -> str:
    return secrets.token_urlsafe(32)


def validate_bearer(header: str | None, expected: str) -> bool:
    if not header or not expected:
        return False
    if not header.startswith(_BEARER_PREFIX):
        return False
    presented = header[len(_BEARER_PREFIX) :]
    return secrets.compare_digest(presented, expected)


def make_auth_dependency(expected_token: str):
    def require_auth(authorization: str | None = Header(default=None)) -> None:
        if not validate_bearer(authorization, expected_token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="missing or invalid auth token",
            )

    return require_auth
