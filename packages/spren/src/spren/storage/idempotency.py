"""Idempotency-Key replay cache.

Storage shape: a single ``_idempotency`` row per ``(method, path, key)``.
24-hour TTL. Mismatched method/path with the same key is treated as a fresh
request — that's the cache key, not the bare key.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

CACHE_TTL = timedelta(hours=24)


@dataclass
class CachedResponse:
    status: int
    body: bytes
    headers: dict[str, str]


def fetch(
    conn: sqlite3.Connection,
    *,
    method: str,
    path: str,
    key: str,
) -> CachedResponse | None:
    row = conn.execute(
        """
        SELECT response_status, response_body, response_headers, expires_at
        FROM _idempotency
        WHERE method = ? AND path = ? AND key = ?
        """,
        (method, path, key),
    ).fetchone()
    if row is None:
        return None
    expires_at = datetime.fromisoformat(row["expires_at"])
    if expires_at < datetime.now(timezone.utc):
        conn.execute(
            "DELETE FROM _idempotency WHERE method = ? AND path = ? AND key = ?",
            (method, path, key),
        )
        return None
    return CachedResponse(
        status=int(row["response_status"]),
        body=bytes(row["response_body"]),
        headers=json.loads(row["response_headers"]),
    )


def store(
    conn: sqlite3.Connection,
    *,
    method: str,
    path: str,
    key: str,
    status: int,
    body: bytes,
    headers: dict[str, str],
) -> None:
    now = datetime.now(timezone.utc)
    expires = now + CACHE_TTL
    conn.execute(
        """
        INSERT OR REPLACE INTO _idempotency (
            method, path, key,
            response_status, response_body, response_headers,
            created_at, expires_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            method,
            path,
            key,
            status,
            body,
            json.dumps(headers),
            now.isoformat(),
            expires.isoformat(),
        ),
    )


def sweep_expired(conn: sqlite3.Connection) -> int:
    """Delete every row whose ``expires_at`` is past. Returns the row count."""
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.execute("DELETE FROM _idempotency WHERE expires_at < ?", (now,))
    return cur.rowcount or 0
