"""Create the workflows and _idempotency tables."""
from __future__ import annotations

import sqlite3


def upgrade(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE workflows (
            id TEXT PRIMARY KEY NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            definition TEXT NOT NULL,
            definition_version INTEGER NOT NULL,
            provenance TEXT NOT NULL,
            provenance_metadata TEXT,
            is_archived INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX idx_workflows_provenance ON workflows(provenance)")
    conn.execute("CREATE INDEX idx_workflows_is_archived ON workflows(is_archived)")
    conn.execute("CREATE INDEX idx_workflows_created_at ON workflows(created_at)")

    conn.execute(
        """
        CREATE TABLE _idempotency (
            method TEXT NOT NULL,
            path TEXT NOT NULL,
            key TEXT NOT NULL,
            response_status INTEGER NOT NULL,
            response_body BLOB NOT NULL,
            response_headers TEXT NOT NULL,
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            PRIMARY KEY (method, path, key)
        )
        """
    )
    conn.execute("CREATE INDEX idx_idempotency_expires_at ON _idempotency(expires_at)")
