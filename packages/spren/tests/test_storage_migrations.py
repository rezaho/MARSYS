"""Migrations + storage layer unit tests."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from spren.storage import Database, MigrationsRunner
from spren.storage.idempotency import sweep_expired
from spren.storage.workflows import workflow_columns


@pytest.fixture
def db(tmp_path: Path) -> Database:
    return Database(tmp_path)


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {row[1] for row in rows}


def _index_names(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA index_list({table})").fetchall()
    return {row[1] for row in rows}


def test_database_creates_parent_dirs(tmp_path: Path):
    db = Database(tmp_path)
    assert db.path.exists()
    assert db.path.parent.is_dir()


def test_pragmas_are_set(db: Database):
    fk = db.connection.execute("PRAGMA foreign_keys").fetchone()[0]
    journal = db.connection.execute("PRAGMA journal_mode").fetchone()[0]
    assert fk == 1
    assert journal.lower() == "wal"


def test_migrations_runner_idempotent(db: Database):
    runner = MigrationsRunner(db.connection)
    first = runner.run()
    second = runner.run()
    assert first  # at least one migration applied
    assert second == []  # second pass is no-op


def test_migrations_table_records_applied(db: Database):
    MigrationsRunner(db.connection).run()
    rows = db.connection.execute("SELECT id, name, applied_at FROM _migrations").fetchall()
    assert rows
    assert any(row["id"] == "01" for row in rows)
    assert all(row["applied_at"] for row in rows)


def test_workflows_table_columns(db: Database):
    MigrationsRunner(db.connection).run()
    expected = set(workflow_columns())
    assert _table_columns(db.connection, "workflows") == expected


def test_workflows_table_column_types(db: Database):
    """column TYPES + NOT NULL constraints + PRIMARY KEY are correct."""
    MigrationsRunner(db.connection).run()
    rows = db.connection.execute("PRAGMA table_info(workflows)").fetchall()
    info = {row[1]: {"type": row[2], "notnull": row[3], "pk": row[5]} for row in rows}

    # cid, name, type, notnull, dflt_value, pk
    assert info["id"] == {"type": "TEXT", "notnull": 1, "pk": 1}
    assert info["name"]["type"] == "TEXT"
    assert info["name"]["notnull"] == 1
    assert info["description"]["type"] == "TEXT"
    assert info["description"]["notnull"] == 0
    assert info["definition"]["notnull"] == 1
    assert info["definition_version"] == {"type": "INTEGER", "notnull": 1, "pk": 0}
    assert info["provenance"]["notnull"] == 1
    assert info["is_archived"] == {"type": "INTEGER", "notnull": 1, "pk": 0}
    assert info["created_at"]["notnull"] == 1
    assert info["updated_at"]["notnull"] == 1


def test_idempotency_table_column_types(db: Database):
    """_idempotency column TYPES + NOT NULL + expires_at index."""
    MigrationsRunner(db.connection).run()
    rows = db.connection.execute("PRAGMA table_info(_idempotency)").fetchall()
    info = {row[1]: {"type": row[2], "notnull": row[3], "pk": row[5]} for row in rows}
    for col in ("method", "path", "key"):
        assert info[col]["pk"] >= 1, col
        assert info[col]["notnull"] == 1
    assert info["response_status"]["type"] == "INTEGER"
    assert info["response_body"]["type"] == "BLOB"
    assert info["response_headers"]["type"] == "TEXT"
    assert info["created_at"]["notnull"] == 1
    assert info["expires_at"]["notnull"] == 1
    indexes = _index_names(db.connection, "_idempotency")
    assert "idx_idempotency_expires_at" in indexes


def test_workflows_table_indexes(db: Database):
    MigrationsRunner(db.connection).run()
    indexes = _index_names(db.connection, "workflows")
    assert "idx_workflows_provenance" in indexes
    assert "idx_workflows_is_archived" in indexes
    assert "idx_workflows_created_at" in indexes


def test_idempotency_table_columns(db: Database):
    MigrationsRunner(db.connection).run()
    cols = _table_columns(db.connection, "_idempotency")
    expected = {
        "method",
        "path",
        "key",
        "response_status",
        "response_body",
        "response_headers",
        "created_at",
        "expires_at",
    }
    assert cols == expected


def test_idempotency_compound_primary_key(db: Database):
    MigrationsRunner(db.connection).run()
    rows = db.connection.execute("PRAGMA table_info(_idempotency)").fetchall()
    pk_cols = {row[1] for row in rows if row[5]}
    assert pk_cols == {"method", "path", "key"}


def test_idempotency_sweep_removes_expired_only(db: Database):
    MigrationsRunner(db.connection).run()
    now = datetime.now(timezone.utc)
    fresh = (now + timedelta(hours=1)).isoformat()
    stale = (now - timedelta(hours=1)).isoformat()
    db.connection.execute(
        "INSERT INTO _idempotency (method, path, key, response_status, response_body, response_headers, created_at, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("POST", "/v1/workflows", "fresh", 201, b"{}", "{}", now.isoformat(), fresh),
    )
    db.connection.execute(
        "INSERT INTO _idempotency (method, path, key, response_status, response_body, response_headers, created_at, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("POST", "/v1/workflows", "stale", 201, b"{}", "{}", (now - timedelta(days=2)).isoformat(), stale),
    )
    deleted = sweep_expired(db.connection)
    assert deleted == 1
    remaining = db.connection.execute("SELECT key FROM _idempotency").fetchall()
    assert {row[0] for row in remaining} == {"fresh"}


def test_partial_migration_failure_leaves_no_row(tmp_path: Path):
    """If a migration raises mid-run, the `_migrations` row must NOT be written."""
    db = Database(tmp_path)
    bad_dir = tmp_path / "fake_migrations"
    bad_dir.mkdir()
    (bad_dir / "01__bomb.py").write_text(
        "def upgrade(conn):\n    conn.execute('CREATE TABLE x (a INTEGER)')\n    raise RuntimeError('boom')\n"
    )
    runner = MigrationsRunner(db.connection, migrations_dir=bad_dir)
    with pytest.raises(RuntimeError, match="boom"):
        runner.run()
    rows = db.connection.execute("SELECT id FROM _migrations").fetchall()
    assert rows == []
    # The aborted CREATE TABLE was rolled back too.
    table_exists = db.connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='x'"
    ).fetchone()
    assert table_exists is None
