"""Thin SQLite wrapper for the Spren sidecar.

The database file lives at `<data-dir>/data/spren.db`. Parent directories are
created on first call. WAL journal mode + foreign-key enforcement are enabled
so concurrent reads during writes don't block and FK references in later
schema versions actually hold.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path


class Database:
    """Wrapper around a `sqlite3.Connection` rooted at the per-user data dir."""

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._db_path = data_dir / "data" / "spren.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            self._db_path,
            detect_types=sqlite3.PARSE_DECLTYPES,
            check_same_thread=False,
            isolation_level=None,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")

    @property
    def path(self) -> Path:
        return self._db_path

    @property
    def connection(self) -> sqlite3.Connection:
        return self._connection

    def close(self) -> None:
        self._connection.close()
