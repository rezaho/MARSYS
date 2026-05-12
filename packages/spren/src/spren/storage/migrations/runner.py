"""Roll-our-own migrations runner.

Reads `<NN>__<slug>.py` files alongside this module, runs unapplied ones in
declared order, and records applied IDs in a `_migrations` table. Each
migration file exports `def upgrade(conn: sqlite3.Connection) -> None`. Forward
only; no downgrade. Each migration runs in a transaction — partial failure
leaves no row in `_migrations` for that migration.

Idempotent: subsequent runs are no-ops because the runner skips IDs already
recorded.
"""
from __future__ import annotations

import importlib.util
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Callable

_MIGRATION_FILE = re.compile(r"^(\d+)__([a-z0-9_]+)\.py$")
_MIGRATIONS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS _migrations (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    applied_at TEXT NOT NULL
)
"""


class MigrationsRunner:
    """Discovers and applies pending migrations against a SQLite connection."""

    def __init__(self, conn: sqlite3.Connection, *, migrations_dir: Path | None = None) -> None:
        self._conn = conn
        self._migrations_dir = migrations_dir or Path(__file__).parent

    def run(self) -> list[str]:
        """Apply all pending migrations. Returns the IDs applied (in order)."""
        self._conn.execute(_MIGRATIONS_TABLE_DDL)
        applied = self._applied_ids()
        pending = [(mid, name, path) for mid, name, path in self._discover() if mid not in applied]
        for migration_id, name, path in pending:
            self._apply(migration_id, name, path)
        return [mid for mid, _, _ in pending]

    # --- internals ---

    def _applied_ids(self) -> set[str]:
        rows = self._conn.execute("SELECT id FROM _migrations").fetchall()
        return {row[0] for row in rows}

    def _discover(self) -> list[tuple[str, str, Path]]:
        items: list[tuple[str, str, Path]] = []
        for path in sorted(self._migrations_dir.iterdir()):
            if not path.is_file() or path.name.startswith("_"):
                continue
            match = _MIGRATION_FILE.match(path.name)
            if not match:
                continue
            items.append((match.group(1), match.group(2), path))
        return items

    def _apply(self, migration_id: str, name: str, path: Path) -> None:
        upgrade = self._load_upgrade(migration_id, name, path)
        try:
            self._conn.execute("BEGIN")
            upgrade(self._conn)
            self._conn.execute(
                "INSERT INTO _migrations (id, name, applied_at) VALUES (?, ?, ?)",
                (migration_id, name, datetime.now(timezone.utc).isoformat()),
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    @staticmethod
    def _load_upgrade(migration_id: str, name: str, path: Path) -> Callable[[sqlite3.Connection], None]:
        module_name = f"_spren_migration_{migration_id}_{name}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load migration {path}")
        module: ModuleType = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        upgrade = getattr(module, "upgrade", None)
        if not callable(upgrade):
            raise RuntimeError(f"migration {path.name} does not export upgrade(conn)")
        return upgrade
