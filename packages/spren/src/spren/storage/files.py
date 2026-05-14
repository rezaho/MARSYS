"""Data-access helpers for the ``files`` table.

Row ↔ Pydantic model conversion lives here. The store is a thin
functional layer (no ORM); query logic stays in the route handlers.

The reference-check delete uses SQLite's ``json_each`` against
``runs.task_input.attachments`` for element-equality matching (immune
to ULID-substring collisions that ``LIKE '%file_id%'`` would risk).
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from spren.models import FileMetadata


def utc_now_iso() -> str:
    """ISO 8601 UTC timestamp with microsecond precision."""
    return datetime.now(timezone.utc).isoformat()


def _row_to_file(row: sqlite3.Row) -> FileMetadata:
    return FileMetadata(
        id=row["id"],
        original_name=row["original_name"],
        mime_type=row["mime_type"],
        size_bytes=row["size_bytes"],
        sha256=row["sha256"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def insert_file(
    conn: sqlite3.Connection,
    *,
    file_id: str,
    original_name: str,
    mime_type: str,
    size_bytes: int,
    path: str,
    sha256: str,
) -> FileMetadata:
    now = utc_now_iso()
    conn.execute(
        """
        INSERT INTO files (id, original_name, mime_type, size_bytes, path, sha256, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (file_id, original_name, mime_type, size_bytes, path, sha256, now),
    )
    return fetch_file(conn, file_id)  # type: ignore[return-value]


def fetch_file(conn: sqlite3.Connection, file_id: str) -> FileMetadata | None:
    row = conn.execute("SELECT * FROM files WHERE id = ?", (file_id,)).fetchone()
    return _row_to_file(row) if row else None


def fetch_file_path(conn: sqlite3.Connection, file_id: str) -> tuple[FileMetadata, str] | None:
    """Return ``(metadata, on_disk_path)`` for the file, or ``None``.

    The on-disk ``path`` is intentionally hidden from ``FileMetadata``
    (which crosses the API boundary). Callers that need the path use
    this helper directly.
    """
    row = conn.execute("SELECT * FROM files WHERE id = ?", (file_id,)).fetchone()
    if row is None:
        return None
    return _row_to_file(row), row["path"]


def total_bytes_used(conn: sqlite3.Connection) -> int:
    """Sum of ``size_bytes`` across all files. Used for storage-cap checks."""
    row = conn.execute("SELECT COALESCE(SUM(size_bytes), 0) AS total FROM files").fetchone()
    return int(row["total"]) if row else 0


def runs_referencing_file(conn: sqlite3.Connection, file_id: str) -> list[str]:
    """Return run IDs whose ``task_input.attachments`` array contains ``file_id``.

    Uses ``json_each`` for element-equality matching — immune to
    ULID-substring collisions a ``LIKE`` would risk.
    """
    rows = conn.execute(
        """
        SELECT runs.id
        FROM runs, json_each(json_extract(runs.task_input, '$.attachments'))
        WHERE json_each.value = ?
        """,
        (file_id,),
    ).fetchall()
    return [r["id"] for r in rows]


def delete_file_row(conn: sqlite3.Connection, file_id: str) -> bool:
    """Delete the row. Returns ``True`` if a row was removed."""
    cur = conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
    return cur.rowcount > 0
