"""Data-access helpers for the `runs` table.

Row ↔ Pydantic model conversion lives here. JSON columns are stored as TEXT
and decoded on read. The store is a thin functional layer (no ORM); query
logic stays in the route handlers.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from spren.models.run import RunListItem, RunRead, RunStatus, TaskInput


def utc_now_iso() -> str:
    """ISO 8601 UTC timestamp with microsecond precision."""
    return datetime.now(timezone.utc).isoformat()


def _row_to_run(row: sqlite3.Row) -> RunRead:
    return RunRead(
        id=row["id"],
        workflow_id=row["workflow_id"],
        status=RunStatus(row["status"]),
        task_input=TaskInput.model_validate_json(row["task_input"]),
        trigger=row["trigger"],
        started_at=(datetime.fromisoformat(row["started_at"]) if row["started_at"] else None),
        finished_at=(datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None),
        total_steps=row["total_steps"],
        total_duration_ms=row["total_duration_ms"],
        total_tokens_input=row["total_tokens_input"],
        total_tokens_output=row["total_tokens_output"],
        total_cost_usd=row["total_cost_usd"],
        final_response=(json.loads(row["final_response"]) if row["final_response"] else None),
        error=row["error"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def _row_to_list_item(row: sqlite3.Row) -> RunListItem:
    return RunListItem(
        id=row["id"],
        workflow_id=row["workflow_id"],
        status=RunStatus(row["status"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        finished_at=(datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None),
        total_duration_ms=row["total_duration_ms"],
        total_cost_usd=row["total_cost_usd"],
    )


def insert_run(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    workflow_id: str,
    task_input: TaskInput,
    trigger: str = "manual",
) -> RunRead:
    now = utc_now_iso()
    conn.execute(
        """
        INSERT INTO runs (
            id, workflow_id, status, task_input, trigger,
            total_tokens_input, total_tokens_output, total_cost_usd,
            created_at, updated_at
        ) VALUES (?, ?, 'queued', ?, ?, 0, 0, 0.0, ?, ?)
        """,
        (run_id, workflow_id, task_input.model_dump_json(), trigger, now, now),
    )
    return _fetch_one(conn, run_id)


def fetch_run(conn: sqlite3.Connection, run_id: str) -> RunRead | None:
    row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    return _row_to_run(row) if row else None


def _fetch_one(conn: sqlite3.Connection, run_id: str) -> RunRead:
    row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    if row is None:
        raise LookupError(f"run {run_id} not found")
    return _row_to_run(row)


def list_runs(
    conn: sqlite3.Connection,
    *,
    cursor: str | None,
    limit: int,
    workflow_id: str | None = None,
    status: RunStatus | None = None,
    since: datetime | None = None,
) -> tuple[list[RunListItem], bool]:
    """Returns (items, has_more). Newest-first ordering.

    Cursor is the ULID of the last returned row. ULIDs are monotonic
    (k-sortable by creation time within a process), so ``id DESC`` is
    equivalent to ``created_at DESC``; cursor walks backwards in time.
    """
    sql = "SELECT * FROM runs WHERE 1=1"
    params: list[Any] = []
    if cursor is not None:
        sql += " AND id < ?"
        params.append(cursor)
    if workflow_id is not None:
        sql += " AND workflow_id = ?"
        params.append(workflow_id)
    if status is not None:
        sql += " AND status = ?"
        params.append(status.value)
    if since is not None:
        sql += " AND created_at >= ?"
        params.append(since.isoformat())
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit + 1)
    rows = conn.execute(sql, params).fetchall()
    has_more = len(rows) > limit
    return [_row_to_list_item(r) for r in rows[:limit]], has_more


def update_run_status(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    status: RunStatus,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
    total_steps: int | None = None,
    total_duration_ms: int | None = None,
    final_response: Any | None = None,
    error: str | None = None,
) -> RunRead:
    now = utc_now_iso()
    sets: list[str] = ["status = ?", "updated_at = ?"]
    params: list[Any] = [status.value, now]
    if started_at is not None:
        sets.append("started_at = ?")
        params.append(started_at.isoformat())
    if finished_at is not None:
        sets.append("finished_at = ?")
        params.append(finished_at.isoformat())
    if total_steps is not None:
        sets.append("total_steps = ?")
        params.append(total_steps)
    if total_duration_ms is not None:
        sets.append("total_duration_ms = ?")
        params.append(total_duration_ms)
    if final_response is not None:
        sets.append("final_response = ?")
        params.append(json.dumps(final_response))
    if error is not None:
        sets.append("error = ?")
        params.append(error)
    params.append(run_id)
    conn.execute(f"UPDATE runs SET {', '.join(sets)} WHERE id = ?", params)
    return _fetch_one(conn, run_id)


def apply_cost_delta(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    cost_usd: float,
    tokens_in: int,
    tokens_out: int,
) -> None:
    now = utc_now_iso()
    conn.execute(
        """
        UPDATE runs
        SET total_cost_usd = total_cost_usd + ?,
            total_tokens_input = total_tokens_input + ?,
            total_tokens_output = total_tokens_output + ?,
            updated_at = ?
        WHERE id = ?
        """,
        (cost_usd, tokens_in, tokens_out, now, run_id),
    )
