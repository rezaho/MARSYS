"""Data-access helpers for the `workflows` table.

Row ↔ Pydantic model conversion lives here. JSON columns are stored as TEXT
and decoded on read. The store is a thin functional layer (no ORM); query
logic stays in the route handlers.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Iterable

from spren.models import Workflow, WorkflowDefinition, WorkflowProvenance


def utc_now_iso() -> str:
    """ISO 8601 UTC timestamp with microsecond precision."""
    return datetime.now(timezone.utc).isoformat()


def row_to_workflow(row: sqlite3.Row) -> Workflow:
    return Workflow(
        id=row["id"],
        name=row["name"],
        description=row["description"],
        definition=WorkflowDefinition.model_validate_json(row["definition"]),
        definition_version=row["definition_version"],
        provenance=row["provenance"],
        provenance_metadata=(
            json.loads(row["provenance_metadata"]) if row["provenance_metadata"] else None
        ),
        is_archived=bool(row["is_archived"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def insert_workflow(
    conn: sqlite3.Connection,
    *,
    workflow_id: str,
    name: str,
    description: str | None,
    definition: WorkflowDefinition,
    provenance: WorkflowProvenance,
    provenance_metadata: dict[str, Any] | None,
    definition_version: int = 1,
) -> Workflow:
    now = utc_now_iso()
    conn.execute(
        """
        INSERT INTO workflows (
            id, name, description, definition, definition_version,
            provenance, provenance_metadata, is_archived, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
        """,
        (
            workflow_id,
            name,
            description,
            definition.model_dump_json(),
            definition_version,
            provenance,
            json.dumps(provenance_metadata) if provenance_metadata is not None else None,
            now,
            now,
        ),
    )
    return _fetch_one(conn, workflow_id)


def fetch_workflow(conn: sqlite3.Connection, workflow_id: str) -> Workflow | None:
    row = conn.execute("SELECT * FROM workflows WHERE id = ?", (workflow_id,)).fetchone()
    return row_to_workflow(row) if row else None


def list_workflows(
    conn: sqlite3.Connection,
    *,
    cursor: str | None,
    limit: int,
    provenance: WorkflowProvenance | None = None,
    archived: bool | None = None,
    include_drafts: bool = False,
) -> tuple[list[Workflow], bool]:
    """Returns (items, has_more). Cursor is the ULID of the last returned row.

    Empty drafts (provenance='visual_builder' AND topology.nodes=[]) are
    filtered out by default; pass ``include_drafts=True`` to see them. Draft
    detection is predicate-based (no schema migration); first explicit save
    populates topology.nodes and the row exits draft state automatically.
    """
    sql = "SELECT * FROM workflows WHERE 1=1"
    params: list[Any] = []
    if cursor is not None:
        sql += " AND id > ?"
        params.append(cursor)
    if provenance is not None:
        sql += " AND provenance = ?"
        params.append(provenance)
    if archived is None:
        sql += " AND is_archived = 0"
    elif archived:
        sql += " AND is_archived = 1"
    else:
        sql += " AND is_archived = 0"
    if not include_drafts:
        # An empty visual_builder draft is hidden — UNLESS it has runs.
        # A workflow with run history is not an abandoned draft (SP-009:
        # run snapshots are immutable); surface it. Kept in sync with the
        # sweeper predicate in delete_empty_drafts_older_than.
        sql += (
            " AND NOT ("
            "provenance = 'visual_builder' AND "
            "json_extract(definition, '$.topology.nodes') = '[]' AND "
            "NOT EXISTS (SELECT 1 FROM runs WHERE runs.workflow_id = workflows.id)"
            ")"
        )
    sql += " ORDER BY id ASC LIMIT ?"
    params.append(limit + 1)
    rows = conn.execute(sql, params).fetchall()
    has_more = len(rows) > limit
    return [row_to_workflow(r) for r in rows[:limit]], has_more


def delete_empty_drafts_older_than(
    conn: sqlite3.Connection,
    *,
    max_age_iso: str,
) -> int:
    """Sweep empty visual-builder drafts older than ``max_age_iso``.

    Returns the number of rows deleted. The predicate matches the
    list-filter exactly so a draft hidden from the list IS the row the
    sweeper acts on, and a row that's been touched (topology populated) is
    never deleted.

    WF-BUG-SWEEPER-1: a workflow with runs is excluded. ``runs.workflow_id``
    has no ``ON DELETE`` (RESTRICT), so deleting such a row raised
    ``IntegrityError`` every sweep tick. SP-009 also makes it correct: a
    workflow with immutable run history is not an abandoned empty draft.
    """
    cur = conn.execute(
        """
        DELETE FROM workflows
        WHERE provenance = 'visual_builder'
          AND json_extract(definition, '$.topology.nodes') = '[]'
          AND updated_at < ?
          AND NOT EXISTS (
            SELECT 1 FROM runs WHERE runs.workflow_id = workflows.id
          )
        """,
        (max_age_iso,),
    )
    return cur.rowcount or 0


def replace_workflow(
    conn: sqlite3.Connection,
    workflow_id: str,
    *,
    name: str,
    description: str | None,
    definition: WorkflowDefinition,
    provenance: WorkflowProvenance,
    provenance_metadata: dict[str, Any] | None,
    definition_version: int = 1,
) -> Workflow | None:
    """PUT — full replacement except `id`/`created_at`/`is_archived`."""
    cur = conn.execute(
        """
        UPDATE workflows
        SET name = ?,
            description = ?,
            definition = ?,
            definition_version = ?,
            provenance = ?,
            provenance_metadata = ?,
            updated_at = ?
        WHERE id = ?
        """,
        (
            name,
            description,
            definition.model_dump_json(),
            definition_version,
            provenance,
            json.dumps(provenance_metadata) if provenance_metadata is not None else None,
            utc_now_iso(),
            workflow_id,
        ),
    )
    if cur.rowcount == 0:
        return None
    return _fetch_one(conn, workflow_id)


def patch_workflow(
    conn: sqlite3.Connection,
    workflow_id: str,
    *,
    fields: dict[str, Any],
) -> Workflow | None:
    """PATCH — only update provided keys. ``fields`` keys must already be column names."""
    if not fields:
        return fetch_workflow(conn, workflow_id)
    columns: list[str] = []
    values: list[Any] = []
    for key, value in fields.items():
        if key == "definition":
            columns.append("definition = ?")
            values.append(value.model_dump_json() if value is not None else None)
        elif key == "provenance_metadata":
            columns.append("provenance_metadata = ?")
            values.append(json.dumps(value) if value is not None else None)
        elif key == "is_archived":
            columns.append("is_archived = ?")
            values.append(1 if value else 0)
        else:
            columns.append(f"{key} = ?")
            values.append(value)
    columns.append("updated_at = ?")
    values.append(utc_now_iso())
    values.append(workflow_id)
    cur = conn.execute(
        f"UPDATE workflows SET {', '.join(columns)} WHERE id = ?",
        values,
    )
    if cur.rowcount == 0:
        return None
    return _fetch_one(conn, workflow_id)


def delete_workflow(conn: sqlite3.Connection, workflow_id: str) -> bool:
    cur = conn.execute("DELETE FROM workflows WHERE id = ?", (workflow_id,))
    return cur.rowcount > 0


def count_runs_referencing(conn: sqlite3.Connection, workflow_id: str) -> int:
    """Count run rows referencing ``workflow_id``.

    The DELETE workflow handler uses this to enforce the
    ``WORKFLOW_HAS_RUNS`` 409 path. The runs table was introduced by
    Session 04; the fallback to return 0 when the table is missing
    guards pre-migration code paths only.
    """
    table_exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='runs'"
    ).fetchone()
    if table_exists is None:
        return 0
    row = conn.execute(
        "SELECT COUNT(*) FROM runs WHERE workflow_id = ?", (workflow_id,)
    ).fetchone()
    return int(row[0]) if row else 0


def _fetch_one(conn: sqlite3.Connection, workflow_id: str) -> Workflow:
    row = conn.execute("SELECT * FROM workflows WHERE id = ?", (workflow_id,)).fetchone()
    if row is None:
        raise RuntimeError(f"workflow {workflow_id} not found after write — race?")
    return row_to_workflow(row)


def workflow_columns() -> Iterable[str]:
    """Convenience for tests asserting the workflows-table shape."""
    return (
        "id",
        "name",
        "description",
        "definition",
        "definition_version",
        "provenance",
        "provenance_metadata",
        "is_archived",
        "created_at",
        "updated_at",
    )
