"""Create the runs table."""
from __future__ import annotations

import sqlite3


def upgrade(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE runs (
            id TEXT PRIMARY KEY NOT NULL,
            workflow_id TEXT NOT NULL,
            status TEXT NOT NULL,
            task_input TEXT NOT NULL,
            trigger TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT,
            total_steps INTEGER,
            total_duration_ms INTEGER,
            total_tokens_input INTEGER NOT NULL DEFAULT 0,
            total_tokens_output INTEGER NOT NULL DEFAULT 0,
            total_cost_usd REAL NOT NULL DEFAULT 0.0,
            final_response TEXT,
            error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (workflow_id) REFERENCES workflows(id)
        )
        """
    )
    conn.execute("CREATE INDEX idx_runs_workflow_id_created_at ON runs(workflow_id, created_at)")
    conn.execute("CREATE INDEX idx_runs_status ON runs(status)")
    conn.execute("CREATE INDEX idx_runs_created_at ON runs(created_at)")
