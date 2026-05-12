"""Periodic sweep for empty visual-builder drafts older than 24 hours.

The +New workflow flow creates an empty draft row on canvas mount (so the
ULID is stable for the autosave/lint URL). Drafts the user abandons before
saving any nodes accumulate; this sweeper deletes them on a 4-hour cadence.

The predicate matches the list-filter at `spren.storage.workflows.list_workflows`
exactly — a row hidden from the default list IS a sweep candidate after the
24-hour grace window. First explicit save advances `updated_at`, so a saved
workflow is never swept regardless of how long ago it was created.
"""
from __future__ import annotations

import asyncio
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Callable

from spren.storage import workflows


logger = logging.getLogger(__name__)


SWEEP_INTERVAL = timedelta(hours=4)
DRAFT_MAX_AGE = timedelta(hours=24)


def sweep_empty_drafts_once(
    conn_factory: Callable[[], sqlite3.Connection],
    *,
    now: datetime | None = None,
    max_age: timedelta = DRAFT_MAX_AGE,
) -> int:
    """Run a single sweep pass; returns the number of rows deleted."""
    current = now or datetime.now(timezone.utc)
    cutoff = (current - max_age).isoformat()
    conn = conn_factory()
    deleted = workflows.delete_empty_drafts_older_than(conn, max_age_iso=cutoff)
    if deleted:
        logger.info("draft-sweeper deleted %d empty drafts older than %s", deleted, cutoff)
    return deleted


async def run_draft_sweeper_forever(
    conn_factory: Callable[[], sqlite3.Connection],
    *,
    interval: timedelta = SWEEP_INTERVAL,
    max_age: timedelta = DRAFT_MAX_AGE,
) -> None:
    """Loop on ``interval``; stops cleanly on ``asyncio.CancelledError``.

    Intended to be scheduled via ``asyncio.create_task`` from the FastAPI
    lifespan handler. The task is cancelled on shutdown, the
    ``CancelledError`` is re-raised, and the lifespan handler then closes
    the database connection.
    """
    try:
        while True:
            try:
                sweep_empty_drafts_once(conn_factory, max_age=max_age)
            except Exception:  # noqa: BLE001 - sweeper must never bring the app down
                logger.exception("draft-sweeper pass failed; will retry on next interval")
            await asyncio.sleep(interval.total_seconds())
    except asyncio.CancelledError:
        logger.info("draft-sweeper task cancelled (server shutdown)")
        raise
