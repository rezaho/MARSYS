"""Per-run AG-UI SSE wrapper.

Wraps the framework's ``AGUIEventStream`` (Framework 06) in an
SSE-friendly async iterator with reconnect-via-Last-Event-ID using the
in-memory replay buffer held on each ``ActiveRun``.

When Framework 06 is not yet present, the iterator yields nothing and
ends — the SSE endpoint surfaces 503 in that case (acceptance criteria
that depend on AG-UI consumption are tagged ``[blocked-on:
framework-06]`` and skip-pass during the parallel period).
"""
from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from spren.runs.lifecycle import ActiveRun

logger = logging.getLogger(__name__)


def _aggui_available() -> bool:
    """Runtime presence check for Framework 06."""
    try:
        import marsys.coordination.aggui  # type: ignore[import-not-found]  # noqa: F401

        return True
    except ImportError:
        return False


def _split_replay_at(active: ActiveRun, last_event_id: str | None) -> list[Any]:
    """Return the suffix of replay events whose event_id > ``last_event_id``.

    The replay buffer holds tuples ``(event_id, encoded_sse)`` keyed by
    AG-UI event ULIDs. ULIDs are lexicographically ordered by creation
    time, so a string comparison ``eid > last_event_id`` yields the
    correct "events newer than the client has seen" semantics.

    If ``last_event_id`` is older than the deque's oldest entry, returns
    a single ``("__GAP__", None)`` sentinel so the caller can emit
    ``Custom("marsys.stream.gap")``.
    """
    items = list(active.replay)
    if not last_event_id:
        return items
    if items and last_event_id < items[0][0]:
        # Older than the deque's oldest entry → gap (events evicted by drop-oldest)
        return [("__GAP__", None)]
    suffix: list[Any] = [(eid, encoded) for (eid, encoded) in items if eid > last_event_id]
    return suffix


async def stream_run_events(
    *,
    active: ActiveRun,
    last_event_id: str | None = None,
) -> AsyncIterator[str]:
    """Yield SSE-encoded strings for one run's AG-UI events.

    1. Replay buffer suffix (if Last-Event-ID was provided).
    2. Live events from ``AGUIEventStream(translator)`` until terminal.

    Emits each event as ``id: {event_id}\\nevent: {type}\\ndata: {json}\\n\\n``.
    The framework's ``aggui_event_to_sse`` does the encoding when
    Framework 06 is present.
    """
    if not _aggui_available():
        logger.info(
            "stream_run_events: marsys.coordination.aggui not available; "
            "no live events for run_id=%s",
            active.run_id,
        )
        return

    # Replay first
    for entry in _split_replay_at(active, last_event_id):
        if entry[0] == "__GAP__":
            yield (
                "event: marsys.stream.gap\n"
                "data: {\"message\": \"replay buffer exhausted; refresh via REST\"}\n\n"
            )
            continue
        _, encoded = entry
        if encoded:
            yield encoded

    # Live consumption
    from marsys.coordination.aggui import (  # type: ignore[import-not-found]
        AGUIEventStream,
        aggui_event_to_sse,
    )

    translator = getattr(active.orchestra, "aggui_translator", None)
    if translator is None:
        logger.info(
            "stream_run_events: orchestra.aggui_translator not set for run_id=%s; "
            "the run's ExecutionConfig.aggui.enabled may be False or Framework 06 not wired",
            active.run_id,
        )
        return

    stream = AGUIEventStream(translator)
    async for event in stream:
        encoded = aggui_event_to_sse(event)
        # Persist into replay buffer keyed by event_id (best-effort)
        event_id = getattr(event, "id", None) or getattr(event, "event_id", None)
        if event_id:
            active.replay.append((str(event_id), encoded))
        yield encoded
