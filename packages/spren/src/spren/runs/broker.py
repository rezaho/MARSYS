"""In-process pub/sub for the aggregate ``GET /v1/runs/events`` SSE stream.

One singleton broker per FastAPI app. The runs lifecycle coordinator
publishes ``RunsListEvent`` instances; the SSE endpoint subscribes
per-client with a bounded async queue.

Backpressure: drop-oldest on overflow with a ``STREAM_LAGGED`` marker
on the next put. Default queue size 256 — the aggregate stream is
low-throughput (one event per row state change, not per AG-UI event).

Does NOT reuse the framework's ``EventBus`` — peer-emitting Spren
events alongside framework events would mix concerns and violate SP-018.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from spren.models import RunsListEvent

logger = logging.getLogger(__name__)


# Sentinel emitted to a subscriber's queue when its queue overflowed and
# events were dropped. Detected by the SSE endpoint and converted to a
# Custom("marsys.stream.lagged") on the wire.
class StreamLaggedMarker:
    """Sentinel for dropped events. Carries the cumulative drop count."""

    __slots__ = ("dropped_count",)

    def __init__(self, dropped_count: int) -> None:
        self.dropped_count = dropped_count


class RunsBroker:
    """Async pub/sub for aggregate run-list events."""

    def __init__(self, *, default_queue_size: int = 256) -> None:
        self._default_queue_size = default_queue_size
        self._subscribers: list[_Subscriber] = []
        self._lock = asyncio.Lock()

    def publish(self, event: RunsListEvent) -> None:
        """Non-blocking. Drop-oldest on full subscriber queues."""
        for sub in list(self._subscribers):
            sub.put_nowait(event)

    async def subscribe(self, *, queue_size: int | None = None) -> "_Subscriber":
        sub = _Subscriber(queue_size or self._default_queue_size)
        async with self._lock:
            self._subscribers.append(sub)
        return sub

    async def unsubscribe(self, sub: "_Subscriber") -> None:
        async with self._lock:
            try:
                self._subscribers.remove(sub)
            except ValueError:
                pass

    @asynccontextmanager
    async def subscription(
        self, *, queue_size: int | None = None
    ) -> AsyncIterator["_Subscriber"]:
        sub = await self.subscribe(queue_size=queue_size)
        try:
            yield sub
        finally:
            await self.unsubscribe(sub)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)


class _Subscriber:
    """One subscriber's bounded queue + drop-oldest semantics."""

    def __init__(self, queue_size: int) -> None:
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
        self._dropped_since_last_put = 0

    def put_nowait(self, event: RunsListEvent) -> None:
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            # drop-oldest: evict head, retry. The dropped event is
            # accounted for by the lag counter.
            try:
                _ = self._queue.get_nowait()
                self._dropped_since_last_put += 1
            except asyncio.QueueEmpty:
                pass
            try:
                self._queue.put_nowait(event)
            except asyncio.QueueFull:
                # Still full after eviction (race). Just count the new event as dropped.
                self._dropped_since_last_put += 1
                logger.warning("RunsBroker: subscriber queue still full after drop-oldest")

    async def get(self) -> RunsListEvent | StreamLaggedMarker:
        """Yields the next event, OR a StreamLaggedMarker if drops occurred."""
        if self._dropped_since_last_put > 0:
            count = self._dropped_since_last_put
            self._dropped_since_last_put = 0
            return StreamLaggedMarker(count)
        return await self._queue.get()
