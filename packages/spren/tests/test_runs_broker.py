"""Unit tests for the RunsBroker pub/sub."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from spren.models import RunCreatedEvent, RunListItem, RunStatus, RunUpdatedEvent
from spren.runs.broker import RunsBroker, StreamLaggedMarker


def _item(run_id: str = "run-1", st: RunStatus = RunStatus.QUEUED) -> RunListItem:
    return RunListItem(
        id=run_id,
        workflow_id="wf-1",
        status=st,
        created_at=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
async def test_publish_with_no_subscribers_is_noop():
    broker = RunsBroker()
    broker.publish(RunCreatedEvent(run=_item()))
    assert broker.subscriber_count == 0


@pytest.mark.asyncio
async def test_subscribe_receives_published_events():
    broker = RunsBroker()
    async with broker.subscription() as sub:
        broker.publish(RunCreatedEvent(run=_item("run-A")))
        broker.publish(RunUpdatedEvent(run=_item("run-A", RunStatus.RUNNING)))
        first = await sub.get()
        second = await sub.get()
        assert isinstance(first, RunCreatedEvent)
        assert first.run.id == "run-A"
        assert isinstance(second, RunUpdatedEvent)


@pytest.mark.asyncio
async def test_unsubscribe_after_context_exit():
    broker = RunsBroker()
    async with broker.subscription():
        assert broker.subscriber_count == 1
    assert broker.subscriber_count == 0


@pytest.mark.asyncio
async def test_drop_oldest_on_overflow():
    """When the queue is full, drop oldest and increment lag counter."""
    broker = RunsBroker()
    sub = await broker.subscribe(queue_size=2)
    try:
        # Fill the queue
        broker.publish(RunCreatedEvent(run=_item("run-1")))
        broker.publish(RunCreatedEvent(run=_item("run-2")))
        # Overflow
        broker.publish(RunCreatedEvent(run=_item("run-3")))
        broker.publish(RunCreatedEvent(run=_item("run-4")))

        # First get returns a StreamLaggedMarker reflecting drops
        first = await sub.get()
        assert isinstance(first, StreamLaggedMarker)
        assert first.dropped_count >= 1

        # Subsequent gets return real events
        second = await sub.get()
        assert isinstance(second, RunCreatedEvent)
    finally:
        await broker.unsubscribe(sub)


@pytest.mark.asyncio
async def test_multiple_subscribers_each_receive_events():
    broker = RunsBroker()
    async with broker.subscription() as sub_a:
        async with broker.subscription() as sub_b:
            broker.publish(RunCreatedEvent(run=_item()))
            event_a = await sub_a.get()
            event_b = await sub_b.get()
            assert isinstance(event_a, RunCreatedEvent)
            assert isinstance(event_b, RunCreatedEvent)


@pytest.mark.asyncio
async def test_get_blocks_until_publish():
    broker = RunsBroker()
    async with broker.subscription() as sub:
        async def publisher():
            await asyncio.sleep(0.05)
            broker.publish(RunCreatedEvent(run=_item()))

        publish_task = asyncio.create_task(publisher())
        try:
            event = await asyncio.wait_for(sub.get(), timeout=1.0)
            assert isinstance(event, RunCreatedEvent)
        finally:
            await publish_task


@pytest.mark.asyncio
async def test_default_queue_size_is_256():
    """AC-67: bounded queue capacity defaults to 256."""
    broker = RunsBroker()
    sub = await broker.subscribe()  # default queue_size
    try:
        # Internal access check — the queue's maxsize should be 256.
        assert sub._queue.maxsize == 256
    finally:
        await broker.unsubscribe(sub)
