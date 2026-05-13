"""AGUIEventStream iterator semantics — slow consumer, terminal close."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("ag_ui")

from ag_ui.core import BaseEvent, CustomEvent

from marsys.coordination.aggui import AGGUIConfig, AGGUITranslator, AGUIEventStream
from marsys.coordination.event_bus import EventBus
from marsys.coordination.status.events import (
    AssistantMessageEvent,
    FinalResponseEvent,
)
from marsys.coordination.tracing.events import ExecutionStartEvent


@pytest.fixture
def bus_and_translator():
    bus = EventBus()
    t = AGGUITranslator(event_bus=bus, config=AGGUIConfig(enabled=True))
    return bus, t


@pytest.mark.asyncio
async def test_stream_is_async_iterator(bus_and_translator):
    bus, t = bus_and_translator
    stream = AGUIEventStream(t)
    assert hasattr(stream, "__aiter__")
    assert hasattr(stream, "__anext__")
    aiter_result = stream.__aiter__()
    assert aiter_result is stream  # __aiter__ returns self


@pytest.mark.asyncio
async def test_stream_yields_events_in_emit_order(bus_and_translator):
    bus, t = bus_and_translator
    stream = AGUIEventStream(t)

    # Fire ExecutionStartEvent — yields 3 events
    await bus.emit(
        ExecutionStartEvent(
            session_id="s1",
            task_summary="t",
            topology_summary={},
            agent_names=[],
            config_summary={},
        )
    )
    # Fire terminal
    await bus.emit(
        FinalResponseEvent(
            session_id="s1",
            final_response="done",
            total_duration=1.0,
            total_steps=1,
            success=True,
        )
    )
    # Drain
    collected = []
    async for event in stream:
        collected.append(event)
    assert len(collected) == 4  # handshake, RunStarted, StateSnapshot, RunFinished
    assert isinstance(collected[0], CustomEvent)
    assert collected[0].name == "marsys.aggui.handshake"


@pytest.mark.asyncio
async def test_stream_raises_stop_async_iteration_after_terminal_and_drain():
    bus = EventBus()
    t = AGGUITranslator(event_bus=bus, config=AGGUIConfig(enabled=True))
    stream = AGUIEventStream(t)
    # Mark terminal and ensure StopAsyncIteration once queue is empty
    t.mark_terminal()
    with pytest.raises(StopAsyncIteration):
        await stream.__anext__()


@pytest.mark.asyncio
async def test_slow_consumer_does_not_deadlock_on_small_queue():
    """100 events through a 10-slot queue + slow consumer (1ms sleep per event).
    The translator drops newest; the stream eventually drains terminally."""
    bus = EventBus()
    t = AGGUITranslator(event_bus=bus, config=AGGUIConfig(enabled=True, queue_max_size=10))

    async def producer():
        for i in range(50):
            await bus.emit(
                AssistantMessageEvent(
                    session_id="s1",
                    agent_name="A",
                    step_number=i,
                    message_id=f"m{i}",
                    content="hi",
                )
            )
        # Terminal
        await bus.emit(
            FinalResponseEvent(
                session_id="s1",
                final_response="done",
                total_duration=1.0,
                total_steps=50,
                success=True,
            )
        )

    async def consumer():
        stream = AGUIEventStream(t)
        count = 0
        async for event in stream:
            count += 1
            await asyncio.sleep(0.001)
        return count

    producer_task = asyncio.create_task(producer())
    consumer_task = asyncio.create_task(consumer())
    # Both should finish within a reasonable timeout — no deadlock
    await asyncio.wait_for(
        asyncio.gather(producer_task, consumer_task), timeout=10.0
    )
    count = consumer_task.result()
    # At least some events were consumed; many may have been dropped — but no deadlock.
    assert count > 0
