"""AGGUITranslator subscription lifecycle + backpressure tests."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("ag_ui")

from ag_ui.core import CustomEvent

from marsys.coordination.aggui import AGGUIConfig, AGGUITranslator
from marsys.coordination.aggui import mapping as _mapping
from marsys.coordination.event_bus import EventBus
from marsys.coordination.status.events import (
    AgentStartEvent,
    AssistantMessageEvent,
    ErrorEvent,
)


@pytest.fixture
def bus() -> EventBus:
    return EventBus()


@pytest.fixture
def translator(bus: EventBus) -> AGGUITranslator:
    return AGGUITranslator(event_bus=bus, config=AGGUIConfig(enabled=True))


def test_subscribes_to_every_dispatch_event_class(bus: EventBus, translator: AGGUITranslator):
    subscribed = set(bus.listeners.keys())
    expected = {cls.__name__ for cls in _mapping.DISPATCH}
    assert expected.issubset(subscribed)


def test_does_not_subscribe_to_internal_only_or_not_yet_emitted(
    bus: EventBus, translator: AGGUITranslator
):
    """INTERNAL_ONLY and NOT_YET_EMITTED classes should NOT have AGGUI handlers
    attached — they would just be no-ops anyway, but cleaner to skip."""
    subscribed = set(bus.listeners.keys())
    for cls in _mapping.INTERNAL_ONLY:
        # Translator should not subscribe to AgentMessagesPreparedEvent or MemoryResetEvent
        # (would always yield empty)
        assert cls.__name__ not in subscribed or len(bus.listeners[cls.__name__]) == 0
    # Note: this is an architectural sanity check; the translator may
    # subscribe to these as a no-op if implementation chose to. The real
    # contract is that map_event returns empty for these classes.


@pytest.mark.asyncio
async def test_close_unsubscribes_and_marks_closed(bus: EventBus, translator: AGGUITranslator):
    assert translator._closed is False
    assert len(bus.listeners) > 0
    await translator.close()
    assert translator._closed is True
    # After close, the listeners list for each DISPATCH event class should be empty
    for cls in _mapping.DISPATCH:
        listeners = bus.listeners.get(cls.__name__, [])
        assert translator._handle not in listeners


@pytest.mark.asyncio
async def test_event_flows_through_translator_to_queue(bus: EventBus, translator: AGGUITranslator):
    event = AssistantMessageEvent(
        session_id="s1",
        agent_name="A",
        step_number=1,
        message_id="msg1",
        content="hi",
    )
    await bus.emit(event)
    # Triple: Start, Content, End
    assert translator.queue.qsize() == 3


@pytest.mark.asyncio
async def test_drop_newest_on_overflow_increments_lagged_counter():
    """Queue with max=2; emit 5 events. First 2 land; next 3 dropped."""
    bus = EventBus()
    t = AGGUITranslator(event_bus=bus, config=AGGUIConfig(enabled=True, queue_max_size=2))
    # Each AssistantMessageEvent emits 3 AG-UI events. After the first emission,
    # the queue holds 2 and 1 was dropped (lagged_count=1).
    await bus.emit(
        AssistantMessageEvent(
            session_id="s1",
            agent_name="A",
            step_number=1,
            message_id="m1",
            content="hello",
        )
    )
    assert t.queue.full()
    assert t._lagged_count >= 1


@pytest.mark.asyncio
async def test_lagged_custom_emits_on_next_successful_enqueue():
    """After overflow + room — the NEXT enqueue prefixes a marsys.stream.lagged Custom."""
    bus = EventBus()
    t = AGGUITranslator(event_bus=bus, config=AGGUIConfig(enabled=True, queue_max_size=2))
    # Fill + overflow
    await bus.emit(
        AssistantMessageEvent(
            session_id="s1",
            agent_name="A",
            step_number=1,
            message_id="m1",
            content="hi",
        )
    )
    assert t._lagged_count >= 1
    drop_count = t._lagged_count
    # Drain to make room
    a = t.queue.get_nowait()
    b = t.queue.get_nowait()
    assert t.queue.empty()
    # Next event triggers lagged Custom to be enqueued BEFORE the new event
    await bus.emit(
        ErrorEvent(
            session_id="s1",
            agent_name="A",
            error_class="X",
            error_message="boom",
        )
    )
    # First out should be the lagged Custom
    first = t.queue.get_nowait()
    assert isinstance(first, CustomEvent)
    assert first.name == "marsys.stream.lagged"
    assert first.value["count"] == drop_count
    assert t._lagged_count == 0


@pytest.mark.asyncio
async def test_mapper_failure_does_not_propagate(monkeypatch, bus: EventBus):
    """If a mapper raises, the exception is logged but other subscribers and
    the rest of the translator continue working."""
    t = AGGUITranslator(event_bus=bus, config=AGGUIConfig(enabled=True))

    def boom(event, ctx):
        raise RuntimeError("mapper failure")

    # Replace the AgentStartEvent mapper with the broken one
    monkeypatch.setitem(_mapping.DISPATCH, AgentStartEvent, boom)
    other_subscriber_called = asyncio.Event()

    async def other(event):
        other_subscriber_called.set()

    bus.subscribe("AgentStartEvent", other)
    # Emit the event — mapper raises but other subscriber still runs
    await bus.emit(AgentStartEvent(session_id="s1", agent_name="A", step_number=1))
    assert other_subscriber_called.is_set()
    # Translator queue still empty (the mapper raised before enqueuing)
    assert t.queue.empty()
    # Translator not closed
    assert t._closed is False


@pytest.mark.asyncio
async def test_terminal_event_marks_closed_via_run_finished():
    """When a terminal mapper (e.g. FinalResponseEvent success) runs,
    ctx.mark_terminal() is called and _closed is True."""
    from marsys.coordination.status.events import FinalResponseEvent
    bus = EventBus()
    t = AGGUITranslator(event_bus=bus, config=AGGUIConfig(enabled=True))
    await bus.emit(
        FinalResponseEvent(
            session_id="s1",
            final_response="done",
            total_duration=1.0,
            total_steps=1,
            success=True,
        )
    )
    assert t._closed is True
