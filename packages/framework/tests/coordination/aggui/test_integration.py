"""Integration test: synthetic 3-agent workflow → full AG-UI event sequence.

Uses the EventBus directly with a curated sequence of framework events that
mimics a User → Researcher → Writer run. Avoids depending on a real LLM
provider (no API key required to run this test). The integration test
verifies the full event sequence is well-formed, validates against the
AG-UI SDK, and round-trips through the SSE encoder.

A separate `@pytest.mark.cheap` test could exercise a real model run; not
included here to keep the suite hermetic.
"""

from __future__ import annotations

import asyncio
import json
from typing import List

import pytest

pytest.importorskip("ag_ui")

from ag_ui.core import (
    BaseEvent,
    CustomEvent,
    RunFinishedEvent,
    RunStartedEvent,
    StateSnapshotEvent,
    StepFinishedEvent,
    StepStartedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
)

from marsys.coordination.aggui import (
    AGGUIConfig,
    AGGUITranslator,
    AGUIEventStream,
    aggui_event_to_sse,
)
from marsys.coordination.event_bus import EventBus
from marsys.coordination.events import BranchCompletedEvent, BranchCreatedEvent
from marsys.coordination.status.events import (
    AgentCompleteEvent,
    AgentStartEvent,
    AssistantMessageEvent,
    FinalResponseEvent,
    ParallelGroupEvent,
    ToolCallEvent,
)
from marsys.coordination.tracing.events import (
    ConvergenceEvent,
    ExecutionStartEvent,
)


SESSION_ID = "01J0AGGUITESTSESSION0000A1"


async def _synthetic_three_agent_run(bus: EventBus) -> None:
    """Emit a curated event sequence mimicking User → Researcher → Writer."""
    # 1. Run start
    await bus.emit(
        ExecutionStartEvent(
            session_id=SESSION_ID,
            task_summary="Research and write a brief",
            topology_summary={"shape": "linear", "nodes": ["User", "Researcher", "Writer"]},
            agent_names=["User", "Researcher", "Writer"],
            config_summary={},
        )
    )

    # 2. Researcher branch spawned
    await bus.emit(
        BranchCreatedEvent(
            session_id=SESSION_ID,
            branch_id="br1",
            branch_name="researcher-branch",
            source_agent="User",
            target_agents=["Researcher"],
            trigger_type="invoke",
        )
    )

    # 3. Researcher steps
    await bus.emit(
        AgentStartEvent(
            session_id=SESSION_ID,
            branch_id="br1",
            agent_name="Researcher",
            step_number=1,
        )
    )
    await bus.emit(
        AssistantMessageEvent(
            session_id=SESSION_ID,
            branch_id="br1",
            agent_name="Researcher",
            step_number=1,
            message_id="msg_R1",
            content="Let me search for information.",
            finish_reason="tool_calls",
        )
    )
    await bus.emit(
        ToolCallEvent(
            session_id=SESSION_ID,
            branch_id="br1",
            agent_name="Researcher",
            tool_name="search",
            status="started",
            arguments={"query": "marsys framework"},
            step_number=1,
        )
    )
    await bus.emit(
        ToolCallEvent(
            session_id=SESSION_ID,
            branch_id="br1",
            agent_name="Researcher",
            tool_name="search",
            status="completed",
            result_summary="Found 3 relevant docs.",
            step_number=1,
        )
    )
    await bus.emit(
        AgentCompleteEvent(
            session_id=SESSION_ID,
            branch_id="br1",
            agent_name="Researcher",
            success=True,
            duration=0.5,
            step_number=1,
        )
    )

    # 4. Researcher branch completes
    await bus.emit(
        BranchCompletedEvent(
            session_id=SESSION_ID,
            branch_id="br1",
            last_agent="Researcher",
            success=True,
            total_steps=1,
        )
    )

    # 5. Writer branch spawned
    await bus.emit(
        BranchCreatedEvent(
            session_id=SESSION_ID,
            branch_id="br2",
            branch_name="writer-branch",
            source_agent="Researcher",
            target_agents=["Writer"],
            trigger_type="invoke",
            parent_branch_id="br1",
        )
    )

    # 6. Writer step
    await bus.emit(
        AgentStartEvent(
            session_id=SESSION_ID,
            branch_id="br2",
            agent_name="Writer",
            step_number=1,
        )
    )
    await bus.emit(
        AssistantMessageEvent(
            session_id=SESSION_ID,
            branch_id="br2",
            agent_name="Writer",
            step_number=1,
            message_id="msg_W1",
            content="Here is the brief based on research.",
            finish_reason="stop",
        )
    )
    await bus.emit(
        AgentCompleteEvent(
            session_id=SESSION_ID,
            branch_id="br2",
            agent_name="Writer",
            success=True,
            duration=0.4,
            step_number=1,
        )
    )

    # 7. Writer branch completes
    await bus.emit(
        BranchCompletedEvent(
            session_id=SESSION_ID,
            branch_id="br2",
            last_agent="Writer",
            success=True,
            total_steps=1,
        )
    )

    # 8. Run finishes
    await bus.emit(
        FinalResponseEvent(
            session_id=SESSION_ID,
            final_response="Here is the brief based on research.",
            total_duration=1.0,
            total_steps=2,
            success=True,
        )
    )


@pytest.fixture
def translator_and_bus():
    bus = EventBus()
    t = AGGUITranslator(event_bus=bus, config=AGGUIConfig(enabled=True))
    return t, bus


@pytest.mark.asyncio
async def test_three_agent_run_emits_well_formed_aggui_sequence(translator_and_bus):
    """Full sequence: handshake → RunStarted → StateSnapshot → … → RunFinished."""
    translator, bus = translator_and_bus
    stream = AGUIEventStream(translator)
    await _synthetic_three_agent_run(bus)
    events: List[BaseEvent] = []
    async for event in stream:
        events.append(event)

    # First event is the handshake Custom
    assert isinstance(events[0], CustomEvent)
    assert events[0].name == "marsys.aggui.handshake"
    assert events[0].value["schema_version"] == 1
    # Second is RunStarted
    assert isinstance(events[1], RunStartedEvent)
    assert events[1].run_id == SESSION_ID
    # Third is initial StateSnapshot
    assert isinstance(events[2], StateSnapshotEvent)
    # Last is RunFinished
    assert isinstance(events[-1], RunFinishedEvent)
    assert events[-1].result["final_response"] == "Here is the brief based on research."
    assert events[-1].result["total_steps"] == 2


@pytest.mark.asyncio
async def test_text_message_triples_are_well_formed(translator_and_bus):
    """Every TextMessageStart has a matching TextMessageEnd with same message_id."""
    translator, bus = translator_and_bus
    stream = AGUIEventStream(translator)
    await _synthetic_three_agent_run(bus)
    events = [e async for e in stream]

    starts = [e for e in events if isinstance(e, TextMessageStartEvent)]
    ends = [e for e in events if isinstance(e, TextMessageEndEvent)]
    assert len(starts) == 2  # Researcher + Writer
    assert len(ends) == 2
    start_ids = sorted(e.message_id for e in starts)
    end_ids = sorted(e.message_id for e in ends)
    assert start_ids == end_ids
    # And every Start is followed by a Content (delta) and then an End for the same message
    for i, ev in enumerate(events):
        if isinstance(ev, TextMessageStartEvent):
            assert isinstance(events[i + 1], TextMessageContentEvent)
            assert events[i + 1].message_id == ev.message_id
            assert isinstance(events[i + 2], TextMessageEndEvent)
            assert events[i + 2].message_id == ev.message_id


@pytest.mark.asyncio
async def test_tool_call_sequences_are_well_formed(translator_and_bus):
    """Every ToolCallStart's tool_call_id matches the subsequent End and Result."""
    translator, bus = translator_and_bus
    stream = AGUIEventStream(translator)
    await _synthetic_three_agent_run(bus)
    events = [e async for e in stream]

    starts = [e for e in events if isinstance(e, ToolCallStartEvent)]
    ends = [e for e in events if isinstance(e, ToolCallEndEvent)]
    results = [e for e in events if isinstance(e, ToolCallResultEvent)]
    assert len(starts) == 1
    assert len(ends) == 1
    assert len(results) == 1
    assert starts[0].tool_call_id == ends[0].tool_call_id == results[0].tool_call_id
    assert starts[0].tool_call_name == "search"
    assert results[0].content == "Found 3 relevant docs."


@pytest.mark.asyncio
async def test_state_delta_fires_when_current_agent_changes(translator_and_bus):
    """When a branch's current_agent flips, a StateDelta is emitted."""
    translator, bus = translator_and_bus
    stream = AGUIEventStream(translator)
    await _synthetic_three_agent_run(bus)
    events = [e async for e in stream]
    # At least one StateDelta for each AgentStartEvent that changed state
    from ag_ui.core import StateDeltaEvent
    deltas = [e for e in events if isinstance(e, StateDeltaEvent)]
    assert len(deltas) >= 2  # one per Researcher#1 and Writer#1


@pytest.mark.asyncio
async def test_every_event_passes_sdk_round_trip(translator_and_bus):
    """Every emitted event validates as a real ag_ui.core.BaseEvent via model_dump/model_validate."""
    translator, bus = translator_and_bus
    stream = AGUIEventStream(translator)
    await _synthetic_three_agent_run(bus)
    events = [e async for e in stream]
    for ev in events:
        cls = type(ev)
        dumped = ev.model_dump()
        reconstructed = cls.model_validate(dumped)
        assert reconstructed == ev


@pytest.mark.asyncio
async def test_sse_encoding_round_trips(translator_and_bus):
    """aggui_event_to_sse produces valid SSE lines that re-parse via the encoder."""
    translator, bus = translator_and_bus
    stream = AGUIEventStream(translator)
    await _synthetic_three_agent_run(bus)
    events = [e async for e in stream]
    for ev in events:
        wire = aggui_event_to_sse(ev)
        # SSE format: `data: {json}\n\n` or `event: TYPE\ndata: {json}\n\n`
        assert wire.endswith("\n\n") or wire.endswith("\n"), wire
        # Should be parseable JSON in the data line
        for line in wire.splitlines():
            if line.startswith("data:"):
                payload = line[5:].strip()
                if payload:
                    json.loads(payload)
                    break
        else:
            pytest.fail(f"No data: line found in SSE output: {wire!r}")


@pytest.mark.asyncio
async def test_no_spren_type_imported():
    """SP-018 spot-check: importing the aggui package must not pull in any spren type."""
    import importlib
    import sys
    # Snapshot loaded modules
    before = set(sys.modules.keys())
    importlib.reload(importlib.import_module("marsys.coordination.aggui"))
    after = set(sys.modules.keys())
    new = after - before
    spren_modules = [m for m in (new | after) if m.startswith("spren") or m == "spren"]
    assert spren_modules == [], (
        f"aggui module imported spren types: {spren_modules}. "
        f"SP-018 violation — framework knows nothing of Spren."
    )
