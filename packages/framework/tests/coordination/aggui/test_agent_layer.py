"""Agent-layer regression tests for AssistantMessageEvent emission (AC-7..AC-10).

Verifies that:
- ``AssistantMessageEvent`` is emitted on the EventBus after ``model.arun()`` returns.
- The event is NOT emitted on the agent error path.
- ``TraceCollector`` subscribes to it and stores content via the content-addressed
  ``MessageStore`` pattern (respecting ``TracingConfig.include_message_content``).
"""

from __future__ import annotations

import asyncio
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("ag_ui")

from marsys.agents.agents import Agent
from marsys.agents.memory import Message
from marsys.agents.registry import AgentRegistry
from marsys.coordination.event_bus import EventBus
from marsys.coordination.status.events import AssistantMessageEvent
from marsys.coordination.tracing.collector import TraceCollector
from marsys.coordination.tracing.config import TracingConfig
from marsys.models import ModelConfig


@pytest.fixture(autouse=True)
def clean_registry():
    AgentRegistry.clear()
    yield
    AgentRegistry.clear()


@pytest.fixture
def agent_with_event_bus():
    """An Agent whose `_step_event_bus` and `_step_context` are pre-wired
    so the emission gate at `agents.py:3110-3119` passes.
    """
    mock_model = MagicMock()
    mock_model.cleanup = AsyncMock()

    with patch("marsys.agents.agents.Agent._create_model_from_config", return_value=mock_model):
        agent = Agent(
            model_config=ModelConfig(
                type="api",
                name="test-model",
                provider="openrouter",
                api_key="test",
            ),
            goal="Test goal",
            instruction="You are a helpful assistant.",
            name="TestAgent",
        )
    agent.model = mock_model
    bus = EventBus()
    agent._step_event_bus = bus
    agent._step_context = {
        "session_id": "s1",
        "branch_id": "br1",
        "step_number": 1,
        "step_span_id": "span1",
    }
    return agent, bus, mock_model


@pytest.mark.asyncio
async def test_assistant_message_event_emitted_after_model_arun_success(
    agent_with_event_bus,
):
    """AC-7: AssistantMessageEvent is emitted on the EventBus after model.arun()
    returns successfully."""
    agent, bus, mock_model = agent_with_event_bus

    captured: List[AssistantMessageEvent] = []

    async def listener(event):
        captured.append(event)

    bus.subscribe("AssistantMessageEvent", listener)

    # Mock model.arun to return a simple harmonized response
    harmonized = MagicMock()
    harmonized.content = "Hello from the model."
    harmonized.tool_calls = None
    harmonized.finish_reason = "stop"
    mock_model.arun = AsyncMock(return_value=harmonized)

    # Patch Message.from_harmonized_response to return a known Message
    with patch.object(
        Message,
        "from_harmonized_response",
        return_value=Message(role="assistant", content="Hello from the model."),
    ):
        result = await agent._run(
            messages=[{"role": "user", "content": "hi"}],
            request_context={},
            run_mode="auto_step",
        )

    assert isinstance(result, Message)
    assert len(captured) == 1
    event = captured[0]
    assert event.agent_name == "TestAgent"
    assert event.session_id == "s1"
    assert event.branch_id == "br1"
    assert event.step_number == 1
    assert event.step_span_id == "span1"
    assert event.content == "Hello from the model."
    assert event.finish_reason == "stop"


@pytest.mark.asyncio
async def test_assistant_message_event_not_emitted_on_error_path(
    agent_with_event_bus,
):
    """AC-8: AssistantMessageEvent is NOT emitted when model.arun() raises."""
    agent, bus, mock_model = agent_with_event_bus
    captured: List[AssistantMessageEvent] = []

    async def listener(event):
        captured.append(event)

    bus.subscribe("AssistantMessageEvent", listener)

    # Mock model.arun to raise
    mock_model.arun = AsyncMock(side_effect=RuntimeError("model exploded"))

    # The agent's error handling converts exceptions to an error Message
    result = await agent._run(
        messages=[{"role": "user", "content": "hi"}],
        request_context={},
        run_mode="auto_step",
    )

    # No AssistantMessageEvent emitted on the error path
    assert captured == []


@pytest.mark.asyncio
async def test_trace_collector_subscribes_to_assistant_message_event():
    """AC-9: TraceCollector includes AssistantMessageEvent in its event_handlers map."""
    bus = EventBus()
    cfg = TracingConfig(enabled=True, output_dir="/tmp/marsys-test-trace")
    collector = TraceCollector(event_bus=bus, config=cfg, sinks=[])
    subscribed = set(bus.listeners.keys())
    assert "AssistantMessageEvent" in subscribed


@pytest.mark.asyncio
async def test_trace_collector_handler_stores_assistant_content_when_enabled(tmp_path):
    """AC-10a: Handler stores content in MessageStore when include_message_content=True."""
    bus = EventBus()
    cfg = TracingConfig(
        enabled=True,
        output_dir=str(tmp_path),
        include_message_content=True,
    )
    collector = TraceCollector(event_bus=bus, config=cfg, sinks=[])

    # Seed a step span so the handler has somewhere to attach the ref.
    from marsys.coordination.status.events import AgentStartEvent
    from marsys.coordination.tracing.events import ExecutionStartEvent

    await bus.emit(
        ExecutionStartEvent(
            session_id="s1",
            task_summary="t",
            topology_summary={},
            agent_names=["A"],
            config_summary={},
        )
    )
    await bus.emit(
        AgentStartEvent(
            session_id="s1",
            branch_id="br1",
            agent_name="A",
            step_number=1,
            step_span_id="span1",
        )
    )

    # Emit the assistant message
    event = AssistantMessageEvent(
        session_id="s1",
        branch_id="br1",
        agent_name="A",
        step_number=1,
        step_span_id="span1",
        message_id="msg_X",
        content="secret content here",
    )
    await bus.emit(event)

    # Content should have been released (replaced with empty string) after hashing
    assert event.content == ""
    # And the step span should carry the ref
    step_span = collector.step_spans.get("span1")
    assert step_span is not None
    output_ref = step_span.attributes.get("output_message_ref")
    assert output_ref is not None
    assert output_ref["message_id"] == "msg_X"
    assert output_ref["content_hash"] is not None


@pytest.mark.asyncio
async def test_trace_collector_handler_respects_include_message_content_false(tmp_path):
    """AC-10b: When include_message_content=False, content is not stored;
    only the message_id correlation is attached."""
    bus = EventBus()
    cfg = TracingConfig(
        enabled=True,
        output_dir=str(tmp_path),
        include_message_content=False,
    )
    collector = TraceCollector(event_bus=bus, config=cfg, sinks=[])

    from marsys.coordination.status.events import AgentStartEvent
    from marsys.coordination.tracing.events import ExecutionStartEvent

    await bus.emit(
        ExecutionStartEvent(
            session_id="s1",
            task_summary="t",
            topology_summary={},
            agent_names=["A"],
            config_summary={},
        )
    )
    await bus.emit(
        AgentStartEvent(
            session_id="s1",
            branch_id="br1",
            agent_name="A",
            step_number=1,
            step_span_id="span1",
        )
    )

    event = AssistantMessageEvent(
        session_id="s1",
        branch_id="br1",
        agent_name="A",
        step_number=1,
        step_span_id="span1",
        message_id="msg_Y",
        content="secret content",
    )
    await bus.emit(event)

    # Content still released for memory reasons
    assert event.content == ""
    step_span = collector.step_spans.get("span1")
    assert step_span is not None
    output_ref = step_span.attributes.get("output_message_ref")
    assert output_ref is not None
    assert output_ref["message_id"] == "msg_Y"
    # Content not stored — hash is None
    assert output_ref["content_hash"] is None
