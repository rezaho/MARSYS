"""Integration tests for TelemetrySink wired through Orchestra.run.

Covers: a user-supplied TelemetrySink registered via TracingConfig.sinks
receives every closed span; close() called at run end; multiple sinks all
see the same stream; user sinks register alongside the default NDJSON sink
without conflict.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Dict, List

import pytest

from marsys.agents import Agent
from marsys.agents.memory import Message, ToolCallMsg
from marsys.agents.registry import AgentRegistry
from marsys.coordination import Orchestra
from marsys.coordination.config import ExecutionConfig
from marsys.coordination.tracing import (
    NDJSONTraceReader,
    NDJSONTraceWriter,
    TelemetrySink,
    TraceTree,
    TracingConfig,
)
from marsys.models import ModelConfig


# ── Test fixtures: minimal multi-step topology ────────────────────────────


def _coord_tool_call(name: str, arguments: dict) -> ToolCallMsg:
    cid = f"call_{uuid.uuid4().hex[:8]}"
    return ToolCallMsg(
        id=cid,
        call_id=cid,
        type="function",
        name=name,
        arguments=json.dumps(arguments),
    )


@pytest.fixture(autouse=True)
def cleanup_registry():
    AgentRegistry.clear()
    yield
    AgentRegistry.clear()


class _SimpleCoord(Agent):
    """Coordinator that fans out to two workers, then terminates."""

    def __init__(self):
        super().__init__(
            model_config=ModelConfig(type="api", name="mock", provider="openai", api_key="mock-key"),
            goal="coordinate",
            instruction="Coordinate two workers.",
            name="Coord",
        )
        self._fanned_out = False

    async def _run(self, messages, request_context, run_mode: str = "default", **kwargs):
        if self._fanned_out:
            return Message(
                role="assistant",
                content="Done.",
                tool_calls=[_coord_tool_call("return_final_response", {"response": "done"})],
                name=self.name,
            )
        self._fanned_out = True
        return Message(
            role="assistant",
            content="Fanning out.",
            tool_calls=[_coord_tool_call("invoke_agent", {
                "invocations": [
                    {"agent_name": "WorkerA", "request": "a"},
                    {"agent_name": "WorkerB", "request": "b"},
                ]
            })],
            name=self.name,
        )


class _SimpleWorker(Agent):
    def __init__(self, name: str):
        super().__init__(
            model_config=ModelConfig(type="api", name="mock", provider="openai", api_key="mock-key"),
            goal=f"work {name}",
            instruction="Do work.",
            name=name,
        )

    async def _run(self, messages, request_context, run_mode: str = "default", **kwargs):
        await asyncio.sleep(0.01)
        return Message(
            role="assistant",
            content=f"{self.name} done",
            tool_calls=[_coord_tool_call(
                "invoke_agent",
                {"invocations": [{"agent_name": "Coord", "request": f"data from {self.name}"}]},
            )],
            name=self.name,
        )


def _topology() -> dict:
    return {
        "agents": ["User", "Coord", "WorkerA", "WorkerB"],
        "flows": [
            "User -> Coord",
            "Coord -> WorkerA",
            "Coord -> WorkerB",
            "WorkerA -> Coord",
            "WorkerB -> Coord",
            "Coord -> User",
        ],
        "exit_points": ["Coord"],
        "rules": [],
    }


# ── Recording sink fixture ────────────────────────────────────────────────


class RecordingTelemetrySink(TelemetrySink):
    """Captures every publish_span call plus the close() call for inspection."""

    def __init__(self) -> None:
        self.spans: List[Any] = []
        self.close_calls = 0

    async def publish_span(self, span) -> None:
        self.spans.append(span)

    async def close(self) -> None:
        self.close_calls += 1


# ── Tests ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_recording_sink_receives_every_closed_span(tmp_path):
    AgentRegistry._test_agents = [
        _SimpleCoord(), _SimpleWorker("WorkerA"), _SimpleWorker("WorkerB"),
    ]
    sink = RecordingTelemetrySink()
    tracing = TracingConfig(enabled=True, output_dir=str(tmp_path), sinks=[sink])
    cfg = ExecutionConfig(tracing=tracing)

    result = await Orchestra.run(
        task="run", topology=_topology(), execution_config=cfg, max_steps=20,
    )
    assert result.success, f"orchestration failed: {result.error}"

    assert len(sink.spans) > 0, "sink received no spans"
    assert sink.close_calls == 1


@pytest.mark.asyncio
async def test_recording_sink_spans_match_ndjson_file_count(tmp_path):
    """The user sink and the default NDJSON sink see the same span set."""
    AgentRegistry._test_agents = [
        _SimpleCoord(), _SimpleWorker("WorkerA"), _SimpleWorker("WorkerB"),
    ]
    sink = RecordingTelemetrySink()
    tracing = TracingConfig(enabled=True, output_dir=str(tmp_path), sinks=[sink])
    cfg = ExecutionConfig(tracing=tracing)

    result = await Orchestra.run(
        task="parity", topology=_topology(), execution_config=cfg, max_steps=20,
    )
    assert result.success

    files = list(tmp_path.glob("*.ndjson"))
    assert len(files) == 1
    reader = NDJSONTraceReader(files[0])
    file_spans = list(reader.stream())

    # Recording sink saw the same number of closed spans the NDJSON file
    # contains (modulo non-span marker lines).
    assert len(sink.spans) == len(file_spans)


@pytest.mark.asyncio
async def test_two_recording_sinks_receive_identical_streams(tmp_path):
    AgentRegistry._test_agents = [
        _SimpleCoord(), _SimpleWorker("WorkerA"), _SimpleWorker("WorkerB"),
    ]
    sink_a = RecordingTelemetrySink()
    sink_b = RecordingTelemetrySink()
    tracing = TracingConfig(
        enabled=True, output_dir=str(tmp_path), sinks=[sink_a, sink_b],
    )
    cfg = ExecutionConfig(tracing=tracing)

    result = await Orchestra.run(
        task="dual", topology=_topology(), execution_config=cfg, max_steps=20,
    )
    assert result.success

    assert len(sink_a.spans) == len(sink_b.spans) > 0
    # Same span IDs in the same order.
    assert [s.span_id for s in sink_a.spans] == [s.span_id for s in sink_b.spans]
    assert sink_a.close_calls == 1
    assert sink_b.close_calls == 1


@pytest.mark.asyncio
async def test_sink_close_called_even_when_orchestra_fails(tmp_path):
    """close() fires in Orchestra's finally block; failure path still closes sinks."""
    # No agents registered → orchestration fails fast.
    AgentRegistry.clear()

    sink = RecordingTelemetrySink()
    tracing = TracingConfig(enabled=True, output_dir=str(tmp_path), sinks=[sink])
    cfg = ExecutionConfig(tracing=tracing)

    result = await Orchestra.run(
        task="fail", topology=_topology(), execution_config=cfg, max_steps=20,
    )
    assert not result.success

    assert sink.close_calls == 1


@pytest.mark.asyncio
async def test_default_ndjson_writer_still_active_when_user_sinks_present(tmp_path):
    """User sinks register alongside the default NDJSON sink, not in place of it."""
    AgentRegistry._test_agents = [
        _SimpleCoord(), _SimpleWorker("WorkerA"), _SimpleWorker("WorkerB"),
    ]
    sink = RecordingTelemetrySink()
    tracing = TracingConfig(enabled=True, output_dir=str(tmp_path), sinks=[sink])
    cfg = ExecutionConfig(tracing=tracing)

    result = await Orchestra.run(
        task="default+user", topology=_topology(), execution_config=cfg, max_steps=20,
    )
    assert result.success

    # NDJSON file written despite the user sink being present.
    files = list(tmp_path.glob("*.ndjson"))
    assert len(files) == 1
    # User sink also got everything.
    assert len(sink.spans) > 0
    # OrchestraResult.metadata["tracing"] still populated by the NDJSON sink.
    assert result.metadata["tracing"]["total_spans"] > 0
