"""Multi-consumer test: three differently-shaped fake sinks all see the same span tree.

Proves the TelemetrySink protocol genuinely fits adapters with different
target shapes in parallel — a `create_run`-style client, an OTel-span-dict
adapter, and a raw `to_dict()` poster. Each fake sink translates the
framework's Span into its own shape; all three see identical content;
secrets carried in ToolCallEvent.arguments are redacted in every view.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Dict, List, Optional

import pytest

from marsys.agents import Agent
from marsys.agents.memory import Message, ToolCallMsg
from marsys.agents.registry import AgentRegistry
from marsys.coordination import Orchestra
from marsys.coordination.config import ExecutionConfig
from marsys.coordination.tracing import (
    Span,
    TelemetrySink,
    TracingConfig,
)
from marsys.models import ModelConfig


# ── Differently-shaped fake sinks ──────────────────────────────────────────


class FakeClientRunSink(TelemetrySink):
    """Records `create_run`-style client calls per closed span (run_type /
    parent_run_id / inputs / outputs shape)."""

    def __init__(self) -> None:
        self.runs: List[Dict[str, Any]] = []
        self.closed = False

    async def publish_span(self, span: Span) -> None:
        self.runs.append({
            "name": span.name,
            "run_type": span.kind,
            "parent_run_id": span.parent_span_id,
            "start_time": span.start_time,
            "end_time": span.end_time,
            "inputs": span.attributes.get("inputs", {}),
            "outputs": span.attributes.get("outputs", {}),
            "attributes": dict(span.attributes),
        })

    async def close(self) -> None:
        self.closed = True


class FakeOtelSpanSink(TelemetrySink):
    """Records OTel-shaped span dicts per closed span (span_id / parent_id /
    uppercased kind / status shape)."""

    def __init__(self) -> None:
        self.spans: List[Dict[str, Any]] = []
        self.closed = False

    async def publish_span(self, span: Span) -> None:
        self.spans.append({
            "span_id": span.span_id,
            "parent_id": span.parent_span_id,
            "name": span.name,
            "kind": span.kind.upper(),
            "start_time": span.start_time,
            "end_time": span.end_time,
            "attributes": dict(span.attributes),
            "status": "OK" if span.status == "ok" else "ERROR",
        })

    async def close(self) -> None:
        self.closed = True


class FakeRawDictSink(TelemetrySink):
    """Records the protocol's calls verbatim (an adapter that posts
    span.to_dict() over HTTP)."""

    def __init__(self) -> None:
        self.payloads: List[Dict[str, Any]] = []
        self.closed = False

    async def publish_span(self, span: Span) -> None:
        self.payloads.append(span.to_dict())

    async def close(self) -> None:
        self.closed = True


# ── Topology fixtures ──────────────────────────────────────────────────────


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


class _Coord(Agent):
    """Coordinator that runs one tool with a secret-bearing argument, then terminates."""

    def __init__(self):
        super().__init__(
            model_config=ModelConfig(type="api", name="mock", provider="openai", api_key="mock-key"),
            goal="multi-consumer",
            instruction="Run a tool then terminate.",
            name="Coord",
        )
        self._tool_used = False

    async def _run(self, messages, request_context, run_mode: str = "default", **kwargs):
        if self._tool_used:
            return Message(
                role="assistant",
                content="Done.",
                tool_calls=[_coord_tool_call("return_final_response", {"response": "ok"})],
                name=self.name,
            )
        self._tool_used = True
        # The secret-bearing tool call: should be redacted by SecretRedactor.
        return Message(
            role="assistant",
            content="Calling tool with secret.",
            tool_calls=[_coord_tool_call("search", {"api_key": "sk-LEAK", "query": "hello"})],
            name=self.name,
        )


def _topology() -> dict:
    return {
        "agents": ["User", "Coord"],
        "flows": ["User -> Coord", "Coord -> User"],
        "exit_points": ["Coord"],
        "rules": [],
    }


# ── Tests ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_three_sink_shapes_all_see_same_span_tree(tmp_path):
    AgentRegistry._test_agents = [_Coord()]

    client_run = FakeClientRunSink()
    otel_span = FakeOtelSpanSink()
    raw_dict = FakeRawDictSink()
    tracing = TracingConfig(
        enabled=True, output_dir=str(tmp_path),
        sinks=[client_run, otel_span, raw_dict],
    )
    cfg = ExecutionConfig(tracing=tracing)

    result = await Orchestra.run(
        task="multi", topology=_topology(), execution_config=cfg, max_steps=10,
    )
    assert result.success, f"orchestration failed: {result.error}"

    # Same span count across all three fake sinks.
    assert len(client_run.runs) == len(otel_span.spans) == len(raw_dict.payloads)
    assert len(client_run.runs) > 0

    # Same span_ids in the same order.
    cr_ids = [r["attributes"].get("span_id") for r in client_run.runs]
    os_ids = [s["span_id"] for s in otel_span.spans]
    rd_ids = [p["span_id"] for p in raw_dict.payloads]
    assert os_ids == rd_ids
    # The client-run shape doesn't emit span_id directly, so just check shape
    # consistency via parent_run_id (parent_span_id) alignment.
    assert [r["parent_run_id"] for r in client_run.runs] == [
        s["parent_id"] for s in otel_span.spans
    ]

    # All three sinks closed.
    assert client_run.closed
    assert otel_span.closed
    assert raw_dict.closed


@pytest.mark.asyncio
async def test_secret_in_tool_arguments_redacted_in_all_three_sinks(tmp_path):
    AgentRegistry._test_agents = [_Coord()]

    client_run = FakeClientRunSink()
    otel_span = FakeOtelSpanSink()
    raw_dict = FakeRawDictSink()
    tracing = TracingConfig(
        enabled=True, output_dir=str(tmp_path),
        sinks=[client_run, otel_span, raw_dict],
    )
    cfg = ExecutionConfig(tracing=tracing)

    result = await Orchestra.run(
        task="secret", topology=_topology(), execution_config=cfg, max_steps=10,
    )
    assert result.success

    def _tool_span_args(records: List[Dict[str, Any]], path: str = "attributes") -> List[Dict[str, Any]]:
        # Find tool spans by run_type/kind, return their attributes dict.
        result_dicts = []
        for r in records:
            attrs = r.get(path) if path in r else r.get("attributes", {})
            kind = r.get("run_type") or r.get("kind", "").lower()
            if kind == "tool":
                result_dicts.append(attrs)
        return result_dicts

    cr_tool_attrs = _tool_span_args(client_run.runs)
    os_tool_attrs = _tool_span_args(otel_span.spans)
    rd_tool_attrs = _tool_span_args(raw_dict.payloads)

    assert cr_tool_attrs and os_tool_attrs and rd_tool_attrs

    # The tool span's `arguments` field should have the api_key redacted.
    for attrs in (*cr_tool_attrs, *os_tool_attrs, *rd_tool_attrs):
        args = attrs.get("arguments")
        if args is None:
            continue  # config.include_tool_results may be False; not a leak
        assert "sk-LEAK" not in json.dumps(args), (
            f"raw secret leaked into tool span attributes: {args}"
        )
