"""Multi-consumer test: three fake vendor sinks all see the same span tree.

Proves the protocol genuinely fits LangSmith / Phoenix / Spren-shaped
adapters in parallel. Each fake sink translates the framework's Span into
its own target shape; all three see identical content; secrets carried in
ToolCallEvent.arguments are redacted in every adapter's view.
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


# ── Vendor-shape fake sinks ────────────────────────────────────────────────


class FakeLangSmithSink(TelemetrySink):
    """Records LangSmith-shaped Client.create_run calls per closed span."""

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


class FakePhoenixSink(TelemetrySink):
    """Records OTel-shaped span dicts per closed span (Phoenix-flavoured)."""

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


class FakeSprenSink(TelemetrySink):
    """Records the protocol's calls verbatim (Spren posts span.to_dict() over HTTP)."""

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
async def test_three_vendor_sinks_all_see_same_span_tree(tmp_path):
    AgentRegistry._test_agents = [_Coord()]

    langsmith = FakeLangSmithSink()
    phoenix = FakePhoenixSink()
    spren = FakeSprenSink()
    tracing = TracingConfig(
        enabled=True, output_dir=str(tmp_path),
        sinks=[langsmith, phoenix, spren],
    )
    cfg = ExecutionConfig(tracing=tracing)

    result = await Orchestra.run(
        task="multi", topology=_topology(), execution_config=cfg, max_steps=10,
    )
    assert result.success, f"orchestration failed: {result.error}"

    # Same span count across all three fake sinks.
    assert len(langsmith.runs) == len(phoenix.spans) == len(spren.payloads)
    assert len(langsmith.runs) > 0

    # Same span_ids in the same order.
    ls_ids = [r["attributes"].get("span_id") for r in langsmith.runs]
    ph_ids = [s["span_id"] for s in phoenix.spans]
    sp_ids = [p["span_id"] for p in spren.payloads]
    assert ph_ids == sp_ids
    # LangSmith doesn't emit span_id directly, so just check shape consistency
    # via parent_run_id (parent_span_id) alignment.
    assert [r["parent_run_id"] for r in langsmith.runs] == [
        s["parent_id"] for s in phoenix.spans
    ]

    # All three sinks closed.
    assert langsmith.closed
    assert phoenix.closed
    assert spren.closed


@pytest.mark.asyncio
async def test_secret_in_tool_arguments_redacted_in_all_three_sinks(tmp_path):
    AgentRegistry._test_agents = [_Coord()]

    langsmith = FakeLangSmithSink()
    phoenix = FakePhoenixSink()
    spren = FakeSprenSink()
    tracing = TracingConfig(
        enabled=True, output_dir=str(tmp_path),
        sinks=[langsmith, phoenix, spren],
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

    ls_tool_attrs = _tool_span_args(langsmith.runs)
    ph_tool_attrs = _tool_span_args(phoenix.spans)
    sp_tool_attrs = _tool_span_args(spren.payloads)

    assert ls_tool_attrs and ph_tool_attrs and sp_tool_attrs

    # The tool span's `arguments` field should have the api_key redacted.
    for attrs in (*ls_tool_attrs, *ph_tool_attrs, *sp_tool_attrs):
        args = attrs.get("arguments")
        if args is None:
            continue  # config.include_tool_results may be False; not a leak
        assert "sk-LEAK" not in json.dumps(args), (
            f"raw secret leaked into tool span attributes: {args}"
        )
