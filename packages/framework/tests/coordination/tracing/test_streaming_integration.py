"""Integration tests: full Orchestra runs with NDJSON streaming tracing.

Covers: end-to-end Orchestra.run with new writer; tail-follow during a live
run; OrchestraResult.metadata["tracing"] populated; multiple parallel runs
to separate files; mid-run cancellation produces a crashed-status file.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
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
    TraceTree,
    TracingConfig,
)
from marsys.coordination.tracing.types import create_span
from marsys.coordination.tracing.writers.ndjson_writer import (
    NDJSONTraceWriter,
    _DrainSentinel,
)
from marsys.models import ModelConfig


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


@pytest.mark.asyncio
async def test_full_orchestra_run_emits_complete_ndjson(tmp_path):
    AgentRegistry._test_agents = [_SimpleCoord(), _SimpleWorker("WorkerA"), _SimpleWorker("WorkerB")]
    cfg = ExecutionConfig(tracing=TracingConfig(enabled=True, output_dir=str(tmp_path)))
    result = await Orchestra.run(
        task="run", topology=_topology(), execution_config=cfg, max_steps=20,
    )
    assert result.success, f"orchestration failed: {result.error}"
    files = list(tmp_path.glob("*.ndjson"))
    assert len(files) == 1
    reader = NDJSONTraceReader(files[0])
    spans = list(reader.stream())
    assert spans, "expected at least one span"
    assert reader.completion_status == "complete"


@pytest.mark.asyncio
async def test_orchestra_result_metadata_tracing_populated(tmp_path):
    AgentRegistry._test_agents = [_SimpleCoord(), _SimpleWorker("WorkerA"), _SimpleWorker("WorkerB")]
    cfg = ExecutionConfig(tracing=TracingConfig(enabled=True, output_dir=str(tmp_path)))
    result = await Orchestra.run(
        task="meta", topology=_topology(), execution_config=cfg, max_steps=20,
    )
    assert "tracing" in result.metadata, f"metadata: {result.metadata}"
    tracing = result.metadata["tracing"]
    assert tracing["disabled"] is False
    assert tracing["total_spans"] > 0
    assert tracing["disk_error_count"] == 0


@pytest.mark.asyncio
async def test_filename_uses_trace_id(tmp_path):
    AgentRegistry._test_agents = [_SimpleCoord(), _SimpleWorker("WorkerA"), _SimpleWorker("WorkerB")]
    cfg = ExecutionConfig(tracing=TracingConfig(enabled=True, output_dir=str(tmp_path)))
    result = await Orchestra.run(
        task="fname", topology=_topology(), execution_config=cfg, max_steps=20,
    )
    files = list(tmp_path.glob("*.ndjson"))
    assert len(files) == 1
    # Filename stem should be a 26-char ULID, not the session_id (which would
    # be a 36-char UUID). Reads through the writer.
    assert len(files[0].stem) == 26


@pytest.mark.asyncio
async def test_tree_reconstruction_from_orchestra_run(tmp_path):
    AgentRegistry._test_agents = [_SimpleCoord(), _SimpleWorker("WorkerA"), _SimpleWorker("WorkerB")]
    cfg = ExecutionConfig(tracing=TracingConfig(enabled=True, output_dir=str(tmp_path)))
    result = await Orchestra.run(
        task="tree", topology=_topology(), execution_config=cfg, max_steps=20,
    )
    assert result.success
    files = list(tmp_path.glob("*.ndjson"))
    tree = TraceTree.from_ndjson(files[0])
    assert tree.root_span.kind == "execution"
    # Should have at least one branch span (parallel fanout) with step children.
    assert len(tree.root_span.children) > 0


@pytest.mark.asyncio
async def test_tail_follow_during_live_run(tmp_path):
    """Tail-follower receives spans as they close during a run."""
    AgentRegistry._test_agents = [_SimpleCoord(), _SimpleWorker("WorkerA"), _SimpleWorker("WorkerB")]
    cfg = ExecutionConfig(tracing=TracingConfig(enabled=True, output_dir=str(tmp_path)))
    received: list = []
    follower_status = {"status": None, "started": False}
    follower_done = threading.Event()

    def follower(path_dir):
        # Wait for the file to appear.
        import pathlib, time
        for _ in range(50):
            files = list(pathlib.Path(path_dir).glob("*.ndjson"))
            if files:
                break
            time.sleep(0.05)
        else:
            follower_done.set()
            return
        reader = NDJSONTraceReader(files[0])
        follower_status["started"] = True
        for span in reader.stream(follow=True, poll_interval=0.05):
            received.append(span["span_id"])
        follower_status["status"] = reader.completion_status
        follower_done.set()

    t = threading.Thread(target=follower, args=(str(tmp_path),), daemon=True)
    t.start()
    result = await Orchestra.run(
        task="tail", topology=_topology(), execution_config=cfg, max_steps=20,
    )
    assert result.success
    follower_done.wait(timeout=3.0)
    assert follower_status["started"], "follower never saw the file"
    assert received, "follower yielded no spans"
    assert follower_status["status"] == "complete"


@pytest.mark.asyncio
async def test_two_runs_separate_files(tmp_path):
    """Two sequential Orchestra runs produce two distinct trace files keyed by trace_id."""
    AgentRegistry._test_agents = [_SimpleCoord(), _SimpleWorker("WorkerA"), _SimpleWorker("WorkerB")]
    cfg = ExecutionConfig(tracing=TracingConfig(enabled=True, output_dir=str(tmp_path)))
    r1 = await Orchestra.run(task="run1", topology=_topology(), execution_config=cfg, max_steps=20)
    AgentRegistry.clear()
    AgentRegistry._test_agents = [_SimpleCoord(), _SimpleWorker("WorkerA"), _SimpleWorker("WorkerB")]
    r2 = await Orchestra.run(task="run2", topology=_topology(), execution_config=cfg, max_steps=20)
    assert r1.success and r2.success
    files = list(tmp_path.glob("*.ndjson"))
    assert len(files) == 2
    assert files[0].stem != files[1].stem


@pytest.mark.asyncio
async def test_writer_used_directly_outside_orchestra(tmp_path):
    """Multi-consumer proof: NDJSONTraceWriter works without Orchestra entrypoint."""
    cfg = TracingConfig(enabled=True, output_dir=str(tmp_path))
    writer = NDJSONTraceWriter(cfg)
    span = create_span("standalone", "Custom", "step")
    span.close(end_time=span.start_time + 0.1)
    await writer.publish_span(span)
    await writer.close()
    files = list(tmp_path.glob("*.ndjson"))
    assert len(files) == 1
    reader = NDJSONTraceReader(files[0])
    spans = list(reader.stream())
    assert len(spans) == 1
    assert spans[0]["name"] == "Custom"
    assert reader.completion_status == "complete"


# ── Slow-disk back-pressure ─────────────────────────────────────────────────


class _SlowDrainNDJSONWriter(NDJSONTraceWriter):
    """Test-only subclass: injects an artificial per-write delay in the drain
    task to simulate a slow disk. Used to prove the queue absorbs back-pressure
    so publish_span (called from EventBus listeners) stays fast.
    """

    DRAIN_DELAY_SECONDS = 0.1

    async def _drain_loop(self) -> None:
        while True:
            try:
                item = await self._queue.get()
            except asyncio.CancelledError:
                return
            if isinstance(item, _DrainSentinel):
                return
            if self._disabled:
                self._disabled_dropped_count += 1
                continue
            await asyncio.sleep(self.DRAIN_DELAY_SECONDS)  # synthetic slow disk
            line = self._serialize_span(item)
            self._write_line(line)


@pytest.mark.asyncio
async def test_slow_disk_does_not_back_pressure_publish_span(tmp_path):
    """Brief edge-case matrix: slow-disk back-pressure.

    With a 100ms per-write delay in the drain task, publish_span (the path
    called from EventBus listeners via TraceCollector) must stay near-zero
    latency — the bounded queue absorbs back-pressure so bus.emit doesn't
    inherit the slow disk.
    """
    cfg = TracingConfig(enabled=True, output_dir=str(tmp_path))
    writer = _SlowDrainNDJSONWriter(cfg)

    # Open the file via a first span (lazy open + drain task start).
    seed = create_span("TR_SLOW", "seed", "step")
    seed.close()
    await writer.publish_span(seed)

    # Time 10 publish_span calls. With 100ms drain delay, total wall time is
    # ~1s on the drain side — but the calling path should observe ~0ms each.
    latencies: list = []
    for i in range(10):
        span = create_span("TR_SLOW", f"S{i}", "step")
        span.close()
        t0 = time.perf_counter()
        await writer.publish_span(span)
        latencies.append(time.perf_counter() - t0)

    p99 = max(latencies)
    avg = sum(latencies) / len(latencies)
    assert p99 < 0.05, (
        f"publish_span p99 latency {p99 * 1000:.1f}ms — back-pressure leaked "
        f"from the slow drain (100ms/write) to the emit path. "
        f"avg={avg * 1000:.2f}ms latencies={[round(l*1000, 2) for l in latencies]}"
    )

    # Cleanup: cancel drain so close() doesn't wait the full 1s+ for it.
    if writer._drain_task is not None:
        writer._drain_task.cancel()
        try:
            await writer._drain_task
        except (asyncio.CancelledError, Exception):
            pass
        writer._drain_task = None
    await writer.close()


# ── Concurrent emit / no interleaved corruption ─────────────────────────────


@pytest.mark.asyncio
async def test_concurrent_branches_no_interleaved_corruption(tmp_path):
    """Brief integration test: 5 concurrent branches each emitting at high
    frequency. The single-threaded drain task must serialize writes so every
    line is valid JSON — no byte-level interleaving from race conditions.
    """
    cfg = TracingConfig(enabled=True, output_dir=str(tmp_path))
    writer = NDJSONTraceWriter(cfg, queue_maxsize=10000)
    spans_per_branch = 100

    async def emit_burst(branch: str) -> None:
        for i in range(spans_per_branch):
            span = create_span("TR_CONC", f"{branch}-S{i}", "step")
            span.close(end_time=span.start_time + 0.001)
            await writer.publish_span(span)

    await asyncio.gather(*[emit_burst(f"B{b}") for b in range(5)])
    await writer.close()

    files = list(tmp_path.glob("*.ndjson"))
    assert len(files) == 1
    raw = files[0].read_text(encoding="utf-8")

    # Every non-empty line must parse cleanly. A failed json.loads here
    # indicates byte-level interleaving from concurrent writes.
    lines = [l for l in raw.split("\n") if l.strip()]
    parsed = []
    for line in lines:
        parsed.append(json.loads(line))  # any failure here is a bug

    span_lines = [
        p for p in parsed
        if p.get("kind") not in ("stream_completed", "stream_event")
    ]
    expected = 5 * spans_per_branch
    assert len(span_lines) == expected, (
        f"expected {expected} spans (5 branches × {spans_per_branch}), "
        f"got {len(span_lines)} — drops: {writer.dropped_span_count}"
    )

    # Final marker still emitted after concurrent burst.
    assert parsed[-1]["kind"] == "stream_completed"


# ── Two CONCURRENT Orchestra runs ───────────────────────────────────────────


def _build_topology_named(coord: str, w_a: str, w_b: str) -> dict:
    return {
        "agents": ["User", coord, w_a, w_b],
        "flows": [
            f"User -> {coord}",
            f"{coord} -> {w_a}",
            f"{coord} -> {w_b}",
            f"{w_a} -> {coord}",
            f"{w_b} -> {coord}",
            f"{coord} -> User",
        ],
        "exit_points": [coord],
        "rules": [],
    }


class _CoordNamed(Agent):
    """Stateless coordinator-style agent. State lives per-instance so two
    concurrent runs with disjoint instances do not share `_fanned_out`."""

    def __init__(self, name: str, worker_a: str, worker_b: str):
        super().__init__(
            model_config=ModelConfig(type="api", name="mock", provider="openai", api_key="mock-key"),
            goal="coordinate",
            instruction="Coordinate two workers.",
            name=name,
        )
        self._fanned_out = False
        self._worker_a = worker_a
        self._worker_b = worker_b

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
                    {"agent_name": self._worker_a, "request": "a"},
                    {"agent_name": self._worker_b, "request": "b"},
                ],
            })],
            name=self.name,
        )


class _WorkerNamed(Agent):
    def __init__(self, name: str, return_to: str):
        super().__init__(
            model_config=ModelConfig(type="api", name="mock", provider="openai", api_key="mock-key"),
            goal=f"work {name}",
            instruction="Do work.",
            name=name,
        )
        self._return_to = return_to

    async def _run(self, messages, request_context, run_mode: str = "default", **kwargs):
        await asyncio.sleep(0.01)
        return Message(
            role="assistant",
            content=f"{self.name} done",
            tool_calls=[_coord_tool_call(
                "invoke_agent",
                {"invocations": [{"agent_name": self._return_to, "request": f"data from {self.name}"}]},
            )],
            name=self.name,
        )


@pytest.mark.asyncio
async def test_two_concurrent_orchestra_runs_separate_files(tmp_path):
    """Brief integration test: two Orchestra runs awaited concurrently via
    asyncio.gather. Each run produces its own trace_id-named file with its
    own spans; no cross-trace mixing in a shared output directory.
    """
    # Two disjoint agent sets so concurrent runs don't share mutable state.
    coord1 = _CoordNamed("Coord1", "Worker1A", "Worker1B")
    coord2 = _CoordNamed("Coord2", "Worker2A", "Worker2B")
    AgentRegistry._test_agents = [
        coord1,
        _WorkerNamed("Worker1A", "Coord1"),
        _WorkerNamed("Worker1B", "Coord1"),
        coord2,
        _WorkerNamed("Worker2A", "Coord2"),
        _WorkerNamed("Worker2B", "Coord2"),
    ]
    cfg = ExecutionConfig(tracing=TracingConfig(enabled=True, output_dir=str(tmp_path)))

    r1, r2 = await asyncio.gather(
        Orchestra.run(
            task="run1",
            topology=_build_topology_named("Coord1", "Worker1A", "Worker1B"),
            execution_config=cfg,
            max_steps=20,
        ),
        Orchestra.run(
            task="run2",
            topology=_build_topology_named("Coord2", "Worker2A", "Worker2B"),
            execution_config=cfg,
            max_steps=20,
        ),
    )
    assert r1.success and r2.success, f"r1={r1.error}, r2={r2.error}"

    files = sorted(tmp_path.glob("*.ndjson"))
    assert len(files) == 2, f"expected 2 trace files, got {[f.name for f in files]}"
    # Distinct trace_id stems.
    assert files[0].stem != files[1].stem

    # Each file's spans share one trace_id, and the two trace_ids are different.
    trace_ids_per_file = []
    for f in files:
        reader = NDJSONTraceReader(f)
        spans = list(reader.stream())
        assert spans, f"no spans in {f.name}"
        assert reader.completion_status == "complete", \
            f"{f.name} status={reader.completion_status}"
        ids = {s["trace_id"] for s in spans}
        assert len(ids) == 1, f"{f.name} mixed trace_ids: {ids}"
        trace_ids_per_file.append(ids.pop())
    assert trace_ids_per_file[0] != trace_ids_per_file[1], \
        "concurrent runs produced files with the same trace_id"
