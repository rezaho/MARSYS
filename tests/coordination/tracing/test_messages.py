"""Tests for Phase 3: content-addressed message store + full-input capture.

Covers:
- ``compute_message_hash`` determinism, UUID-strip, cross-instance stability.
- ``InMemoryMessageStore`` + ``FilesystemMessageStore`` roundtrip, idempotency,
  cross-trace dedup, atomic writes.
- ``build_input_messages_ref`` ref shape (history / base / patch).
- ``diff_history`` append-only optimization and divergent fallback.
- Collector integration: enabling tracing writes blobs and attaches
  ``input_messages_ref`` to the step span (always on; no opt-in flag).
- Branch fork inheritance: child branch's first step diffs against the
  parent branch's last history.
- Redaction: secrets in messages are scrubbed before hashing/storage.
"""

from __future__ import annotations

import asyncio
import json
import pathlib

import pytest

from marsys.coordination.event_bus import EventBus
from marsys.coordination.status.events import AgentMessagesPreparedEvent, AgentStartEvent, BranchEvent
from marsys.coordination.tracing.collector import TraceCollector
from marsys.coordination.tracing.config import TracingConfig
from marsys.coordination.tracing.events import ExecutionStartEvent
from marsys.coordination.tracing.messages import (
    FilesystemMessageStore,
    InMemoryMessageStore,
    MessageStore,
    build_input_messages_ref,
    compute_message_hash,
    diff_history,
)


# ── compute_message_hash ──────────────────────────────────────────────────


class TestComputeMessageHash:
    def test_deterministic(self):
        m = {"role": "user", "content": "hi"}
        assert compute_message_hash(m) == compute_message_hash(m)

    def test_uuid_excluded(self):
        a = {"role": "user", "content": "hi", "message_id": "uuid-1"}
        b = {"role": "user", "content": "hi", "message_id": "uuid-2"}
        assert compute_message_hash(a) == compute_message_hash(b)

    def test_content_change_changes_hash(self):
        a = {"role": "user", "content": "hi"}
        b = {"role": "user", "content": "bye"}
        assert compute_message_hash(a) != compute_message_hash(b)

    def test_key_order_independent(self):
        a = {"role": "user", "content": "hi"}
        b = {"content": "hi", "role": "user"}
        assert compute_message_hash(a) == compute_message_hash(b)

    def test_unicode_safe(self):
        m = {"role": "user", "content": "héllo 你好"}
        # Just verify no encoding error and it's stable.
        assert compute_message_hash(m) == compute_message_hash(m)


# ── diff_history ──────────────────────────────────────────────────────────


class TestDiffHistory:
    def test_first_step_full_add(self):
        patch = diff_history(None, ["a", "b", "c"])
        assert patch == [
            {"op": "add", "path": "/-", "value": "a"},
            {"op": "add", "path": "/-", "value": "b"},
            {"op": "add", "path": "/-", "value": "c"},
        ]

    def test_append_only_single_add(self):
        patch = diff_history(["a", "b"], ["a", "b", "c"])
        assert patch == [{"op": "add", "path": "/-", "value": "c"}]

    def test_divergent_replace(self):
        patch = diff_history(["a", "b"], ["x", "y", "z"])
        # Non-append divergence → single replace op.
        assert len(patch) == 1
        assert patch[0]["op"] == "replace"
        assert patch[0]["value"] == ["x", "y", "z"]


# ── InMemoryMessageStore ──────────────────────────────────────────────────


class TestInMemoryMessageStore:
    def test_roundtrip(self):
        store = InMemoryMessageStore()
        m = {"role": "user", "content": "hi"}
        h = store.write_blob(m)
        assert store.read_blob(h) == m

    def test_idempotent_write(self):
        store = InMemoryMessageStore()
        m = {"role": "user", "content": "hi"}
        h1 = store.write_blob(m)
        h2 = store.write_blob(m)
        assert h1 == h2
        assert store.blob_count == 1

    def test_dedup_across_messages(self):
        store = InMemoryMessageStore()
        a = {"role": "user", "content": "hi", "message_id": "u1"}
        b = {"role": "user", "content": "hi", "message_id": "u2"}
        store.write_blob(a)
        store.write_blob(b)
        # Same content, different ids → one blob.
        assert store.blob_count == 1

    def test_missing_blob_returns_none(self):
        assert InMemoryMessageStore().read_blob("does-not-exist") is None

    def test_reconstruct_via_ref(self):
        store = InMemoryMessageStore()
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
        ref = build_input_messages_ref(msgs, store=store, prev_history=None)
        result = store.reconstruct(ref)
        assert result == msgs

    def test_caller_mutation_does_not_corrupt_store(self):
        """Writing a message and then mutating the input dict must not affect the store."""
        store = InMemoryMessageStore()
        m = {"role": "user", "content": "hi"}
        h = store.write_blob(m)
        m["content"] = "mutated"
        assert store.read_blob(h)["content"] == "hi"


# ── FilesystemMessageStore ────────────────────────────────────────────────


class TestFilesystemMessageStore:
    def test_roundtrip_on_disk(self, tmp_path):
        store = FilesystemMessageStore(base_dir=tmp_path)
        m = {"role": "user", "content": "hi"}
        h = store.write_blob(m)
        # File present.
        path = tmp_path / "messages" / f"{h}.json"
        assert path.exists()
        assert json.loads(path.read_text(encoding="utf-8")) == m
        # Read back via the API.
        assert store.read_blob(h) == m

    def test_idempotent_write_no_rewrite(self, tmp_path):
        store = FilesystemMessageStore(base_dir=tmp_path)
        m = {"role": "user", "content": "hi"}
        store.write_blob(m)
        path = list((tmp_path / "messages").glob("*.json"))[0]
        first_mtime = path.stat().st_mtime_ns
        # Second write should be a no-op (cheap stat, no rewrite).
        store.write_blob(m)
        assert path.stat().st_mtime_ns == first_mtime

    def test_dir_lazily_created(self, tmp_path):
        # Creating the store doesn't touch disk.
        store = FilesystemMessageStore(base_dir=tmp_path)
        messages_dir = tmp_path / "messages"
        assert not messages_dir.exists()
        # First write creates it.
        store.write_blob({"role": "user", "content": "hi"})
        assert messages_dir.exists()

    def test_dedup_across_traces(self, tmp_path):
        """Two separate stores rooted at the same dir share blobs (cross-trace dedup)."""
        store_a = FilesystemMessageStore(base_dir=tmp_path)
        store_b = FilesystemMessageStore(base_dir=tmp_path)
        m = {"role": "system", "content": "shared system prompt"}
        h_a = store_a.write_blob(m)
        h_b = store_b.write_blob(m)
        assert h_a == h_b
        # On disk: exactly one file.
        files = list((tmp_path / "messages").glob("*.json"))
        assert len(files) == 1


# ── Collector integration ────────────────────────────────────────────────


def _open_step(bus, *, session_id, branch_id, agent_name, step_number, step_span_id):
    """Helper: emit AgentStartEvent so collector opens the step span."""
    return bus.emit(AgentStartEvent(
        session_id=session_id,
        branch_id=branch_id,
        agent_name=agent_name,
        request_summary="r",
        step_number=step_number,
        step_span_id=step_span_id,
    ))


class TestCollectorFullInputCapture:
    @pytest.mark.asyncio
    async def test_capture_on_by_default(self, tmp_path):
        """Tracing enabled → full-input capture is automatic (Option A): the
        step span gets an input_messages_ref and blobs are written."""
        bus = EventBus()
        cfg = TracingConfig(enabled=True, output_dir=str(tmp_path))
        collector = TraceCollector(event_bus=bus, config=cfg, sinks=[])

        await bus.emit(ExecutionStartEvent(session_id="s", task_summary="t",
                                           topology_summary={}, agent_names=["A"], config_summary={}))
        await _open_step(bus, session_id="s", branch_id=None,
                         agent_name="A", step_number=1, step_span_id="step-1")
        await bus.emit(AgentMessagesPreparedEvent(
            session_id="s", agent_name="A",
            step_number=1, step_span_id="step-1",
            messages=[{"role": "user", "content": "captured"}],
        ))
        await asyncio.sleep(0.05)

        span = collector.step_spans["step-1"]
        assert "input_messages_ref" in span.attributes
        # The dedup sidecar directory is created on first blob write.
        assert (tmp_path / "messages").exists()

    @pytest.mark.asyncio
    async def test_capture_attaches_ref_and_writes_blobs(self, tmp_path):
        bus = EventBus()
        cfg = TracingConfig(
            enabled=True, output_dir=str(tmp_path),
        )
        collector = TraceCollector(event_bus=bus, config=cfg, sinks=[])

        await bus.emit(ExecutionStartEvent(session_id="s", task_summary="t",
                                           topology_summary={}, agent_names=["A"], config_summary={}))
        await _open_step(bus, session_id="s", branch_id="branch-1",
                         agent_name="A", step_number=1, step_span_id="step-1")
        await bus.emit(AgentMessagesPreparedEvent(
            session_id="s", branch_id="branch-1",
            agent_name="A", step_number=1, step_span_id="step-1",
            messages=[
                {"role": "system", "content": "you are X"},
                {"role": "user", "content": "hi"},
            ],
        ))
        await asyncio.sleep(0.05)

        span = collector.step_spans["step-1"]
        ref = span.attributes.get("input_messages_ref")
        assert ref is not None
        assert "history" in ref and "base" in ref and "patch" in ref
        assert len(ref["history"]) == 2
        # First step on a branch → base is None.
        assert ref["base"] is None
        # Two add ops in the patch (one per message).
        assert len(ref["patch"]) == 2
        # Blobs persisted on disk.
        files = list((tmp_path / "messages").glob("*.json"))
        assert len(files) == 2

    @pytest.mark.asyncio
    async def test_event_messages_nulled_after_handler(self, tmp_path):
        """Heavy-payload mitigation: the bus retains the event shell with messages=None."""
        bus = EventBus()
        cfg = TracingConfig(
            enabled=True, output_dir=str(tmp_path),
        )
        collector = TraceCollector(event_bus=bus, config=cfg, sinks=[])

        await bus.emit(ExecutionStartEvent(session_id="s", task_summary="t",
                                           topology_summary={}, agent_names=["A"], config_summary={}))
        await _open_step(bus, session_id="s", branch_id="b1",
                         agent_name="A", step_number=1, step_span_id="step-1")
        event = AgentMessagesPreparedEvent(
            session_id="s", branch_id="b1",
            agent_name="A", step_number=1, step_span_id="step-1",
            messages=[{"role": "user", "content": "hi"}],
        )
        await bus.emit(event)
        await asyncio.sleep(0.05)

        # Heavy field released — the bus retains the event shell but the
        # underlying list of dicts is garbage-collectable.
        assert event.messages is None
        # The ref still landed on the span, so the data made it to the store.
        span = collector.step_spans["step-1"]
        assert "input_messages_ref" in span.attributes

    @pytest.mark.asyncio
    async def test_redaction_applied_before_hashing(self, tmp_path):
        bus = EventBus()
        cfg = TracingConfig(
            enabled=True, output_dir=str(tmp_path),
        )
        collector = TraceCollector(event_bus=bus, config=cfg, sinks=[])

        original_messages = [
            {"role": "user", "content": "use api_key=sk-secret"},
            {"role": "tool", "content": {"api_key": "sk-leak", "data": "ok"}},
        ]
        await bus.emit(ExecutionStartEvent(session_id="s", task_summary="t",
                                           topology_summary={}, agent_names=["A"], config_summary={}))
        await _open_step(bus, session_id="s", branch_id="branch-1",
                         agent_name="A", step_number=1, step_span_id="step-1")
        await bus.emit(AgentMessagesPreparedEvent(
            session_id="s", branch_id="branch-1",
            agent_name="A", step_number=1, step_span_id="step-1",
            messages=original_messages,
        ))
        await asyncio.sleep(0.05)

        # Original input dict is untouched (deep-copy preserved working memory).
        assert original_messages[1]["content"]["api_key"] == "sk-leak"

        # On disk: the api_key value is redacted.
        for path in (tmp_path / "messages").glob("*.json"):
            content = json.loads(path.read_text(encoding="utf-8"))
            payload = json.dumps(content)
            assert "sk-leak" not in payload

    @pytest.mark.asyncio
    async def test_dedup_across_steps_same_branch(self, tmp_path):
        """A repeated system prompt across two steps stores one blob."""
        bus = EventBus()
        cfg = TracingConfig(
            enabled=True, output_dir=str(tmp_path),
        )
        collector = TraceCollector(event_bus=bus, config=cfg, sinks=[])

        sys_msg = {"role": "system", "content": "you are X"}
        await bus.emit(ExecutionStartEvent(session_id="s", task_summary="t",
                                           topology_summary={}, agent_names=["A"], config_summary={}))
        await _open_step(bus, session_id="s", branch_id="b1",
                         agent_name="A", step_number=1, step_span_id="step-1")
        await bus.emit(AgentMessagesPreparedEvent(
            session_id="s", branch_id="b1",
            agent_name="A", step_number=1, step_span_id="step-1",
            messages=[sys_msg, {"role": "user", "content": "hi"}],
        ))
        await _open_step(bus, session_id="s", branch_id="b1",
                         agent_name="A", step_number=2, step_span_id="step-2")
        await bus.emit(AgentMessagesPreparedEvent(
            session_id="s", branch_id="b1",
            agent_name="A", step_number=2, step_span_id="step-2",
            messages=[
                sys_msg,
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "more"},
            ],
        ))
        await asyncio.sleep(0.05)

        # Step 2 should see step 1's history as base, with the patch
        # being two add ops (assistant turn + user turn).
        ref1 = collector.step_spans["step-1"].attributes["input_messages_ref"]
        ref2 = collector.step_spans["step-2"].attributes["input_messages_ref"]
        assert ref2["base"] is not None
        assert ref2["base"] != ""
        # Append-only — patch is only the 2 new turns.
        assert all(op["op"] == "add" for op in ref2["patch"])
        assert len(ref2["patch"]) == 2
        # First two history entries are shared between steps.
        assert ref2["history"][:2] == ref1["history"]

    @pytest.mark.asyncio
    async def test_fork_inherits_parent_history_for_same_agent_only(self, tmp_path):
        """Per-(branch, agent) inheritance: only the SAME agent reruning on the
        child branch inherits its parent-branch tail. Cross-agent forks start
        fresh because each agent has its own conversation memory.
        """
        bus = EventBus()
        cfg = TracingConfig(
            enabled=True, output_dir=str(tmp_path),
        )
        collector = TraceCollector(event_bus=bus, config=cfg, sinks=[])

        sys_msg = {"role": "system", "content": "shared"}
        user_msg = {"role": "user", "content": "ask"}

        await bus.emit(ExecutionStartEvent(session_id="s", task_summary="t",
                                           topology_summary={}, agent_names=["A", "B"], config_summary={}))

        # Parent branch runs agent A.
        from marsys.coordination.events import BranchCreatedEvent
        await bus.emit(BranchCreatedEvent(
            session_id="s", branch_id="parent-branch", branch_name="parent",
            source_agent="root", target_agents=["A"], trigger_type="root",
        ))
        await _open_step(bus, session_id="s", branch_id="parent-branch",
                         agent_name="A", step_number=1, step_span_id="parent-step-1")
        await bus.emit(AgentMessagesPreparedEvent(
            session_id="s", branch_id="parent-branch",
            agent_name="A", step_number=1, step_span_id="parent-step-1",
            messages=[sys_msg, user_msg],
        ))
        await asyncio.sleep(0.05)

        # Child branch (fork). Both agents A (continues) and B (cross-agent)
        # will run on it. Inheritance only applies to A.
        await bus.emit(BranchCreatedEvent(
            session_id="s", branch_id="child-branch", branch_name="child",
            source_agent="A", target_agents=["A", "B"], trigger_type="invoke",
            parent_branch_id="parent-branch",
        ))

        # Child step 1 — agent A continues. Should inherit parent A's history
        # and produce an append-only patch with the new turn.
        await _open_step(bus, session_id="s", branch_id="child-branch",
                         agent_name="A", step_number=1, step_span_id="child-A1")
        await bus.emit(AgentMessagesPreparedEvent(
            session_id="s", branch_id="child-branch",
            agent_name="A", step_number=1, step_span_id="child-A1",
            messages=[sys_msg, user_msg, {"role": "user", "content": "continued"}],
        ))

        # Child step 2 — agent B runs for the first time. Should NOT inherit
        # A's history (different agents have different conversations).
        await _open_step(bus, session_id="s", branch_id="child-branch",
                         agent_name="B", step_number=2, step_span_id="child-B1")
        await bus.emit(AgentMessagesPreparedEvent(
            session_id="s", branch_id="child-branch",
            agent_name="B", step_number=2, step_span_id="child-B1",
            messages=[{"role": "system", "content": "B's system"},
                      {"role": "user", "content": "B's first turn"}],
        ))
        await asyncio.sleep(0.05)

        parent_ref = collector.step_spans["parent-step-1"].attributes["input_messages_ref"]
        child_a_ref = collector.step_spans["child-A1"].attributes["input_messages_ref"]
        child_b_ref = collector.step_spans["child-B1"].attributes["input_messages_ref"]

        # Same-agent fork: A inherits parent's history; first 2 hashes match.
        assert child_a_ref["history"][:2] == parent_ref["history"]
        assert all(op["op"] == "add" for op in child_a_ref["patch"])
        assert len(child_a_ref["patch"]) == 1  # just the new turn
        assert child_a_ref["base"] is not None

        # Cross-agent fork: B starts fresh; no inheritance from A.
        assert child_b_ref["base"] is None
        # All ops are 'add' because there's no prior history (full populate).
        assert all(op["op"] == "add" for op in child_b_ref["patch"])
        assert len(child_b_ref["patch"]) == 2

    @pytest.mark.asyncio
    async def test_different_agent_in_same_branch_chains_per_agent(self, tmp_path):
        """Sequential A→B→A on a single branch_id keeps per-agent diff chains.

        Without per-(branch, agent) keying, B's history overwrites A's anchor
        and A's second step produces a non-prefix replace patch instead of
        the correct append-only adds. Regression test for the live test's
        ``same_branch_diff_chain`` failure.
        """
        bus = EventBus()
        cfg = TracingConfig(
            enabled=True, output_dir=str(tmp_path),
        )
        collector = TraceCollector(event_bus=bus, config=cfg, sinks=[])

        sys_msg = {"role": "system", "content": "system"}
        await bus.emit(ExecutionStartEvent(session_id="s", task_summary="t",
                                           topology_summary={}, agent_names=["A", "B"], config_summary={}))

        # Step 1 — agent A on branch BR.
        await _open_step(bus, session_id="s", branch_id="BR",
                         agent_name="A", step_number=1, step_span_id="step-A1")
        await bus.emit(AgentMessagesPreparedEvent(
            session_id="s", branch_id="BR",
            agent_name="A", step_number=1, step_span_id="step-A1",
            messages=[sys_msg, {"role": "user", "content": "ask A"}],
        ))

        # Step 2 — agent B on the same branch BR (sequential invoke).
        await _open_step(bus, session_id="s", branch_id="BR",
                         agent_name="B", step_number=2, step_span_id="step-B1")
        await bus.emit(AgentMessagesPreparedEvent(
            session_id="s", branch_id="BR",
            agent_name="B", step_number=2, step_span_id="step-B1",
            messages=[{"role": "system", "content": "B's system"},
                      {"role": "user", "content": "ask B"}],
        ))

        # Step 3 — agent A returns on the same branch. Its history extends
        # step 1 (NOT step 2 — A's memory is per-A, doesn't see B's turns).
        await _open_step(bus, session_id="s", branch_id="BR",
                         agent_name="A", step_number=3, step_span_id="step-A2")
        await bus.emit(AgentMessagesPreparedEvent(
            session_id="s", branch_id="BR",
            agent_name="A", step_number=3, step_span_id="step-A2",
            messages=[
                sys_msg,
                {"role": "user", "content": "ask A"},
                {"role": "assistant", "content": "A's first reply"},
                {"role": "user", "content": "B's answer relayed back"},
            ],
        ))
        await asyncio.sleep(0.05)

        ref_a1 = collector.step_spans["step-A1"].attributes["input_messages_ref"]
        ref_b1 = collector.step_spans["step-B1"].attributes["input_messages_ref"]
        ref_a2 = collector.step_spans["step-A2"].attributes["input_messages_ref"]

        # A's first step: no prior history, base=None.
        assert ref_a1["base"] is None
        # B's first step on BR: still no prior B-history (different agent).
        assert ref_b1["base"] is None
        # A's second step: anchors on A's first step (NOT on B's). Append-only.
        assert ref_a2["base"] is not None
        assert all(op["op"] == "add" for op in ref_a2["patch"]), (
            f"Expected append-only patch (proves diff anchors against A's own "
            f"history, not B's). Got: {ref_a2['patch']}"
        )
        # The first two history hashes match A's step 1.
        assert ref_a2["history"][:2] == ref_a1["history"]

    @pytest.mark.asyncio
    async def test_event_count_equals_dispatch_count(self, tmp_path):
        """One AgentMessagesPreparedEvent per model dispatch (Acceptance Part A.3)."""
        bus = EventBus()
        cfg = TracingConfig(
            enabled=True, output_dir=str(tmp_path),
        )
        collector = TraceCollector(event_bus=bus, config=cfg, sinks=[])

        await bus.emit(ExecutionStartEvent(session_id="s", task_summary="t",
                                           topology_summary={}, agent_names=["A"], config_summary={}))
        # Three steps simulate three model dispatches.
        for n in (1, 2, 3):
            sid = f"step-{n}"
            await _open_step(bus, session_id="s", branch_id="b1",
                             agent_name="A", step_number=n, step_span_id=sid)
            await bus.emit(AgentMessagesPreparedEvent(
                session_id="s", branch_id="b1",
                agent_name="A", step_number=n, step_span_id=sid,
                messages=[{"role": "user", "content": f"step {n}"}],
            ))
        await asyncio.sleep(0.05)

        emitted = [
            e for e in bus.events
            if e.__class__.__name__ == "AgentMessagesPreparedEvent"
        ]
        assert len(emitted) == 3
        # And every step span has its ref attached.
        for n in (1, 2, 3):
            assert "input_messages_ref" in collector.step_spans[f"step-{n}"].attributes

    @pytest.mark.asyncio
    async def test_run_context_plumbs_step_span_id(self, tmp_path):
        """Regression: step_span_id must reach the agent so AgentMessagesPreparedEvent
        keys into the right span. Originally caught by the reviewer; this test
        fails if the executor's run_context dict drops step_span_id again.
        """
        from marsys.coordination.execution.step_executor import StepExecutor

        # Simulate the executor's run_context construction; assert step_span_id
        # is present (the field the agent reads to populate _step_context).
        # We don't need a full run — just construct the executor and inspect
        # the dict shape via a focused unit-style check.
        import inspect
        src = inspect.getsource(StepExecutor.execute_step)
        assert '"step_span_id": step_span_id' in src, (
            "step_executor's run_context dict must include step_span_id so the "
            "agent can stamp tracing events with the correct span. If this "
            "assertion fails, AgentMessagesPreparedEvent emissions silently "
            "land on no span (collector.step_spans.get(None) returns None)."
        )

    @pytest.mark.asyncio
    async def test_ndjson_roundtrip_after_rework(self, tmp_path):
        """Existing NDJSONTraceReader parses traces with the new ref attribute."""
        from marsys.coordination.tracing.writers.ndjson_writer import NDJSONTraceWriter
        from marsys.coordination.tracing.readers.ndjson_reader import NDJSONTraceReader

        bus = EventBus()
        cfg = TracingConfig(
            enabled=True, output_dir=str(tmp_path),
        )
        writer = NDJSONTraceWriter(cfg)
        collector = TraceCollector(event_bus=bus, config=cfg, sinks=[writer])

        await bus.emit(ExecutionStartEvent(session_id="s", task_summary="t",
                                           topology_summary={}, agent_names=["A"], config_summary={}))
        await _open_step(bus, session_id="s", branch_id="b1",
                         agent_name="A", step_number=1, step_span_id="step-1")
        await bus.emit(AgentMessagesPreparedEvent(
            session_id="s", branch_id="b1",
            agent_name="A", step_number=1, step_span_id="step-1",
            messages=[{"role": "user", "content": "hi"}],
        ))
        await collector.finalize("s")
        await writer.close()

        # Find the trace file and read it back.
        ndjson_files = list(pathlib.Path(tmp_path).glob("*.ndjson"))
        assert len(ndjson_files) == 1
        reader = NDJSONTraceReader(ndjson_files[0])
        tree = reader.to_tree()
        assert reader.completion_status == "complete"

        # Walk to find the step span and confirm the ref shape survived the round-trip.
        def find_step(span):
            if span.kind == "step":
                return span
            for c in span.children:
                got = find_step(c)
                if got is not None:
                    return got
            return None

        step = find_step(tree.root_span)
        assert step is not None
        ref = step.attributes.get("input_messages_ref")
        assert ref is not None
        assert "history" in ref and "base" in ref and "patch" in ref
