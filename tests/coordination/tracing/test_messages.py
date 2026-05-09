"""Tests for Phase 3: content-addressed message store + full-input capture.

Covers:
- ``compute_message_hash`` determinism, UUID-strip, cross-instance stability.
- ``InMemoryMessageStore`` + ``FilesystemMessageStore`` roundtrip, idempotency,
  cross-trace dedup, atomic writes.
- ``build_input_messages_ref`` ref shape (history / base / patch).
- ``diff_history`` append-only optimization and divergent fallback.
- Collector integration: ``capture_full_input`` enables blob writes and
  attaches ``input_messages_ref`` to the step span.
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
from marsys.coordination.status.events import AgentStartEvent, BranchEvent
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


class TestCollectorFullInputCapture:
    @pytest.mark.asyncio
    async def test_capture_disabled_by_default(self, tmp_path):
        """capture_full_input=False → no input_messages_ref attribute, no messages dir."""
        bus = EventBus()
        cfg = TracingConfig(enabled=True, output_dir=str(tmp_path))
        collector = TraceCollector(event_bus=bus, config=cfg, sinks=[])

        await bus.emit(ExecutionStartEvent(session_id="s", task_summary="t",
                                           topology_summary={}, agent_names=["A"], config_summary={}))
        await bus.emit(AgentStartEvent(
            session_id="s", agent_name="A", request_summary="r",
            step_number=1, step_span_id="step-1",
            messages=[{"role": "user", "content": "ignored"}],  # event has data, but config is off
        ))
        await asyncio.sleep(0.05)

        span = collector.step_spans["step-1"]
        assert "input_messages_ref" not in span.attributes
        # No messages directory created.
        assert not (tmp_path / "messages").exists()

    @pytest.mark.asyncio
    async def test_capture_attaches_ref_and_writes_blobs(self, tmp_path):
        bus = EventBus()
        cfg = TracingConfig(
            enabled=True, output_dir=str(tmp_path), capture_full_input=True,
        )
        collector = TraceCollector(event_bus=bus, config=cfg, sinks=[])

        await bus.emit(ExecutionStartEvent(session_id="s", task_summary="t",
                                           topology_summary={}, agent_names=["A"], config_summary={}))
        await bus.emit(AgentStartEvent(
            session_id="s", branch_id="branch-1",
            agent_name="A", request_summary="r",
            step_number=1, step_span_id="step-1",
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
    async def test_redaction_applied_before_hashing(self, tmp_path):
        bus = EventBus()
        cfg = TracingConfig(
            enabled=True, output_dir=str(tmp_path), capture_full_input=True,
        )
        collector = TraceCollector(event_bus=bus, config=cfg, sinks=[])

        original_messages = [
            {"role": "user", "content": "use api_key=sk-secret"},
            {"role": "tool", "content": {"api_key": "sk-leak", "data": "ok"}},
        ]
        await bus.emit(ExecutionStartEvent(session_id="s", task_summary="t",
                                           topology_summary={}, agent_names=["A"], config_summary={}))
        await bus.emit(AgentStartEvent(
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
            enabled=True, output_dir=str(tmp_path), capture_full_input=True,
        )
        collector = TraceCollector(event_bus=bus, config=cfg, sinks=[])

        sys_msg = {"role": "system", "content": "you are X"}
        await bus.emit(ExecutionStartEvent(session_id="s", task_summary="t",
                                           topology_summary={}, agent_names=["A"], config_summary={}))
        await bus.emit(AgentStartEvent(
            session_id="s", branch_id="b1",
            agent_name="A", step_number=1, step_span_id="step-1",
            messages=[sys_msg, {"role": "user", "content": "hi"}],
        ))
        await bus.emit(AgentStartEvent(
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
    async def test_branch_fork_inherits_parent_history(self, tmp_path):
        """Child branch's first step diffs against parent's last history."""
        bus = EventBus()
        cfg = TracingConfig(
            enabled=True, output_dir=str(tmp_path), capture_full_input=True,
        )
        collector = TraceCollector(event_bus=bus, config=cfg, sinks=[])

        sys_msg = {"role": "system", "content": "shared"}
        user_msg = {"role": "user", "content": "ask"}

        await bus.emit(ExecutionStartEvent(session_id="s", task_summary="t",
                                           topology_summary={}, agent_names=["P", "C"], config_summary={}))

        # Parent branch step.
        # Use BranchCreatedEvent — but that's not in status events.
        # The collector uses _handle_branch_created which subscribes to
        # 'BranchCreatedEvent'. Let's emit one synthetically by direct call.
        from marsys.coordination.events import BranchCreatedEvent
        parent_branch = BranchCreatedEvent(
            session_id="s",
            branch_id="parent-branch",
            branch_name="parent",
            source_agent="root",
            target_agents=["P"],
            trigger_type="root",
        )
        await bus.emit(parent_branch)
        await bus.emit(AgentStartEvent(
            session_id="s", branch_id="parent-branch",
            agent_name="P", step_number=1, step_span_id="parent-step-1",
            messages=[sys_msg, user_msg],
        ))
        await asyncio.sleep(0.05)

        # Child branch — fork from parent.
        child_branch = BranchCreatedEvent(
            session_id="s",
            branch_id="child-branch",
            branch_name="child",
            source_agent="P",
            target_agents=["C"],
            trigger_type="invoke",
            parent_branch_id="parent-branch",
        )
        await bus.emit(child_branch)
        # Child's first step has the same prefix as parent + a new turn.
        await bus.emit(AgentStartEvent(
            session_id="s", branch_id="child-branch",
            agent_name="C", step_number=1, step_span_id="child-step-1",
            messages=[sys_msg, user_msg, {"role": "user", "content": "delegate"}],
        ))
        await asyncio.sleep(0.05)

        parent_ref = collector.step_spans["parent-step-1"].attributes["input_messages_ref"]
        child_ref = collector.step_spans["child-step-1"].attributes["input_messages_ref"]

        # Parent's history is the prefix for child's first step.
        assert child_ref["history"][:2] == parent_ref["history"]
        # Child's patch is just the new turn (one add op), not a full re-add.
        assert all(op["op"] == "add" for op in child_ref["patch"])
        assert len(child_ref["patch"]) == 1
        # Child base anchors on parent's history.
        assert child_ref["base"] is not None
