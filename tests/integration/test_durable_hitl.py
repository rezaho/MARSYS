"""Durable HITL — Orchestra public-surface round-trip (ADR-012).

The cornerstone (AC-31): produce a REAL durable suspend (a UserNode flagged
durable in the topology *spec*), write it to a FileStorageBackend via the same
``_snapshot_and_write`` helper ``Orchestra.execute()`` uses on its awaiting-user
exit, then resume it in a FRESH ``Orchestra`` via ``resume_session(user_response=…)``
— driving the resume agent through ``RealRuntime`` to a terminal ``OrchestraResult``.

Mirrors ``test_pause_resume.py``'s real-agent resume pattern. The suspend half
uses ``DeterministicRuntime`` (A ``SINGLE_INVOKE``s the durable ``User`` node); the
resume half uses a deterministic stub agent (overrides ``_run``, no model call).
The execute()-driven awaiting-user exit and cost-across-resume are the gated live
test ``test_durable_hitl_live.py``.
"""
from __future__ import annotations

import asyncio
import json as _json
import uuid as _uuid

import pytest

from marsys.coordination.config import ExecutionConfig
from marsys.coordination.execution.deterministic_runtime import DeterministicRuntime
from marsys.coordination.execution.orchestrator import Orchestrator
from marsys.coordination.execution.orchestrator_types import (
    ConvergencePolicy,
    StepResult,
    reset_ids,
)
from marsys.coordination.orchestra import Orchestra
from marsys.coordination.state import FileStorageBackend
from marsys.coordination.topology.core import Edge, Node, NodeKind, Topology


def _durable_spec() -> Topology:
    """Start → A → User(durable) → B → End, durability declared in the spec via
    the USER node's metadata (the workflow-definition path, founder option 2)."""
    return Topology(
        nodes=[
            Node(name="Start", kind=NodeKind.START),
            Node(name="A", kind=NodeKind.AGENT),
            Node(name="User", kind=NodeKind.USER, metadata={"durable": True}),
            Node(name="B", kind=NodeKind.AGENT),
            Node(name="End", kind=NodeKind.END),
        ],
        edges=[
            Edge(source="Start", target="A"),
            Edge(source="A", target="User"),
            Edge(source="User", target="B"),
            Edge(source="B", target="End"),
        ],
    )


def _stub_resume_agent_cls():
    """A deterministic resume agent 'B' that finalizes on its first tick — no
    model call (same pattern test_pause_resume.py's real-agent test uses)."""
    from marsys.agents import Agent
    from marsys.agents.memory import Message, ToolCallMsg
    from marsys.models import ModelConfig

    class _B(Agent):
        def __init__(self):
            super().__init__(
                model_config=ModelConfig(
                    type="api", name="mock-model", provider="openai", api_key="mock-key",
                ),
                goal="finish", instruction="Finish immediately.", name="B",
            )

        async def _run(self, messages, request_context, run_mode="default", **kwargs):
            cid = f"call_{_uuid.uuid4().hex[:8]}"
            return Message(
                role="assistant", content="done", name="B",
                tool_calls=[ToolCallMsg(
                    id=cid, call_id=cid, type="function",
                    name="return_final_response",
                    arguments=_json.dumps({"response": "resumed-ok"}),
                )],
            )

    return _B


def _ask_user_stub_cls():
    """A stub entry agent 'A' that emits an `ask_user` action on its first tick —
    drives Orchestra.execute() to the durable User node through RealRuntime with no
    model call (the validator maps the ask_user tool call → ASK_USER →
    SINGLE_INVOKE('User'), gated on the A→User edge)."""
    from marsys.agents import Agent
    from marsys.agents.memory import Message, ToolCallMsg
    from marsys.models import ModelConfig

    class _A(Agent):
        def __init__(self):
            super().__init__(
                model_config=ModelConfig(
                    type="api", name="mock-model", provider="openai", api_key="mock-key",
                ),
                goal="ask", instruction="Ask the user to re-authenticate.", name="A",
            )

        async def _run(self, messages, request_context, run_mode="default", **kwargs):
            cid = f"call_{_uuid.uuid4().hex[:8]}"
            return Message(
                role="assistant", content="asking", name="A",
                tool_calls=[ToolCallMsg(
                    id=cid, call_id=cid, type="function", name="ask_user",
                    arguments=_json.dumps({"question": "Please re-authenticate"}),
                )],
            )

    return _A


@pytest.mark.asyncio
async def test_execute_drives_durable_suspend_and_flags_metadata(tmp_path):
    """AC-1/2/4/5/6/25: a REAL RealRuntime dispatch through Orchestra.execute() —
    a stub agent emits ask_user on a spec-declared durable User node; execute()
    takes the awaiting-user exit, writes the snapshot ITSELF before returning, and
    flags both metadata observables; the run is not presented as finished."""
    from marsys.agents.registry import AgentRegistry

    reset_ids()
    backend = FileStorageBackend(tmp_path / "snapshots")
    sid = "exec-durable"
    AgentRegistry.clear()
    try:
        a = _ask_user_stub_cls()()
        AgentRegistry._test_agents = [a]
        orch = Orchestra(agent_registry=AgentRegistry, storage_backend=backend)
        out = await orch.execute(
            task="go", topology=_durable_spec(), context={"session_id": sid},
        )
        # AC-1/AC-2: the public metadata observables (not just the internal sentinel).
        assert out.metadata.get("paused") is True
        assert out.metadata.get("awaiting_user") is True
        # AC-6 intent: the run is suspended, not presented as a completed success.
        assert out.success is False
        assert out.final_response is None
        # AC-5: execute() itself wrote the snapshot before returning (no external pause).
        assert (tmp_path / "snapshots" / sid / "snapshot.json").is_file()
    finally:
        AgentRegistry.clear()


@pytest.mark.asyncio
async def test_pause_session_on_durable_parked_run_does_not_hang(tmp_path):
    """AC-21: a run that durably suspended itself via execute() does NOT leave
    pause_session hanging — the loop already exited at the durable boundary, so a
    subsequent pause_session is an idempotent no-op (it finds the snapshot already
    on disk), completing well within a timeout rather than blocking on a human."""
    from marsys.agents.registry import AgentRegistry

    reset_ids()
    backend = FileStorageBackend(tmp_path / "snapshots")
    sid = "exec-durable-pause"
    AgentRegistry.clear()
    try:
        a = _ask_user_stub_cls()()
        AgentRegistry._test_agents = [a]
        orch = Orchestra(agent_registry=AgentRegistry, storage_backend=backend)
        out = await orch.execute(
            task="go", topology=_durable_spec(), context={"session_id": sid},
        )
        assert out.metadata.get("awaiting_user") is True
        # The durable wait never blocked the loop, so pause_session can't hang on
        # it — it idempotent-no-ops on the already-written snapshot. (Pre-durable,
        # a UserNode-parked run blocked pause_session until the 300s timeout.)
        await asyncio.wait_for(orch.pause_session(sid), timeout=5.0)
    finally:
        AgentRegistry.clear()


async def _suspend_durable_to_disk(backend, sid, spec) -> Orchestra:
    """Drive a durable workload to its awaiting-user exit and persist the
    snapshot the way execute() does. Returns the pause-side Orchestra."""
    pause_orch = Orchestra(agent_registry=None, storage_backend=backend)
    pause_orch._build_topology_graph(spec, ExecutionConfig())
    # The shim-built durable UserNode has no handler bound, and needs none — the
    # durable path skips the SYNC handler guard (it never invokes the handler).
    rt = DeterministicRuntime()
    rt.queue_agent("A", StepResult(
        kind="SINGLE_INVOKE", next_agent="User", value="authenticate?",
    ))
    underlying = Orchestrator(pause_orch.topology_graph, rt, ConvergencePolicy())
    result = await underlying.run(task="go", entry_agent="A")
    assert result.error == "awaiting_user", f"did not durably suspend: {result.error}"
    await pause_orch._snapshot_and_write(sid, underlying)
    return pause_orch


@pytest.mark.asyncio
async def test_orchestra_durable_round_trip(tmp_path):
    """AC-5/12/14/25/31: a spec-declared durable user step suspends to disk via
    the Orchestra surface and resumes in a FRESH Orchestra with the response,
    driving the resume agent to a terminal success; the snapshot is discarded."""
    from marsys.agents.registry import AgentRegistry

    reset_ids()
    backend = FileStorageBackend(tmp_path / "snapshots")
    sid = "durable-rt"
    snap_path = tmp_path / "snapshots" / sid / "snapshot.json"

    AgentRegistry.clear()
    try:
        agent = _stub_resume_agent_cls()()      # auto-registers (weakref store)
        AgentRegistry._test_agents = [agent]    # keep a strong ref

        await _suspend_durable_to_disk(backend, sid, _durable_spec())
        assert snap_path.is_file()                                    # AC-5

        resume_orch = Orchestra(agent_registry=AgentRegistry, storage_backend=backend)
        out = await resume_orch.resume_session(
            sid, canonical_topology=_durable_spec(), user_response="AUTH_DONE",
        )

        assert out.success, f"durable resume did not complete: {out.error}"   # AC-12/14
        assert out.final_response == "resumed-ok"                    # the real agent ran
        assert out.metadata.get("resumed") is True
        assert not snap_path.is_file()                               # AC-31 discarded
    finally:
        AgentRegistry.clear()


@pytest.mark.asyncio
async def test_multi_consumer_round_trip_no_spren(tmp_path):
    """AC-30/31: the full durable round-trip from a plain consumer-style entry
    point — construct Orchestra(FileStorageBackend), durably suspend, construct a
    NEW Orchestra, resume_session(user_response=…), complete — with no Spren
    coupling. (The grep-for-imports check is AC-30 proper; this proves the
    consumer-style API path works standalone.)"""
    from marsys.agents.registry import AgentRegistry

    reset_ids()
    backend = FileStorageBackend(tmp_path / "snapshots")
    sid = "multi-consumer"

    AgentRegistry.clear()
    try:
        agent = _stub_resume_agent_cls()()
        AgentRegistry._test_agents = [agent]

        await _suspend_durable_to_disk(backend, sid, _durable_spec())

        # A brand-new Orchestra (the "process B" consumer) resumes from disk.
        consumer = Orchestra(agent_registry=AgentRegistry, storage_backend=backend)
        listed = await consumer.list_paused_sessions()
        assert any(m.session_id == sid for m in listed)

        out = await consumer.resume_session(
            sid, canonical_topology=_durable_spec(), user_response="done",
        )
        assert out.success, f"consumer-side resume failed: {out.error}"
    finally:
        AgentRegistry.clear()
