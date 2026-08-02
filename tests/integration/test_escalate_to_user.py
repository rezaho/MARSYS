"""Integration: the `escalate_to_user` control directive end to end (ADR-013).

A *granted* agent on a topology with NO `User` node emits `escalate_to_user` → the
run durably suspends (paused-awaiting-user) → `resume_session(user_response)`
resumes the EMITTING agent to terminal. The durable suspend/resume is framework 16,
reused unchanged: escalate produces the SAME `pending_user_interaction` shape and
routes into it without a topology `User` node.

Mirrors tests/integration/test_durable_hitl.py. The real emit→validate→translate
→route path is driven through `Orchestra.execute()` with a granted stub agent; the
resume half is driven from a scripted `ESCALATE_USER` step + a deterministic stub
agent (overrides `_run`, no model call). The full real round-trip with a live model
is the gated `test_escalate_to_user_live.py`.
"""
from __future__ import annotations

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


def _escalate_spec() -> Topology:
    """Start → A → End — NO User node. A (granted can_escalate) escalates."""
    return Topology(
        nodes=[
            Node(name="Start", kind=NodeKind.START),
            Node(name="A", kind=NodeKind.AGENT),
            Node(name="End", kind=NodeKind.END),
        ],
        edges=[
            Edge(source="Start", target="A"),
            Edge(source="A", target="End"),
        ],
    )


def _escalate_stub_cls(*, can_escalate: bool):
    """Stub agent 'A' that emits escalate_to_user on its first tick (no model call).
    Granted or not per `can_escalate` — the validator gate is what differs."""
    from marsys.agents import Agent
    from marsys.agents.memory import Message, ToolCallMsg
    from marsys.models import ModelConfig

    class _A(Agent):
        def __init__(self):
            super().__init__(
                model_config=ModelConfig(
                    type="api", name="mock-model", provider="openai", api_key="mock-key",
                ),
                goal="escalate", instruction="Escalate to the user.", name="A",
                can_escalate=can_escalate,
            )

        async def _run(self, messages, request_context, run_mode="default", **kwargs):
            cid = f"call_{_uuid.uuid4().hex[:8]}"
            return Message(
                role="assistant", content="escalating", name="A",
                tool_calls=[ToolCallMsg(
                    id=cid, call_id=cid, type="function", name="escalate_to_user",
                    arguments=_json.dumps({"prompt": "Please re-authenticate to example.com"}),
                )],
            )

    return _A


def _finalize_stub_cls():
    """Resume agent 'A' (the emitter) that finalizes on its first tick. It echoes
    whether the user_response reached it as input (AC-7): returns 'resumed-ok' only
    if it saw the re-auth marker, else 'resumed-NO-INPUT'."""
    from marsys.agents import Agent
    from marsys.agents.memory import Message, ToolCallMsg
    from marsys.models import ModelConfig

    class _A(Agent):
        def __init__(self):
            super().__init__(
                model_config=ModelConfig(
                    type="api", name="mock-model", provider="openai", api_key="mock-key",
                ),
                goal="finish", instruction="Finish immediately.", name="A",
                can_escalate=True,
            )

        async def _run(self, messages, request_context, run_mode="default", **kwargs):
            blob = repr(messages) + repr(request_context)
            saw_response = "REAUTH_DONE" in blob
            cid = f"call_{_uuid.uuid4().hex[:8]}"
            return Message(
                role="assistant", content="done", name="A",
                tool_calls=[ToolCallMsg(
                    id=cid, call_id=cid, type="function", name="return_final_response",
                    arguments=_json.dumps(
                        {"response": "resumed-ok" if saw_response else "resumed-NO-INPUT"}
                    ),
                )],
            )

    return _A


@pytest.mark.asyncio
async def test_execute_drives_escalate_suspend(tmp_path):
    """AC-1..AC-5: a REAL RealRuntime dispatch — a granted stub agent emits
    escalate_to_user on a User-LESS topology; execute() takes the awaiting-user
    exit, writes the snapshot itself, and flags both metadata observables; the run
    is not presented as finished. This is the case the has_edge_to_usernode gate
    forbids for ask_user."""
    from marsys.agents.registry import AgentRegistry

    reset_ids()
    backend = FileStorageBackend(tmp_path / "snapshots")
    sid = "exec-escalate"
    AgentRegistry.clear()
    try:
        a = _escalate_stub_cls(can_escalate=True)()
        AgentRegistry._test_agents = [a]
        orch = Orchestra(agent_registry=AgentRegistry, storage_backend=backend)
        out = await orch.execute(
            task="go", topology=_escalate_spec(), context={"session_id": sid},
        )
        assert out.metadata.get("paused") is True
        assert out.metadata.get("awaiting_user") is True
        assert out.success is False
        assert out.final_response is None
        assert (tmp_path / "snapshots" / sid / "snapshot.json").is_file()
    finally:
        AgentRegistry.clear()


@pytest.mark.asyncio
async def test_ungranted_escalate_does_not_suspend(tmp_path):
    """AC-3: an UNGRANTED agent emitting escalate_to_user is rejected by validation
    — the run does NOT durably suspend (no awaiting-user) and is not presented as a
    success."""
    from marsys.agents.registry import AgentRegistry

    reset_ids()
    backend = FileStorageBackend(tmp_path / "snapshots")
    sid = "exec-escalate-ungranted"
    AgentRegistry.clear()
    try:
        a = _escalate_stub_cls(can_escalate=False)()
        AgentRegistry._test_agents = [a]
        orch = Orchestra(agent_registry=AgentRegistry, storage_backend=backend)
        out = await orch.execute(
            task="go", topology=_escalate_spec(), context={"session_id": sid},
        )
        assert out.metadata.get("awaiting_user") is not True
        assert out.success is False
    finally:
        AgentRegistry.clear()


async def _suspend_escalate_to_disk(backend, sid, spec) -> Orchestra:
    """Drive an escalation to its awaiting-user exit via the ESCALATE_USER step and
    persist the snapshot the way execute() does. Returns the pause-side Orchestra.
    resume_agent is set by the orchestrator arm to the emitting agent (A)."""
    pause_orch = Orchestra(agent_registry=None, storage_backend=backend)
    pause_orch._build_topology_graph(spec, ExecutionConfig())
    rt = DeterministicRuntime()
    rt.queue_agent(
        "A", StepResult(kind="ESCALATE_USER", value="Please re-authenticate")
    )
    underlying = Orchestrator(pause_orch.topology_graph, rt, ConvergencePolicy())
    result = await underlying.run(task="go", entry_agent="A")
    assert result.error == "awaiting_user", f"did not durably suspend: {result.error}"
    await pause_orch._snapshot_and_write(sid, underlying)
    return pause_orch


@pytest.mark.asyncio
async def test_escalate_durable_round_trip(tmp_path):
    """AC-2/AC-6..AC-9: an ESCALATE_USER suspend persists to disk and resumes in a
    FRESH Orchestra via resume_session(user_response=…), driving the EMITTING agent
    (A) — not a successor — to terminal success with the response delivered as its
    input; the snapshot is discarded."""
    from marsys.agents.registry import AgentRegistry

    reset_ids()
    backend = FileStorageBackend(tmp_path / "snapshots")
    sid = "escalate-rt"
    snap_path = tmp_path / "snapshots" / sid / "snapshot.json"

    AgentRegistry.clear()
    try:
        agent = _finalize_stub_cls()()          # the emitter 'A', re-run on resume
        AgentRegistry._test_agents = [agent]

        await _suspend_escalate_to_disk(backend, sid, _escalate_spec())
        assert snap_path.is_file()                                    # AC-4

        resume_orch = Orchestra(agent_registry=AgentRegistry, storage_backend=backend)
        out = await resume_orch.resume_session(
            sid, canonical_topology=_escalate_spec(), user_response="REAUTH_DONE",
        )

        assert out.success, f"escalate resume did not complete: {out.error}"   # AC-2/AC-8
        # 'resumed-ok' only if the user_response reached the resumed agent (AC-7).
        assert out.final_response == "resumed-ok"
        assert out.metadata.get("resumed") is True
        assert not snap_path.is_file()                               # AC-9 (discarded)
    finally:
        AgentRegistry.clear()
