"""Durable human-in-the-loop suspend/resume — the orchestrator mechanism (ADR-012).

Deterministic coverage of the durable `ask_user`/`UserNode` path, driven through
the LIVE `Orchestrator` via `DeterministicRuntime` (the same pattern as
`tests/integration/test_pause_resume.py`, NOT the drifted simulator). These tests
exercise the mechanism without a model call:

  - durable `enqueue_user_interaction` records the in-flight interaction in the
    `pending_user_interaction` scalar and spawns NO `_drive` task (AC-1/AC-4);
  - the dispatch loop snapshots-and-exits with the `awaiting_user` sentinel
    rather than blocking on the in-memory queue (AC-3);
  - `snapshot()` captures the four in-flight fields and round-trips through
    `StateSnapshot` (AC-5/7/8/9/10), with old snapshots defaulting cleanly (AC-11);
  - restore → inject (via the existing `resume_branch_with_user_response` seam,
    which clears the scalar) → resume drives the resume_agent to terminal with
    the response as its input (AC-12/13/14);
  - a resumed run can re-suspend at a second durable interaction (AC-19);
  - `Orchestra.resume_session`'s argument contract (AC-16/17/18) and version lock
    (AC-26); and the workflow-definition `durable` plumbing through the shim
    (AC-24/25 — the spec → durable UserNode path).

The full `Orchestra.execute()` round-trip with real agents (and cost-across-resume)
is the gated live test in `tests/integration/test_durable_hitl_live.py`.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

import marsys
from marsys.coordination.execution.deterministic_runtime import DeterministicRuntime
from marsys.coordination.execution.det_nodes import UserNode
from marsys.coordination.execution.orchestrator import Orchestrator
from marsys.coordination.execution.orchestrator_types import (
    ConvergencePolicy,
    StepResult,
    reset_ids,
)
from marsys.coordination.orchestra import Orchestra
from marsys.coordination.state import (
    FileStorageBackend,
    IncompatibleSnapshotError,
    StateSnapshot,
)
from marsys.coordination.state.snapshot import UserInteractionState
from marsys.coordination.topology.core import Node, NodeKind, Topology

from tests.coordination.orchestrator._helpers import build_topology


# ─── Helpers ─────────────────────────────────────────────────────────────────

# A durable interaction never invokes the handler (it bypasses `_drive`); the
# UserNode only needs a non-None handler to pass on_single_invoke's guard.
_DUMMY_HANDLER = object()


def _durable_topology():
    """Start → A → User(durable) → B → End. A SINGLE_INVOKEs User; the resume
    agent (User's successor) is B."""
    return build_topology(
        nodes=[
            "Start", "A",
            UserNode("User", handler=_DUMMY_HANDLER, durable=True),
            "B", "End",
        ],
        flows=["Start -> A", "A -> User", "User -> B", "B -> End"],
    )


def _suspend_at_durable_user():
    """Run a fresh durable workload to its awaiting-user suspend and return
    (orchestrator, result). A is scripted to SINGLE_INVOKE the durable User
    node; the run snapshots-and-exits there."""
    reset_ids()
    topo = _durable_topology()
    runtime = DeterministicRuntime()
    runtime.queue_agent("A", StepResult(
        kind="SINGLE_INVOKE", next_agent="User", value="please authenticate",
    ))
    orch = Orchestrator(topo, runtime, ConvergencePolicy())
    return orch, topo, runtime


def _make_orchestra(tmp_path: Path) -> Orchestra:
    return Orchestra(
        agent_registry=type("R", (), {"get": staticmethod(lambda n: None),
                                      "clear": staticmethod(lambda: None)}),
        storage_backend=FileStorageBackend(tmp_path / "snapshots"),
    )


async def _write_snapshot(orch: Orchestra, sid: str, snapshot: StateSnapshot) -> None:
    await orch.storage_backend.write(
        f"{sid}/snapshot.json", snapshot.model_dump_json().encode("utf-8")
    )


def _durable_snapshot(sid: str, *, version: str | None = None,
                      pending: UserInteractionState | None) -> StateSnapshot:
    now = datetime.now(tz=timezone.utc)
    return StateSnapshot(
        framework_version=version or marsys.__version__,
        session_id=sid,
        topology_digest="x",
        created_at=now,
        paused_at=now,
        branches={}, barriers={}, convergence_barriers={},
        runnable=[], fire_queue=[], completed_emitted=[],
        user_interactions=[], user_interaction_inflight=pending is not None,
        pending_user_interaction=pending,
    )


# ─── Durable enqueue + dispatch-loop exit ────────────────────────────────────


@pytest.mark.asyncio
async def test_durable_enqueue_sets_scalar_and_spawns_no_drive():
    """AC-1 + AC-4: reaching a durable user interaction records the in-flight
    interaction in pending_user_interaction and spawns NO in-memory `_drive`
    (the resume queue is never created)."""
    orch, _topo, _rt = _suspend_at_durable_user()
    result = await orch.run(task="go", entry_agent="A")

    assert orch.pending_user_interaction is not None            # AC-1
    bid, prompt, resume_agent, target, durable = orch.pending_user_interaction
    assert durable is True                                      # the durable bit is carried
    assert prompt == "please authenticate"
    assert resume_agent == "B"                                  # User's successor
    assert orch._resume_user_responses is None                 # AC-4: no _drive/queue
    assert result.error == "awaiting_user"                     # AC-3 sentinel


@pytest.mark.asyncio
async def test_durable_run_returns_promptly_without_blocking():
    """AC-21 (no-hang): a durably-parked run does NOT block on an in-memory
    queue — run() returns the awaiting-user result rather than waiting for a
    human (provable by completing well within a tight timeout)."""
    import asyncio
    orch, _topo, _rt = _suspend_at_durable_user()
    result = await asyncio.wait_for(orch.run(task="go", entry_agent="A"), timeout=5.0)
    assert result.error == "awaiting_user"


# ─── Snapshot capture + round-trip ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_snapshot_captures_pending_and_round_trips(tmp_path):
    """AC-5/7/8/9/10: snapshot() captures the four in-flight fields, and the
    OrchestratorState ↔ StateSnapshot mapping round-trips them through JSON."""
    orch, _topo, _rt = _suspend_at_durable_user()
    await orch.run(task="go", entry_agent="A")

    state = orch.snapshot()
    assert state.pending_user_interaction is not None
    bid, prompt, resume_agent, target, _durable = state.pending_user_interaction

    # OrchestratorState → StateSnapshot (wire) via the Orchestra mapper (the path
    # pause_session / _snapshot_and_write use), then JSON → back.
    helper = _make_orchestra(tmp_path)
    helper.canonical_topology = type("T", (), {"nodes": [], "edges": []})()
    snap = helper._build_state_snapshot("sid", orch)

    assert snap.pending_user_interaction is not None
    assert snap.pending_user_interaction.suspended_branch_id == bid     # AC-7
    assert snap.pending_user_interaction.prompt == prompt               # AC-8
    assert snap.pending_user_interaction.resume_agent == resume_agent   # AC-9
    assert snap.pending_user_interaction.delivery_target == target      # AC-10

    # JSON round-trip → back to OrchestratorState preserves the 4-tuple.
    reloaded = StateSnapshot.model_validate_json(snap.model_dump_json())
    back = helper._snapshot_to_orchestrator_state(reloaded)
    assert back.pending_user_interaction == state.pending_user_interaction


def test_old_snapshot_without_pending_defaults_clean():
    """AC-11: a snapshot written before this feature (no pending field) loads
    and defaults pending_user_interaction to None under extra='forbid'."""
    now = datetime.now(tz=timezone.utc)
    raw = {
        "framework_version": marsys.__version__, "session_id": "old",
        "topology_digest": "x", "created_at": now.isoformat(),
        "paused_at": now.isoformat(), "branches": {}, "barriers": {},
        "convergence_barriers": {}, "runnable": [], "fire_queue": [],
        "completed_emitted": [], "user_interactions": [],
        "user_interaction_inflight": False,
    }
    import json
    snap = StateSnapshot.model_validate_json(json.dumps(raw))
    assert snap.pending_user_interaction is None


# ─── Restore → inject → resume to terminal ───────────────────────────────────


@pytest.mark.asyncio
async def test_full_durable_round_trip_to_terminal():
    """AC-12/13/14: restore a durable suspend into a fresh Orchestrator, inject
    the human's response via the existing seam, and drive the resume_agent to a
    terminal success — with the response delivered as the resume_agent's input."""
    orch1, topo, _rt1 = _suspend_at_durable_user()
    await orch1.run(task="go", entry_agent="A")
    state = orch1.snapshot()
    bid, _prompt, resume_agent, _target, _durable = state.pending_user_interaction

    # Fresh Orchestrator (simulating a restart), restore, inject, resume.
    rt2 = DeterministicRuntime()
    rt2.queue_agent("B", StepResult(kind="FINAL_RESPONSE", value="farewell"))
    orch2 = Orchestrator(_durable_topology(), rt2, ConvergencePolicy())
    orch2.restore_from(state)
    assert orch2.pending_user_interaction is not None            # restored

    orch2.resume_branch_with_user_response(bid, "AUTH_DONE", resume_agent)
    assert orch2.pending_user_interaction is None                # cleared on consume

    # AC-13: the resume_agent branch carries the response as its input.
    b_inputs = [br.input for br in orch2.branches.values() if br.current_agent == "B"]
    assert "AUTH_DONE" in b_inputs

    result = await orch2.resume()
    assert result.success                                        # AC-12/14 terminal


@pytest.mark.asyncio
async def test_resumed_run_can_resuspend_at_second_durable_interaction():
    """AC-19: a resumed run that reaches a SECOND durable interaction durably
    suspends again (the awaiting-user exit is reachable from resume(), not only
    the first run())."""
    reset_ids()
    topo = build_topology(
        nodes=[
            "Start", "A",
            UserNode("User", handler=_DUMMY_HANDLER, durable=True),
            "B",
            UserNode("User2", handler=_DUMMY_HANDLER, durable=True),
            "C", "End",
        ],
        flows=[
            "Start -> A", "A -> User", "User -> B", "B -> User2",
            "User2 -> C", "C -> End",
        ],
    )
    rt1 = DeterministicRuntime()
    rt1.queue_agent("A", StepResult(kind="SINGLE_INVOKE", next_agent="User", value="q1"))
    orch1 = Orchestrator(topo, rt1, ConvergencePolicy())
    r1 = await orch1.run(task="go", entry_agent="A")
    assert r1.error == "awaiting_user"
    state1 = orch1.snapshot()
    bid1, _p1, ra1, _t1, _d1 = state1.pending_user_interaction

    # Resume: B runs, SINGLE_INVOKEs the SECOND durable User node → re-suspends.
    rt2 = DeterministicRuntime()
    rt2.queue_agent("B", StepResult(kind="SINGLE_INVOKE", next_agent="User2", value="q2"))
    orch2 = Orchestrator(topo, rt2, ConvergencePolicy())
    orch2.restore_from(state1)
    orch2.resume_branch_with_user_response(bid1, "first-done", ra1)
    r2 = await orch2.resume()
    assert r2.error == "awaiting_user"                           # AC-19 re-suspend
    assert orch2.pending_user_interaction is not None
    assert orch2.pending_user_interaction[2] == "C"             # User2's successor

    # AC-19: the re-suspended run is itself resumable to terminal.
    state2 = orch2.snapshot()
    bid2, _p2, ra2, _t2, _d2 = state2.pending_user_interaction
    rt3 = DeterministicRuntime()
    rt3.queue_agent("C", StepResult(kind="FINAL_RESPONSE", value="all-done"))
    orch3 = Orchestrator(topo, rt3, ConvergencePolicy())
    orch3.restore_from(state2)
    orch3.resume_branch_with_user_response(bid2, "second-answer", ra2)
    r3 = await orch3.resume()
    assert r3.success                                            # second resume → terminal


# ─── Orchestra.resume_session argument contract + version lock ────────────────


@pytest.mark.asyncio
async def test_resume_session_pending_without_response_raises(tmp_path):
    """AC-17: resume_session WITHOUT user_response on a snapshot that HAS a
    pending durable interaction raises (a human-wait needs the answer)."""
    orch = _make_orchestra(tmp_path)
    sid = "pending-no-resp"
    pending = UserInteractionState(
        suspended_branch_id="b1", prompt="auth?", resume_agent="B",
        delivery_target="root", durable=True,
    )
    await _write_snapshot(orch, sid, _durable_snapshot(sid, pending=pending))
    with pytest.raises(ValueError, match="awaiting a user response"):
        await orch.resume_session(sid)


@pytest.mark.asyncio
async def test_resume_session_response_without_pending_raises(tmp_path):
    """AC-16: resume_session WITH user_response on a snapshot that has NO pending
    durable interaction raises (the response is not silently dropped)."""
    orch = _make_orchestra(tmp_path)
    sid = "resp-no-pending"
    await _write_snapshot(orch, sid, _durable_snapshot(sid, pending=None))
    with pytest.raises(ValueError, match="no pending durable"):
        await orch.resume_session(sid, user_response="surprise")


@pytest.mark.asyncio
async def test_resume_session_user_response_is_keyword_only(tmp_path):
    """AC-18: user_response is keyword-only — a positional fourth arg is a
    TypeError (the `*` gates it alongside FW17's params)."""
    orch = _make_orchestra(tmp_path)
    with pytest.raises(TypeError):
        await orch.resume_session("sid", None, None, "positional-response")


@pytest.mark.asyncio
async def test_resume_session_version_lock_on_durable_snapshot(tmp_path):
    """AC-26: a durable snapshot under a mismatched framework_version fails with
    IncompatibleSnapshotError (the version lock is inherited, not weakened)."""
    orch = _make_orchestra(tmp_path)
    sid = "durable-badver"
    pending = UserInteractionState(
        suspended_branch_id="b1", prompt="auth?", resume_agent="B",
        delivery_target="root", durable=True,
    )
    await _write_snapshot(
        orch, sid, _durable_snapshot(sid, version="0.0.0-nope", pending=pending),
    )
    with pytest.raises(IncompatibleSnapshotError):
        await orch.resume_session(sid, user_response="x")


# ─── Workflow-definition durable trigger (founder option 2) ──────────────────


def test_spec_durable_user_node_materializes_durable(tmp_path):
    """AC-24/25 (plumbing): a USER node carrying metadata['durable'] in the
    topology spec materializes a durable UserNode through the analyze+shim path
    that Orchestra.execute()/resume_session use for a spec-declared durable user
    step. Uses an explicit Start det-node so the analyzer takes the modern entry
    path (no legacy entry_point detection)."""
    from marsys.coordination.config import ExecutionConfig
    from marsys.coordination.topology.core import Edge
    canonical = Topology(
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
    orch = _make_orchestra(tmp_path)
    orch._build_topology_graph(canonical, ExecutionConfig())

    user_dets = [
        d for d in (orch.topology_graph.det_nodes or {}).values()
        if isinstance(d, UserNode)
    ]
    assert user_dets, "shim did not register a UserNode for the spec USER node"
    assert any(d.durable for d in user_dets), (
        "spec metadata['durable'] did not materialize a durable UserNode"
    )


def test_spec_user_node_without_durable_is_sync():
    """A spec USER node WITHOUT durable metadata materializes a non-durable
    (SYNC) UserNode — the flag is opt-in."""
    node = UserNode("User")
    assert node.durable is False


# ─── Directive-style trigger (FW18's escalate_to_user path) ──────────────────


def test_enqueue_user_interaction_durable_directly():
    """AC-23: the directive-style trigger — calling enqueue_user_interaction with
    durable=True DIRECTLY (no topology User node, as FW18's escalate_to_user does)
    records the in-flight durable interaction and spawns no in-memory wait."""
    reset_ids()
    topo = build_topology(nodes=["Start", "A", "End"], flows=["Start -> A", "A -> End"])
    orch = Orchestrator(topo, DeterministicRuntime(), ConvergencePolicy())
    orch.init_workflow(task="go", entry_agent="A")
    branch = next(iter(orch.branches.values()))

    orch.enqueue_user_interaction(branch, prompt="re-auth?", resume_agent="A", durable=True)

    assert orch.pending_user_interaction is not None
    bid, prompt, resume_agent, _target, durable = orch.pending_user_interaction
    assert (prompt, resume_agent, durable) == ("re-auth?", "A", True)
    assert orch._resume_user_responses is None        # no _drive/queue spawned
    assert branch.status == "WAITING"


def test_durable_sibling_re_arms_durably_not_sync():
    """ADR-012 FIFO durability (carry-the-bit guard): when a durable in-flight
    interaction is resolved and a DURABLE sibling was queued, the sibling re-arms
    DURABLY (it becomes the new durable pending, no in-memory _drive) — it does not
    silently revert to the SYNC path."""
    reset_ids()
    topo = build_topology(nodes=["Start", "A", "B", "End"], flows=["Start -> A", "A -> B", "B -> End"])
    orch = Orchestrator(topo, DeterministicRuntime(), ConvergencePolicy())
    orch.init_workflow(task="go", entry_agent="A")
    b1 = next(iter(orch.branches.values()))
    root = orch.root_barrier_id

    # One durable interaction in-flight, plus a second DURABLE branch queued as a
    # sibling (the multi-pending-durable case the deque must not downgrade).
    orch.enqueue_user_interaction(b1, prompt="q1", resume_agent="B", durable=True)
    b2 = orch._spawn(agent="B", input="x", delivery_target=root, parent_spawn=None)
    orch._user_interactions.append((b2.id, "q2", "B", root, True))

    # Resolve the in-flight one → the sibling is popped and re-dispatched.
    orch.resume_branch_with_user_response(b1.id, "answer1", "B")

    assert orch.pending_user_interaction is not None
    assert orch.pending_user_interaction[4] is True   # sibling re-armed DURABLY
    assert orch._resume_user_responses is None         # NOT a SYNC _drive


def test_no_spren_imports_in_framework():
    """AC-30: the framework has zero Spren coupling — no `from spren` / `import
    spren` anywhere under src/marsys (SP-018). Runs the grep the AC names."""
    root = Path(__file__).resolve().parents[2] / "src" / "marsys"
    hits = []
    for path in root.rglob("*.py"):
        for line in path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s.startswith("from spren") or s.startswith("import spren"):
                hits.append(f"{path}: {s}")
    assert not hits, f"Spren coupling found in the framework: {hits}"
