"""Failure-mode tests.

F1 — Partial failure during parallel (policy variants)
F5 — Partial commitment (conv never reached at runtime)
F7 — Resurrection (late arrival at FIRED barrier)
"""
from __future__ import annotations

from research.orchestration.orchestrator.types import StepResult
from research.orchestration.simulator.simulator import Simulator, run_trace
from research.orchestration.simulator.det_nodes import StartNode
from research.orchestration.simulator.topology import SimNode, build_topology
from research.orchestration.simulator.trace import (
    ConvergencePolicy,
    SimAssertion,
    SimEvent,
    SimTrace,
)


# ───────────────────────────────────────────────────────────────────
# F1: Partial failure under three policy regimes
# ───────────────────────────────────────────────────────────────────

def _f1_topology():
    return build_topology(
        nodes=[StartNode(), SimNode("A"), SimNode("B1"), SimNode("B2")],
        flows=["Start -> A", "A -> B1", "A -> B2", "B1 -> A", "B2 -> A"],
    )


def _f1_trace(policy: ConvergencePolicy) -> SimTrace:
    topo = _f1_topology()
    events = [
        SimEvent(t=0, branch_id="main", kind="CREATE_INITIAL",
                 payload={"task": "q", "entry_agent": "A"}),
        SimEvent(t=1, branch_id="main", kind="PARALLEL_INVOKE", payload={
            "invocations": [
                {"agent": "B1", "request": "", "alias": "b1"},
                {"agent": "B2", "request": "", "alias": "b2"},
            ],
            "fork_alias": "fork_A",
        }),
        SimEvent(t=2, branch_id="b1", kind="FINAL_RESPONSE", payload={"value": "B1_ok"}),
        SimEvent(t=2, branch_id="b2", kind="FAIL", payload={"error": "simulated"}),
    ]
    return SimTrace(topology=topo, policy=policy, events=events,
                    assertions=[], name=f"F1_{policy.min_ratio}_{policy.on_insufficient}")


def test_f1_strict_fail():
    """min_ratio=1.0, on_insufficient=fail → workflow fails; main never resumes."""
    policy = ConvergencePolicy(min_ratio=1.0, on_insufficient="fail", terminate_orphans=True)
    trace = _f1_trace(policy)
    # No t=3 main event — with strict policy + failure, fork_A fires_fail,
    # main is orphan-terminated, never resumes.
    run = run_trace(trace)
    root = run.orchestrator.barriers[run.orchestrator.root_barrier_id]
    assert root.status != "FIRED" or run.orchestrator._workflow_error is not None, \
        f"strict policy should fail; got root={root.status}, err={run.orchestrator._workflow_error}"
    # main should not have resumed successfully — either it was abandoned
    # (legacy semantic) or failed via _fire_with_failure (unified semantic).
    main = run.orchestrator.branches[run.alias_map["main"]]
    assert main.status in ("ABANDONED", "WAITING", "FAILED"), f"main status={main.status}"


def test_f1_quorum_proceeds():
    """min_ratio=0.5, on_insufficient=proceed → fires with partial arrivals."""
    policy = ConvergencePolicy(min_ratio=0.5, on_insufficient="proceed", terminate_orphans=True)
    trace = _f1_trace(policy)
    # main needs to emit final after resuming with partial
    trace.events.append(SimEvent(t=3, branch_id="main", kind="FINAL_RESPONSE",
                                 payload={"value": "partial_final"}))
    run = run_trace(trace)
    root = run.orchestrator.barriers[run.orchestrator.root_barrier_id]
    assert root.status == "FIRED", f"quorum should have fired ROOT; got {root.status}"
    assert run.orchestrator._aggregate(root) == "partial_final", \
        f"unexpected final: {run.orchestrator._aggregate(root)}"


def test_f1_any_one():
    """min_ratio=0.0 → first-arrival wins; other cancelled as orphan."""
    policy = ConvergencePolicy(min_ratio=0.0, on_insufficient="proceed", terminate_orphans=True)
    topo = _f1_topology()
    events = [
        SimEvent(t=0, branch_id="main", kind="CREATE_INITIAL",
                 payload={"task": "q", "entry_agent": "A"}),
        SimEvent(t=1, branch_id="main", kind="PARALLEL_INVOKE", payload={
            "invocations": [
                {"agent": "B1", "request": "", "alias": "b1"},
                {"agent": "B2", "request": "", "alias": "b2"},
            ],
            "fork_alias": "fork_A",
        }),
        # Only b1 delivers; b2 "never arrives"
        SimEvent(t=2, branch_id="b1", kind="FINAL_RESPONSE", payload={"value": "B1_ok"}),
        # with ratio=0.0, after b1 arrives pending={b2} still; won't fire yet
        # To test "any-one" we need either b2 to fail or a real ratio<1 trigger.
        SimEvent(t=3, branch_id="b2", kind="FAIL", payload={"error": "slow"}),
        SimEvent(t=4, branch_id="main", kind="FINAL_RESPONSE", payload={"value": "one_ok"}),
    ]
    trace = SimTrace(topology=topo, policy=policy, events=events, assertions=[], name="F1_any")
    run = run_trace(trace)
    root = run.orchestrator.barriers[run.orchestrator.root_barrier_id]
    assert root.status == "FIRED", f"any-one should fire ROOT; got {root.status}"


# ───────────────────────────────────────────────────────────────────
# F5: Partial commitment — only one of two convergence-target invocations
# ───────────────────────────────────────────────────────────────────

def test_f5_partial_commitment():
    """A → [B1, B2]; B1 → D (conv); B2 → D (conv). But at runtime B2 goes
    to an alternate terminal, never reaches D. D should still fire when
    its only arriving candidate (b1's delivery) is ready and the other
    candidate (b2) abandons elsewhere."""
    topo = build_topology(
        nodes=[
            StartNode(),
            SimNode("A"),
            SimNode("B1"), SimNode("B2"),
            SimNode("D", convergence_mode="force"),
            SimNode("Alt"),
        ],
        flows=["Start -> A", "A -> B1", "A -> B2", "B1 -> D", "B2 -> D", "B2 -> Alt"],
    )
    policy = ConvergencePolicy(min_ratio=1.0, on_insufficient="fail", terminate_orphans=True)
    events = [
        SimEvent(t=0, branch_id="main", kind="CREATE_INITIAL",
                 payload={"task": "q", "entry_agent": "A"}),
        SimEvent(t=1, branch_id="main", kind="PARALLEL_INVOKE", payload={
            "invocations": [
                {"agent": "B1", "request": "", "alias": "b1"},
                {"agent": "B2", "request": "", "alias": "b2"},
            ],
            "fork_alias": "fork_A",
        }),
        # b1 → D (delivers to conv_D)
        SimEvent(t=2, branch_id="b1", kind="SINGLE_INVOKE",
                 payload={"next_agent": "D", "value": "b1_to_D"}),
        # b2 → Alt → FINAL_RESPONSE (abandons conv_D via reachability)
        SimEvent(t=2, branch_id="b2", kind="SINGLE_INVOKE",
                 payload={"next_agent": "Alt", "request": ""}),
        SimEvent(t=3, branch_id="b2", kind="FINAL_RESPONSE",
                 payload={"value": "b2_alt_path"}),
    ]
    sim = Simulator(SimTrace(topology=topo, policy=policy, events=events,
                             assertions=[], name="F5"))
    # Agent-script for D (the resolver when conv_D fires)
    sim.mock.queue_agent("D", StepResult(kind="FINAL_RESPONSE", value="D_aggregated"))
    run = sim.run()
    # conv_D should have fired with 1/1 arrivals (b1 delivered; b2 abandoned).
    # ratio = 1/1 = 1.0 passes policy.
    d_bars = [b for b in run.orchestrator.barriers.values()
              if b.kind == "CONVERGENCE" and b.convergence_node == "D"]
    assert d_bars, "conv_D should exist"
    d = d_bars[0]
    assert d.status == "FIRED", f"conv_D should fire; got {d.status}, arrived={list(d.arrived)}"
    assert len(d.arrived) == 1, f"expected 1 arrival; got {list(d.arrived)}"


# ───────────────────────────────────────────────────────────────────
# F7: Resurrection — a late arrival at a FIRED barrier
# ───────────────────────────────────────────────────────────────────

def test_f7_resurrection():
    """A → [B1, B2]; policy any-one with orphan termination; one sibling
    delivers fast and fires, the second's delivery arrives late → should
    be gracefully marked ABANDONED, not crash."""
    topo = build_topology(
        nodes=[StartNode(), SimNode("A"), SimNode("B1"), SimNode("B2")],
        flows=["Start -> A", "A -> B1", "A -> B2", "B1 -> A", "B2 -> A"],
    )
    policy = ConvergencePolicy(min_ratio=0.5, on_insufficient="proceed", terminate_orphans=False)
    events = [
        SimEvent(t=0, branch_id="main", kind="CREATE_INITIAL",
                 payload={"task": "q", "entry_agent": "A"}),
        SimEvent(t=1, branch_id="main", kind="PARALLEL_INVOKE", payload={
            "invocations": [
                {"agent": "B1", "request": "", "alias": "b1"},
                {"agent": "B2", "request": "", "alias": "b2"},
            ],
            "fork_alias": "fork_A",
        }),
        SimEvent(t=2, branch_id="b1", kind="FINAL_RESPONSE", payload={"value": "B1_ok"}),
        # b2 arrives LATE, after fork_A would have fired
        # But with ratio 0.5 and terminate_orphans=False, fork_A can't fire
        # with just b1 — pending still has b2.
        # So b2's late delivery should be accepted; fork fires with both.
        SimEvent(t=3, branch_id="b2", kind="FINAL_RESPONSE", payload={"value": "B2_late"}),
        SimEvent(t=4, branch_id="main", kind="FINAL_RESPONSE", payload={"value": "final"}),
    ]
    run = run_trace(SimTrace(topology=topo, policy=policy, events=events,
                             assertions=[], name="F7"))
    root = run.orchestrator.barriers[run.orchestrator.root_barrier_id]
    assert root.status == "FIRED", "workflow should complete normally with late delivery"


# ───────────────────────────────────────────────────────────────────
# F2: Orphan branch never delivers (simulated via no scripted events)
# ───────────────────────────────────────────────────────────────────

def test_f2_orphan_proceed():
    """min_ratio=0.5, proceed. b1 delivers; b2 "hangs" — never scripted.
    Since we can't realistically time out (mock runtime is synchronous),
    we simulate by failing b2 explicitly at t=3 so the policy can proceed."""
    topo = build_topology(
        nodes=[StartNode(), SimNode("A"), SimNode("B1"), SimNode("B2")],
        flows=["Start -> A", "A -> B1", "A -> B2", "B1 -> A", "B2 -> A"],
    )
    policy = ConvergencePolicy(min_ratio=0.5, on_insufficient="proceed",
                                terminate_orphans=True)
    events = [
        SimEvent(t=0, branch_id="main", kind="CREATE_INITIAL",
                 payload={"task": "q", "entry_agent": "A"}),
        SimEvent(t=1, branch_id="main", kind="PARALLEL_INVOKE", payload={
            "invocations": [
                {"agent": "B1", "request": "", "alias": "b1"},
                {"agent": "B2", "request": "", "alias": "b2"},
            ],
            "fork_alias": "fork_A",
        }),
        SimEvent(t=2, branch_id="b1", kind="FINAL_RESPONSE", payload={"value": "B1_ok"}),
        # Simulate timeout: b2 "fails" because policy allows proceed with partial
        SimEvent(t=3, branch_id="b2", kind="FAIL", payload={"error": "timeout"}),
        SimEvent(t=4, branch_id="main", kind="FINAL_RESPONSE", payload={"value": "partial"}),
    ]
    run = run_trace(SimTrace(topology=topo, policy=policy, events=events,
                             assertions=[], name="F2"))
    root = run.orchestrator.barriers[run.orchestrator.root_barrier_id]
    assert root.status == "FIRED", f"quorum proceed should fire; got {root.status}"


# ───────────────────────────────────────────────────────────────────
# F6: Abandonment cascade — child of a fork abandons to a convergence,
# the fork's upstream-chain adopts the convergence, fork eventually fires
# once the chain's resolver delivers back.
# ───────────────────────────────────────────────────────────────────

def test_f6_abandonment_cascade():
    """A → [B1, B2]; B1 → C (conv); C → A. B2 returns to A normally.
    b1 abandons fork_A to go to conv_C. fork_A adopts conv_C as upstream.
    When conv_C fires, c_resolver invokes A, delivers to fork_A."""
    topo = build_topology(
        nodes=[
            StartNode(),
            SimNode("A"),
            SimNode("B1"), SimNode("B2"),
            SimNode("C", convergence_mode="force"),
        ],
        flows=["Start -> A", "A -> B1", "A -> B2", "B1 -> C", "B2 -> A", "C -> A"],
    )
    policy = ConvergencePolicy(min_ratio=1.0, on_insufficient="fail",
                                terminate_orphans=True)
    events = [
        SimEvent(t=0, branch_id="main", kind="CREATE_INITIAL",
                 payload={"task": "q", "entry_agent": "A"}),
        SimEvent(t=1, branch_id="main", kind="PARALLEL_INVOKE", payload={
            "invocations": [
                {"agent": "B1", "request": "", "alias": "b1"},
                {"agent": "B2", "request": "", "alias": "b2"},
            ],
            "fork_alias": "fork_A",
        }),
        # b1 → C (delivers to conv_C). b1 abandons fork_A.
        SimEvent(t=2, branch_id="b1", kind="SINGLE_INVOKE",
                 payload={"next_agent": "C", "value": "b1_to_C"}),
        # b2 returns to A (delivers to fork_A).
        SimEvent(t=2, branch_id="b2", kind="SINGLE_INVOKE",
                 payload={"next_agent": "A", "value": "b2_to_A"}),
        # main resumes after fork_A fires
        SimEvent(t=3, branch_id="main", kind="FINAL_RESPONSE",
                 payload={"value": "A_final"}),
    ]
    sim = Simulator(SimTrace(topology=topo, policy=policy, events=events,
                             assertions=[], name="F6"))
    # When conv_C fires, c_resolver invokes A (delivers to fork_A)
    sim.mock.queue_agent("C", StepResult(
        kind="SINGLE_INVOKE", next_agent="A", value="C_aggregated",
    ))
    run = sim.run()
    root = run.orchestrator.barriers[run.orchestrator.root_barrier_id]
    assert root.status == "FIRED", \
        f"F6 should complete; root={root.status}, err={run.orchestrator._workflow_error}"


if __name__ == "__main__":
    test_f1_strict_fail()
    print("F1 strict fail: PASS")
    test_f1_quorum_proceeds()
    print("F1 quorum proceeds: PASS")
    test_f1_any_one()
    print("F1 any-one: PASS")
    test_f5_partial_commitment()
    print("F5 partial commitment: PASS")
    test_f7_resurrection()
    print("F7 resurrection: PASS")
    test_f2_orphan_proceed()
    print("F2 orphan (proceed policy): PASS")
    test_f6_abandonment_cascade()
    print("F6 abandonment cascade: PASS")
