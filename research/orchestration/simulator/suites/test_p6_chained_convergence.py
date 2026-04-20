"""P6: chained convergence (user's hairy example).

Topology:
    A  -> B1, B2
    B1 -> C1, C2
    B2 -> C2, D
    C1 -> D
    C2 -> D
    D  -> A

C2 and D are EXPLICITLY marked as convergence nodes. All other nodes are
regular dispatchers/workers.

Because this topology has a cycle (D→A→Bi→...→D) and C2, D are both in the
cycle's SCC, preds(C2) = ∅ and preds(D) = ∅ under the cycle-break rule.
Ordering emerges naturally from candidate tracking: conv_D's candidates
include c2_resolver (spawned when conv_C2 fires), so conv_D waits until
c2_resolver delivers.

Flow:
  t=0  SPAWN main at A.
  t=1  main PARALLEL_INVOKE([B1, B2]) → fork_A, b1, b2 spawned.
  t=2  b1 PARALLEL_INVOKE([C1, C2]) → fork_B1, b1_c1 spawned, b1 delivers
       to conv_C2 directly (Model B).
       b2 PARALLEL_INVOKE([C2, D]) → all convergences; b2 delivers to
       conv_C2 and conv_D, terminates.
       conv_C2 now has {b1, b2} arrived → fires. Spawns c2_resolver at C2.
  t=3  b1_c1 runs C1, SINGLE_INVOKE(D) → delivers to conv_D.
       c2_resolver runs C2, SINGLE_INVOKE(D) → delivers to conv_D.
       conv_D has {b2, b1_c1, c2_resolver} → fires. Spawns d_resolver.
       d_resolver runs D, SINGLE_INVOKE(A) → resolve_target finds fork_A
       (main's waiting resolver) → delivers to fork_A.
       fork_A fires → main RESUMES.
  t=4  main FINAL_RESPONSE → ROOT fires.

Note: c2_resolver and d_resolver are auto-spawned; their steps are scripted
by AGENT name (not branch id) via mock.queue_agent.
"""
from __future__ import annotations

from research.orchestration.orchestrator.types import Invocation, StepResult
from research.orchestration.simulator.simulator import Simulator, run_trace
from research.orchestration.simulator.topology import SimNode, build_topology
from research.orchestration.simulator.trace import (
    ConvergencePolicy,
    SimAssertion,
    SimEvent,
    SimTrace,
)


def build_p6_topology():
    return build_topology(
        nodes=[
            SimNode("A", is_entry=True),
            SimNode("B1"),
            SimNode("B2"),
            SimNode("C1"),
            SimNode("C2", convergence_mode="force"),
            SimNode("D", convergence_mode="force"),
        ],
        flows=[
            "A -> B1", "A -> B2",
            "B1 -> C1", "B1 -> C2",
            "B2 -> C2", "B2 -> D",
            "C1 -> D", "C2 -> D",
            "D -> A",
        ],
    )


def build_p6_trace() -> SimTrace:
    topo = build_p6_topology()
    policy = ConvergencePolicy(min_ratio=1.0, on_insufficient="fail", terminate_orphans=True)

    events = [
        SimEvent(t=0, branch_id="main", kind="CREATE_INITIAL",
                 payload={"task": "q", "entry_agent": "A"}),

        SimEvent(t=1, branch_id="main", kind="PARALLEL_INVOKE", payload={
            "invocations": [
                {"agent": "B1", "request": "to_B1", "alias": "b1"},
                {"agent": "B2", "request": "to_B2", "alias": "b2"},
            ],
            "fork_alias": "fork_A",
        }),

        SimEvent(t=2, branch_id="b1", kind="PARALLEL_INVOKE", payload={
            "invocations": [
                {"agent": "C1", "request": "to_C1", "alias": "b1_c1"},
                {"agent": "C2", "request": "to_C2_via_B1"},
            ],
            "fork_alias": "fork_B1",
        }),
        SimEvent(t=2, branch_id="b2", kind="PARALLEL_INVOKE", payload={
            "invocations": [
                {"agent": "D",  "request": "to_D_direct"},
                {"agent": "C2", "request": "to_C2_via_B2"},
            ],
        }),

        SimEvent(t=3, branch_id="b1_c1", kind="SINGLE_INVOKE", payload={
            "next_agent": "D", "value": "C1_result",
        }),

        SimEvent(t=4, branch_id="main", kind="FINAL_RESPONSE",
                 payload={"value": "A_final"}),
    ]

    assertions = [
        SimAssertion(at="final", kind="workflow_succeeded", target=None, value=None),
        SimAssertion(at="final", kind="workflow_final_response", target=None, value="A_final"),
        SimAssertion(at="final", kind="no_deadlock", target=None, value=None),
    ]

    return SimTrace(
        topology=topo, policy=policy, events=events,
        assertions=assertions, name="P6_full",
    )


def test_p6_full():
    trace = build_p6_trace()
    sim = Simulator(trace)

    # Script auto-spawned resolvers by agent name.
    # c2_resolver will run at agent=C2, should SINGLE_INVOKE(D).
    sim.mock.queue_agent("C2", StepResult(
        kind="SINGLE_INVOKE", next_agent="D", value="C2_aggregated",
    ))
    # d_resolver will run at agent=D, should SINGLE_INVOKE(A).
    sim.mock.queue_agent("D", StepResult(
        kind="SINGLE_INVOKE", next_agent="A", value="D_aggregated",
    ))

    run = sim.run()
    if not run.passed:
        msg = "\n".join(f"  {f.assertion.kind} target={f.assertion.target}: {f.message}" for f in run.failures)
        # Dump state for debugging
        state_dump = []
        for bid, b in run.orchestrator.branches.items():
            state_dump.append(f"    {bid}: agent={b.current_agent} status={b.status}")
        for bar_id, b in run.orchestrator.barriers.items():
            state_dump.append(f"    {bar_id}[{b.kind} node={b.convergence_node}] "
                              f"status={b.status} arr={list(b.arrived)} up={b.upstream}")
        raise AssertionError(
            f"P6 full failed:\n{msg}\n\nState:\n" + "\n".join(state_dump)
        )


if __name__ == "__main__":
    test_p6_full()
    print("P6 full: PASS")
