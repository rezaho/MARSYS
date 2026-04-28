"""P11: Loop topology terminating via End det-node.

Topology:
    Start → A
    A → [B1, B2]   (parallel)
    B1, B2 → C
    C → A          (loop)
    C → End        (terminate via End det-node — agent's explicit decision)

Auto-detection:
  - A: incomings={Start, C}, outgoings={B1, B2}; diff={Start, C}, |diff|=2 → rendezvous.
  - C: incomings={B1, B2}, outgoings={A, End}; diff={B1, B2}, |diff|=2 → rendezvous.

Each iteration creates fresh barrier instances. Iteration 1's resolver_C
loops back via SINGLE_INVOKE(A) which lands at fork_A_inner (main's
WAITING fork from its parallel_invoke). main wakes and runs iteration 2.
Iteration 2's resolver_C invokes End — that's the agent's explicit
decision to terminate. End delivers value to ROOT. Eventually all
branches settle (Fix 1: ROOT defers until then) and ROOT fires with
the answer.

This test exercises:
  - Det-node integration (Start as auto-dispatch, End as one-way exit).
  - Loop topology with auto-rendezvous on the looped node (A).
  - Fix 1: ROOT defer until all branches settled.
  - Multi-iteration cycle terminated by an explicit agent decision (not
    by step_count exhaustion).
"""
from __future__ import annotations

from research.orchestration.orchestrator.types import Invocation, StepResult
from research.orchestration.simulator.simulator import Simulator
from research.orchestration.simulator.topology import SimNode, build_topology
from research.orchestration.simulator.trace import (
    ConvergencePolicy,
    SimEvent,
    SimTrace,
)


def build_p11_topology():
    from research.orchestration.simulator.det_nodes import EndNode, StartNode
    return build_topology(
        nodes=[
            StartNode(),
            SimNode("A"),
            SimNode("B1"),
            SimNode("B2"),
            SimNode("C"),
            EndNode(),
        ],
        flows=[
            "Start -> A",
            "A -> B1", "A -> B2",
            "B1 -> C", "B2 -> C",
            "C -> A",      # loop
            "C -> End",    # terminate
        ],
    )


def test_p11_consensus_via_end_node():
    """C loops back once, then chooses End on iteration 2.
    Workflow terminates with End's value at ROOT."""
    topo = build_p11_topology()
    policy = ConvergencePolicy(min_ratio=1.0, on_insufficient="fail", terminate_orphans=True)

    events = [
        SimEvent(t=0, branch_id="main", kind="CREATE_INITIAL", payload={"task": "q"}),
    ]
    trace = SimTrace(topology=topo, policy=policy, events=events,
                     assertions=[], name="P11_loop")
    sim = Simulator(trace)

    # A always parallel_invokes [B1, B2].  Up to 3 iterations queued.
    for _ in range(3):
        sim.mock.queue_agent("A", StepResult(
            kind="PARALLEL_INVOKE",
            invocations=[Invocation("B1", ""), Invocation("B2", "")],
        ))
    # B1, B2 always go to C. 4 iterations × 2 children = 8 total.
    for _ in range(4):
        sim.mock.queue_agent("B1", StepResult(
            kind="SINGLE_INVOKE", next_agent="C", value="b1_ans",
        ))
        sim.mock.queue_agent("B2", StepResult(
            kind="SINGLE_INVOKE", next_agent="C", value="b2_ans",
        ))
    # C: loop on iteration 1, terminate via End on iteration 2.
    sim.mock.queue_agent("C", StepResult(
        kind="SINGLE_INVOKE", next_agent="A", request="retry",
    ))
    sim.mock.queue_agent("C", StepResult(
        kind="SINGLE_INVOKE", next_agent="End", value="ANSWER",
    ))

    run = sim.run()
    root = run.orchestrator.barriers[run.orchestrator.root_barrier_id]
    if run.orchestrator._workflow_error is not None:
        raise AssertionError(
            f"P11 expected success; got workflow_error={run.orchestrator._workflow_error}"
        )
    if root.status != "FIRED":
        raise AssertionError(f"P11 expected ROOT FIRED; got {root.status}")
    if not any(v == "ANSWER" for v in root.arrived.values()):
        raise AssertionError(
            f"P11 expected ROOT.arrived to contain ANSWER; got {dict(root.arrived)}"
        )


if __name__ == "__main__":
    test_p11_consensus_via_end_node()
    print("P11 loop with End: PASS")
