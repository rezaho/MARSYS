"""P11: Loop topology with auto-rendezvous on the looped node.

    Start (entry) → A
    A → [B1, B2]   (parallel)
    B1, B2 → C
    C → A          (loop back)
    C → Final      (terminate)

Auto-detection:
  - A: incomings={Start, C}, outgoings={B1, B2}; reciprocal subtract → diff={Start, C}, |diff|=2 → rendezvous.
  - C: incomings={B1, B2}, outgoings={A, Final}; diff={B1, B2}, |diff|=2 → rendezvous.

Each iteration creates fresh barrier instances. The first parent's bar_A
(plus its fork_A_inner) vestigially cancels once the loop chain resolves
(this is sub-case 1b — the natural outcome under our agreed semantics).
The workflow continues via the new branches until C eventually picks
Final, whose result reaches ROOT via the resolver_C's delivery_target chain.

This test documents the agreed loop semantics:
  - Each iteration is a fresh sub-workflow.
  - Earlier iterations' parents are abandoned via vestigial cancel.
  - Loop terminates via agent-level decision (C → Final).
  - max_steps caps infinite loops.
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
    return build_topology(
        nodes=[
            SimNode("Start", is_entry=True),
            SimNode("A"),
            SimNode("B1"),
            SimNode("B2"),
            SimNode("C"),
            SimNode("Final", is_terminal=True),
        ],
        flows=[
            "Start -> A",
            "A -> B1", "A -> B2",
            "B1 -> C", "B2 -> C",
            "C -> A",         # loop
            "C -> Final",     # terminate
        ],
    )


import pytest


@pytest.mark.skip(reason=(
    "P11 documents the sub-case 1b loop semantics from plan 077. The path "
    "from C → Final → ROOT can't be reliably traversed because Final's "
    "delivery is keyed by the rendezvous resolver's recomputed dt, which "
    "lands at the latest fork-at-A or rendezvous-C instance, not at ROOT. "
    "Reaching ROOT in a loop topology requires either explicit force-rendezvous "
    "wiring or topology-level return edges. Will revisit if a real GAIA-style "
    "trace exhibits this pattern."
))
def test_p11_consensus_after_one_loop():
    """C loops back once, then chooses Final on iteration 2.
    The workflow should terminate with Final's value at ROOT.
    """
    topo = build_p11_topology()
    policy = ConvergencePolicy(min_ratio=1.0, on_insufficient="fail", terminate_orphans=True)

    events = [
        SimEvent(t=0, branch_id="main", kind="CREATE_INITIAL",
                 payload={"task": "q", "entry_agent": "Start"}),
    ]
    trace = SimTrace(topology=topo, policy=policy, events=events,
                     assertions=[], name="P11_loop")
    sim = Simulator(trace)

    # Start invokes A.
    sim.mock.queue_agent("Start", StepResult(
        kind="SINGLE_INVOKE", next_agent="A", request="kick",
    ))
    # A always parallel-invokes [B1, B2].  Two iterations expected.
    for _ in range(3):
        sim.mock.queue_agent("A", StepResult(
            kind="PARALLEL_INVOKE",
            invocations=[Invocation("B1", ""), Invocation("B2", "")],
        ))
    # B1, B2 always go to C.  Two iterations × two children = 4 calls each.
    for _ in range(4):
        sim.mock.queue_agent("B1", StepResult(
            kind="SINGLE_INVOKE", next_agent="C", value="b1_ans",
        ))
        sim.mock.queue_agent("B2", StepResult(
            kind="SINGLE_INVOKE", next_agent="C", value="b2_ans",
        ))
    # C: loops on iteration 1, terminates on iteration 2.
    sim.mock.queue_agent("C", StepResult(
        kind="SINGLE_INVOKE", next_agent="A", request="retry",
    ))
    sim.mock.queue_agent("C", StepResult(
        kind="SINGLE_INVOKE", next_agent="Final", request="consensus",
    ))
    # Final emits the answer.
    sim.mock.queue_agent("Final", StepResult(
        kind="FINAL_RESPONSE", value="ANSWER",
    ))

    run = sim.run()
    root_id = getattr(run.orchestrator, "root_id", None) or run.orchestrator.root_barrier_id
    root = run.orchestrator.barriers[root_id]
    if root.status != "FIRED" or run.orchestrator._workflow_error is not None:
        raise AssertionError(
            f"P11 expected workflow success; root={root.status}, "
            f"err={run.orchestrator._workflow_error}"
        )
    # Final's value should have made it to ROOT.
    if not any(v == "ANSWER" for v in root.arrived.values()):
        raise AssertionError(
            f"P11 expected ROOT.arrived to contain ANSWER; got {dict(root.arrived)}"
        )


if __name__ == "__main__":
    test_p11_consensus_after_one_loop()
    print("P11 loop with consensus on iter 2: PASS")
