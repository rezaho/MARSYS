"""P9: rendezvous resolver invokes End to deliver the final answer.

Topology:
    Start → A
    A → [B1, B2]   (parallel)
    B1, B2 → C     (rendezvous)
    C → End

Flow:
  - main spawned at A, parallel_invokes [B1, B2]. Children abandon to bar_C.
  - main's fork_A waits on bar_C (upstream link via abandonment).
  - bar_C fires; resolver_C runs C; C invokes End explicitly.
  - End delivers value to ROOT.
  - main's fork_A vestigial-cancels (no children returned). main abandoned.
  - All branches settle. Fix 1 has ROOT defer until then; ROOT fires
    with End's value.
"""
from __future__ import annotations

from research.orchestration.orchestrator.types import Invocation, StepResult
from research.orchestration.simulator.det_nodes import EndNode, StartNode
from research.orchestration.simulator.simulator import Simulator
from research.orchestration.simulator.topology import SimNode, build_topology
from research.orchestration.simulator.trace import (
    ConvergencePolicy,
    SimEvent,
    SimTrace,
)


def build_p9_topology():
    return build_topology(
        nodes=[
            StartNode(),
            SimNode("A"),
            SimNode("B1"), SimNode("B2"),
            SimNode("C"),
            EndNode(),
        ],
        flows=[
            "Start -> A",
            "A -> B1", "A -> B2",
            "B1 -> C", "B2 -> C",
            "C -> End",
        ],
    )


def test_p9_rendezvous_to_end():
    topo = build_p9_topology()
    policy = ConvergencePolicy(min_ratio=1.0, on_insufficient="fail", terminate_orphans=True)

    events = [
        SimEvent(t=0, branch_id="main", kind="CREATE_INITIAL", payload={"task": "q"}),
    ]
    trace = SimTrace(topology=topo, policy=policy, events=events,
                     assertions=[], name="P9_rendezvous_end")
    sim = Simulator(trace)

    sim.mock.queue_agent("A", StepResult(
        kind="PARALLEL_INVOKE",
        invocations=[Invocation("B1", ""), Invocation("B2", "")],
    ))
    sim.mock.queue_agent("B1", StepResult(
        kind="SINGLE_INVOKE", next_agent="C", value="b1_ans"))
    sim.mock.queue_agent("B2", StepResult(
        kind="SINGLE_INVOKE", next_agent="C", value="b2_ans"))
    sim.mock.queue_agent("C", StepResult(
        kind="SINGLE_INVOKE", next_agent="End", value="FINAL_ANSWER"))

    run = sim.run()
    root = run.orchestrator.barriers[run.orchestrator.root_barrier_id]
    if run.orchestrator._workflow_error is not None:
        raise AssertionError(f"P9 expected success; got error={run.orchestrator._workflow_error}")
    if root.status != "FIRED":
        raise AssertionError(f"P9 expected ROOT FIRED; got {root.status}")
    if not any(v == "FINAL_ANSWER" for v in root.arrived.values()):
        raise AssertionError(
            f"P9 expected ROOT.arrived to contain FINAL_ANSWER; got {dict(root.arrived)}"
        )


if __name__ == "__main__":
    test_p9_rendezvous_to_end()
    print("P9 rendezvous → End: PASS")
