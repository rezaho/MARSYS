"""P10: fork-children mixed — some return to parent, one invokes End.

Topology:
    Start → A
    A → [B1, B2, B3]   (parallel)
    B1, B2 → A         (return to parent fork)
    B3 → End           (escalate / final answer directly)

Flow:
  - main parallel_invokes [B1, B2, B3].
  - B1, B2 return their results to fork_A via FINAL_RESPONSE → main aggregates.
  - B3 explicitly invokes End → value goes to ROOT.
  - Both contributions land at ROOT eventually:
    * B3's value via direct End delivery.
    * main's aggregate (from B1, B2) via main → ... → ROOT.
  - ROOT collects multi-arrival list.
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


def build_p10_topology():
    return build_topology(
        nodes=[
            StartNode(),
            SimNode("A"),
            SimNode("B1"), SimNode("B2"), SimNode("B3"),
            EndNode(),
        ],
        flows=[
            "Start -> A",
            "A -> B1", "A -> B2", "A -> B3",
            "B1 -> A", "B2 -> A",
            "B3 -> End",
        ],
    )


def test_p10_mixed_fork_with_end():
    topo = build_p10_topology()
    policy = ConvergencePolicy(min_ratio=1.0, on_insufficient="fail", terminate_orphans=True)

    events = [
        SimEvent(t=0, branch_id="main", kind="CREATE_INITIAL", payload={"task": "q"}),
        SimEvent(t=1, branch_id="main", kind="PARALLEL_INVOKE", payload={
            "invocations": [
                {"agent": "B1", "request": "", "alias": "b1"},
                {"agent": "B2", "request": "", "alias": "b2"},
                {"agent": "B3", "request": "", "alias": "b3"},
            ],
            "fork_alias": "fork_A",
        }),
        SimEvent(t=2, branch_id="b1", kind="FINAL_RESPONSE", payload={"value": "b1_result"}),
        SimEvent(t=2, branch_id="b2", kind="FINAL_RESPONSE", payload={"value": "b2_result"}),
        SimEvent(t=2, branch_id="b3", kind="SINGLE_INVOKE",
                 payload={"next_agent": "End", "value": "b3_to_user"}),
        SimEvent(t=3, branch_id="main", kind="FINAL_RESPONSE", payload={"value": "main_aggregate"}),
    ]
    trace = SimTrace(topology=topo, policy=policy, events=events,
                     assertions=[], name="P10_mixed_end")
    sim = Simulator(trace)

    run = sim.run()
    orch = run.orchestrator
    root = orch.barriers[orch.root_barrier_id]
    if orch._workflow_error is not None:
        raise AssertionError(f"P10 expected success; got error={orch._workflow_error}")
    if root.status != "FIRED":
        raise AssertionError(f"P10 expected ROOT FIRED; got {root.status}")
    # Both contributions present
    arrived_values = list(root.arrived.values())
    if "b3_to_user" not in arrived_values:
        raise AssertionError(f"P10 expected ROOT.arrived to contain 'b3_to_user'; got {arrived_values}")
    if "main_aggregate" not in arrived_values:
        raise AssertionError(f"P10 expected ROOT.arrived to contain 'main_aggregate'; got {arrived_values}")


if __name__ == "__main__":
    test_p10_mixed_fork_with_end()
    print("P10 mixed fork-children with End: PASS")
