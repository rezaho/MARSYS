"""P4: pipeline with a local fork.

    A → B → C → [D1, D2] → C → E

Runtime flow:
  main at A → SINGLE_INVOKE(B) → SINGLE_INVOKE(C) → PARALLEL_INVOKE([D1, D2])
  main WAITING on fork_C (the fork created when C did parallel_invoke)
  d1, d2 → FINAL_RESPONSE (to fork_C)
  main resumes at C with aggregated, SINGLE_INVOKE(E) → FINAL_RESPONSE → ROOT.

No convergence nodes needed — just sequential transitions with a local fork.
"""
from __future__ import annotations

from research.orchestration.simulator.simulator import run_trace
from research.orchestration.simulator.topology import SimNode, build_topology
from research.orchestration.simulator.trace import (
    ConvergencePolicy,
    SimAssertion,
    SimEvent,
    SimTrace,
)


def build_p4_topology():
    return build_topology(
        nodes=[
            SimNode("A", is_entry=True),
            SimNode("B"), SimNode("C"),
            SimNode("D1"), SimNode("D2"),
            SimNode("E", is_terminal=True),
        ],
        flows=[
            "A -> B", "B -> C",
            "C -> D1", "C -> D2",
            "D1 -> C", "D2 -> C",
            "C -> E",
        ],
    )


def build_p4_trace() -> SimTrace:
    topo = build_p4_topology()
    policy = ConvergencePolicy(min_ratio=1.0, on_insufficient="fail", terminate_orphans=True)

    events = [
        SimEvent(t=0, branch_id="main", kind="CREATE_INITIAL",
                 payload={"task": "q", "entry_agent": "A"}),
        # A → B
        SimEvent(t=1, branch_id="main", kind="SINGLE_INVOKE",
                 payload={"next_agent": "B", "request": "from_A"}),
        # B → C
        SimEvent(t=2, branch_id="main", kind="SINGLE_INVOKE",
                 payload={"next_agent": "C", "request": "from_B"}),
        # C → parallel [D1, D2]
        SimEvent(t=3, branch_id="main", kind="PARALLEL_INVOKE", payload={
            "invocations": [
                {"agent": "D1", "request": "", "alias": "d1"},
                {"agent": "D2", "request": "", "alias": "d2"},
            ],
            "fork_alias": "fork_C",
        }),
        SimEvent(t=4, branch_id="d1", kind="FINAL_RESPONSE", payload={"value": "D1_res"}),
        SimEvent(t=4, branch_id="d2", kind="FINAL_RESPONSE", payload={"value": "D2_res"}),
        # main resumes; C's next agent is E
        SimEvent(t=5, branch_id="main", kind="SINGLE_INVOKE",
                 payload={"next_agent": "E", "request": "C_aggregated"}),
        # E emits final
        SimEvent(t=6, branch_id="main", kind="FINAL_RESPONSE", payload={"value": "E_final"}),
    ]
    return SimTrace(
        topology=topo, policy=policy, events=events,
        assertions=[
            SimAssertion(at="final", kind="workflow_succeeded", target=None, value=None),
            SimAssertion(at="final", kind="workflow_final_response", target=None, value="E_final"),
            SimAssertion(at="final", kind="no_leaked_barriers", target=None, value=None),
            SimAssertion(at="final", kind="barrier_fired_count", target=None, value=2),  # fork + root
        ], name="P4",
    )


def test_p4_pipeline():
    run = run_trace(build_p4_trace())
    if not run.passed:
        msg = "\n".join(f"  {f.assertion.kind} target={f.assertion.target}: {f.message}" for f in run.failures)
        raise AssertionError(f"P4 failed:\n{msg}")


if __name__ == "__main__":
    test_p4_pipeline()
    print("P4 pipeline with local fork: PASS")
