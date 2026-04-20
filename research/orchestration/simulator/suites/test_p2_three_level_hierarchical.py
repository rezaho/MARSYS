"""P2: 3-level hierarchical (stack-depth stress).

    A → [B1, B2]
    B1 → [B11, B12]
    B11 → [B111, B112]
    B12, B2 are leaves (produce FINAL_RESPONSE)
    B111, B112 are leaves

Confirms nested fork-resumption works at arbitrary depth with no barrier
leak. Same as P1 but one level deeper under B1.
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


def build_p2_topology():
    return build_topology(
        nodes=[
            SimNode("A", is_entry=True),
            SimNode("B1"),
            SimNode("B2"),
            SimNode("B11"),
            SimNode("B12"),
            SimNode("B111"),
            SimNode("B112"),
        ],
        flows=[
            "A -> B1", "A -> B2",
            "B1 -> B11", "B1 -> B12",
            "B11 -> B111", "B11 -> B112",
            "B111 -> B11", "B112 -> B11",
            "B11 -> B1", "B12 -> B1",
            "B1 -> A", "B2 -> A",
        ],
    )


def build_p2_trace() -> SimTrace:
    topo = build_p2_topology()
    policy = ConvergencePolicy(min_ratio=1.0, on_insufficient="fail", terminate_orphans=True)

    events = [
        SimEvent(t=0, branch_id="main", kind="CREATE_INITIAL",
                 payload={"task": "q", "entry_agent": "A"}),
        SimEvent(t=1, branch_id="main", kind="PARALLEL_INVOKE", payload={
            "invocations": [
                {"agent": "B1", "request": "B1", "alias": "b1"},
                {"agent": "B2", "request": "B2", "alias": "b2"},
            ],
            "fork_alias": "fork_A",
        }),
        SimEvent(t=2, branch_id="b1", kind="PARALLEL_INVOKE", payload={
            "invocations": [
                {"agent": "B11", "request": "B11", "alias": "b11"},
                {"agent": "B12", "request": "B12", "alias": "b12"},
            ],
            "fork_alias": "fork_B1",
        }),
        SimEvent(t=3, branch_id="b11", kind="PARALLEL_INVOKE", payload={
            "invocations": [
                {"agent": "B111", "request": "B111", "alias": "b111"},
                {"agent": "B112", "request": "B112", "alias": "b112"},
            ],
            "fork_alias": "fork_B11",
        }),
        SimEvent(t=4, branch_id="b111", kind="FINAL_RESPONSE", payload={"value": "B111_res"}),
        SimEvent(t=4, branch_id="b112", kind="FINAL_RESPONSE", payload={"value": "B112_res"}),
        SimEvent(t=4, branch_id="b12", kind="FINAL_RESPONSE", payload={"value": "B12_res"}),
        SimEvent(t=4, branch_id="b2", kind="FINAL_RESPONSE", payload={"value": "B2_res"}),
        # After t=4: fork_B11 fires, b11 resumes
        SimEvent(t=5, branch_id="b11", kind="FINAL_RESPONSE", payload={"value": "B11_res"}),
        # After t=5: fork_B1 fires, b1 resumes
        SimEvent(t=6, branch_id="b1", kind="FINAL_RESPONSE", payload={"value": "B1_res"}),
        # fork_A fires, main resumes
        SimEvent(t=7, branch_id="main", kind="FINAL_RESPONSE", payload={"value": "A_final"}),
    ]

    assertions = [
        SimAssertion(at="final", kind="workflow_succeeded", target=None, value=None),
        SimAssertion(at="final", kind="workflow_final_response", target=None, value="A_final"),
        SimAssertion(at="final", kind="no_deadlock", target=None, value=None),
        SimAssertion(at="final", kind="no_leaked_barriers", target=None, value=None),
        # 3 forks + 1 root = 4 fired barriers (no convergences in this topology)
        SimAssertion(at="final", kind="barrier_fired_count", target=None, value=4),
    ]

    return SimTrace(topology=topo, policy=policy, events=events,
                    assertions=assertions, name="P2")


def test_p2_three_level():
    run = run_trace(build_p2_trace())
    if not run.passed:
        msg = "\n".join(f"  {f.assertion.kind} target={f.assertion.target}: {f.message}" for f in run.failures)
        raise AssertionError(f"P2 failed:\n{msg}")


if __name__ == "__main__":
    test_p2_three_level()
    print("P2 3-level hierarchical: PASS")
