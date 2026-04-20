"""P7: Mixed hierarchical + swarm composition.

    Coord → [Sub1, Sub2, Sub3]
    Sub1 → [W1a, W1b]  (swarm inside a hierarchical role)
    Sub2 → W2a         (simple forward)
    Sub3 → [W3a, W3b, W3c]
    Workers → their Subi → Coord

Verifies that hierarchical (fork-per-level) and swarm (fan-out-fan-in)
compose without interference.
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


def build_p7_topology():
    return build_topology(
        nodes=[
            SimNode("Coord", is_entry=True),
            SimNode("Sub1"),
            SimNode("Sub2"),
            SimNode("Sub3"),
            SimNode("W1a"), SimNode("W1b"),
            SimNode("W2a"),
            SimNode("W3a"), SimNode("W3b"), SimNode("W3c"),
        ],
        flows=[
            "Coord -> Sub1", "Coord -> Sub2", "Coord -> Sub3",
            "Sub1 -> W1a", "Sub1 -> W1b",
            "Sub2 -> W2a",
            "Sub3 -> W3a", "Sub3 -> W3b", "Sub3 -> W3c",
            "W1a -> Sub1", "W1b -> Sub1",
            "W2a -> Sub2",
            "W3a -> Sub3", "W3b -> Sub3", "W3c -> Sub3",
            "Sub1 -> Coord", "Sub2 -> Coord", "Sub3 -> Coord",
        ],
    )


def build_p7_trace() -> SimTrace:
    topo = build_p7_topology()
    policy = ConvergencePolicy(min_ratio=1.0, on_insufficient="fail", terminate_orphans=True)

    events = [
        SimEvent(t=0, branch_id="main", kind="CREATE_INITIAL",
                 payload={"task": "q", "entry_agent": "Coord"}),
        SimEvent(t=1, branch_id="main", kind="PARALLEL_INVOKE", payload={
            "invocations": [
                {"agent": "Sub1", "request": "S1", "alias": "s1"},
                {"agent": "Sub2", "request": "S2", "alias": "s2"},
                {"agent": "Sub3", "request": "S3", "alias": "s3"},
            ],
            "fork_alias": "fork_Coord",
        }),
        SimEvent(t=2, branch_id="s1", kind="PARALLEL_INVOKE", payload={
            "invocations": [
                {"agent": "W1a", "request": "", "alias": "w1a"},
                {"agent": "W1b", "request": "", "alias": "w1b"},
            ],
            "fork_alias": "fork_S1",
        }),
        SimEvent(t=2, branch_id="s2", kind="SINGLE_INVOKE", payload={
            "next_agent": "W2a", "request": "",
        }),
        SimEvent(t=2, branch_id="s3", kind="PARALLEL_INVOKE", payload={
            "invocations": [
                {"agent": "W3a", "request": "", "alias": "w3a"},
                {"agent": "W3b", "request": "", "alias": "w3b"},
                {"agent": "W3c", "request": "", "alias": "w3c"},
            ],
            "fork_alias": "fork_S3",
        }),
        # Leaves finish at t=3
        SimEvent(t=3, branch_id="w1a", kind="FINAL_RESPONSE", payload={"value": "W1a"}),
        SimEvent(t=3, branch_id="w1b", kind="FINAL_RESPONSE", payload={"value": "W1b"}),
        # s2 transitioned to W2a at t=2; s2's branch is now at W2a
        SimEvent(t=3, branch_id="s2", kind="FINAL_RESPONSE", payload={"value": "W2a_then_back"}),
        SimEvent(t=3, branch_id="w3a", kind="FINAL_RESPONSE", payload={"value": "W3a"}),
        SimEvent(t=3, branch_id="w3b", kind="FINAL_RESPONSE", payload={"value": "W3b"}),
        SimEvent(t=3, branch_id="w3c", kind="FINAL_RESPONSE", payload={"value": "W3c"}),
        # Subs resume
        SimEvent(t=4, branch_id="s1", kind="FINAL_RESPONSE", payload={"value": "Sub1_res"}),
        SimEvent(t=4, branch_id="s3", kind="FINAL_RESPONSE", payload={"value": "Sub3_res"}),
        # Coord resumes
        SimEvent(t=5, branch_id="main", kind="FINAL_RESPONSE", payload={"value": "Coord_final"}),
    ]

    return SimTrace(
        topology=topo, policy=policy, events=events,
        assertions=[
            SimAssertion(at="final", kind="workflow_succeeded", target=None, value=None),
            SimAssertion(at="final", kind="workflow_final_response", target=None, value="Coord_final"),
            SimAssertion(at="final", kind="no_leaked_barriers", target=None, value=None),
            # fork_Coord + fork_S1 + fork_S3 + ROOT = 4 fired
            SimAssertion(at="final", kind="barrier_fired_count", target=None, value=4),
        ], name="P7_mixed",
    )


def test_p7_mixed():
    run = run_trace(build_p7_trace())
    if not run.passed:
        msg = "\n".join(f"  {f.assertion.kind} target={f.assertion.target}: {f.message}" for f in run.failures)
        raise AssertionError(f"P7 failed:\n{msg}")


if __name__ == "__main__":
    test_p7_mixed()
    print("P7 mixed: PASS")
