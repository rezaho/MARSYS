"""P3: Swarm pattern — Coord dispatches N workers in parallel, they return.

    Coord → [W1, W2, W3, W4]
    Wi → Coord

All workers run in parallel, return to Coord. Coord aggregates and produces
final. No convergence nodes — just a single fork.
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


def build_p3_topology(n: int = 4):
    from research.orchestration.simulator.det_nodes import StartNode
    nodes = [StartNode(), SimNode("Coord")] + [SimNode(f"W{i}") for i in range(1, n + 1)]
    flows = ["Start -> Coord"] + [f"Coord -> W{i}" for i in range(1, n + 1)] + [f"W{i} -> Coord" for i in range(1, n + 1)]
    return build_topology(nodes=nodes, flows=flows)


def build_p3_trace(n: int = 4) -> SimTrace:
    topo = build_p3_topology(n)
    policy = ConvergencePolicy(min_ratio=1.0, on_insufficient="fail", terminate_orphans=True)

    events = [
        SimEvent(t=0, branch_id="main", kind="CREATE_INITIAL",
                 payload={"task": "q", "entry_agent": "Coord"}),
        SimEvent(t=1, branch_id="main", kind="PARALLEL_INVOKE", payload={
            "invocations": [
                {"agent": f"W{i}", "request": f"req{i}", "alias": f"w{i}"}
                for i in range(1, n + 1)
            ],
            "fork_alias": "fork_Coord",
        }),
    ]
    # Each worker finishes in parallel at t=2
    for i in range(1, n + 1):
        events.append(SimEvent(t=2, branch_id=f"w{i}", kind="FINAL_RESPONSE",
                               payload={"value": f"W{i}_result"}))
    # After t=2, fork_Coord fires, main resumes → final response
    events.append(SimEvent(t=3, branch_id="main", kind="FINAL_RESPONSE",
                           payload={"value": "Coord_final"}))

    return SimTrace(
        topology=topo, policy=policy, events=events,
        assertions=[
            SimAssertion(at="final", kind="workflow_succeeded", target=None, value=None),
            SimAssertion(at="final", kind="workflow_final_response", target=None, value="Coord_final"),
            SimAssertion(at="final", kind="no_leaked_barriers", target=None, value=None),
            SimAssertion(at="final", kind="barrier_fired_count", target=None, value=2),  # fork + root
        ], name="P3_swarm_n4",
    )


def test_p3_swarm():
    run = run_trace(build_p3_trace(n=4))
    if not run.passed:
        msg = "\n".join(f"  {f.assertion.kind} target={f.assertion.target}: {f.message}" for f in run.failures)
        raise AssertionError(f"P3 failed:\n{msg}")


if __name__ == "__main__":
    test_p3_swarm()
    print("P3 swarm (n=4): PASS")
