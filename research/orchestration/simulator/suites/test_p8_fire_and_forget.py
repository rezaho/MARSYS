"""P8: fire-and-forget. Parent spawns children that don't return.

    A → [B1, B2]
    B1, B2 → Terminal (each produces FINAL_RESPONSE delivered to fork_A)
    A → Final

The parent (main, running A) parallel_invokes [B1, B2]. Children run and
FINAL_RESPONSE to fork_A. Fork fires, main resumes. main transitions to
Final (terminal), then FINAL_RESPONSE to ROOT.

Distinguishes from hierarchical-parent-convergence only in that the children
are leaves (no further agent calls). The framework handles it identically:
FINAL_RESPONSE delivers to fork_A, not to A the agent.
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


def build_p8_topology():
    from research.orchestration.simulator.det_nodes import StartNode
    return build_topology(
        nodes=[
            StartNode(),
            SimNode("A"),
            SimNode("B1"), SimNode("B2"),
            SimNode("Final"),
        ],
        flows=["Start -> A", "A -> B1", "A -> B2", "A -> Final"],
    )


def build_p8_trace() -> SimTrace:
    topo = build_p8_topology()
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
        SimEvent(t=2, branch_id="b1", kind="FINAL_RESPONSE", payload={"value": "B1_res"}),
        SimEvent(t=2, branch_id="b2", kind="FINAL_RESPONSE", payload={"value": "B2_res"}),
        # main resumes; transitions to Final (terminal)
        SimEvent(t=3, branch_id="main", kind="SINGLE_INVOKE",
                 payload={"next_agent": "Final", "request": ""}),
        SimEvent(t=4, branch_id="main", kind="FINAL_RESPONSE",
                 payload={"value": "workflow_complete"}),
    ]
    return SimTrace(
        topology=topo, policy=policy, events=events,
        assertions=[
            SimAssertion(at="final", kind="workflow_succeeded", target=None, value=None),
            SimAssertion(at="final", kind="workflow_final_response", target=None, value="workflow_complete"),
            SimAssertion(at="final", kind="no_leaked_barriers", target=None, value=None),
        ], name="P8",
    )


def test_p8_fire_and_forget():
    run = run_trace(build_p8_trace())
    if not run.passed:
        msg = "\n".join(f"  {f.assertion.kind} target={f.assertion.target}: {f.message}" for f in run.failures)
        raise AssertionError(f"P8 failed:\n{msg}")


if __name__ == "__main__":
    test_p8_fire_and_forget()
    print("P8 fire-and-forget: PASS")
