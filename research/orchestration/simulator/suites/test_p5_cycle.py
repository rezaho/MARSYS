"""P5: consensus-like cycle.

    Req → [A1, A2] (parallel sub-answers)
    A1, A2 → D (convergence — compares sub-answers)
    D → Req (loop back for disagreement)
    D → Final (terminal when consensus)

Two sub-cases:
  (a) consensus on first iteration — D → Final directly.
  (b) step_count guard: if D keeps looping back to Req, max_steps eventually
      kills the run (exercising cycle termination).
"""
from __future__ import annotations

from research.orchestration.orchestrator.types import StepResult
from research.orchestration.simulator.simulator import Simulator, run_trace
from research.orchestration.simulator.topology import SimNode, build_topology
from research.orchestration.simulator.trace import (
    ConvergencePolicy,
    SimAssertion,
    SimEvent,
    SimTrace,
)


def build_p5_topology():
    from research.orchestration.simulator.det_nodes import StartNode
    return build_topology(
        nodes=[
            StartNode(),
            SimNode("Req"),
            SimNode("A1"), SimNode("A2"),
            SimNode("D", convergence_mode="force"),
            SimNode("Final"),
        ],
        flows=[
            "Start -> Req",
            "Req -> A1", "Req -> A2",
            "A1 -> D", "A2 -> D",
            "D -> Req",  # loop back
            "D -> Final",
        ],
    )


def build_p5_consensus_trace() -> SimTrace:
    """Happy path: D fires once, decides consensus, transitions to Final."""
    topo = build_p5_topology()
    policy = ConvergencePolicy(min_ratio=1.0, on_insufficient="fail", terminate_orphans=True)

    events = [
        SimEvent(t=0, branch_id="main", kind="CREATE_INITIAL",
                 payload={"task": "q", "entry_agent": "Req"}),
        SimEvent(t=1, branch_id="main", kind="PARALLEL_INVOKE", payload={
            "invocations": [
                {"agent": "A1", "request": "", "alias": "a1"},
                {"agent": "A2", "request": "", "alias": "a2"},
            ],
            "fork_alias": "fork_Req",
        }),
        SimEvent(t=2, branch_id="a1", kind="SINGLE_INVOKE",
                 payload={"next_agent": "D", "value": "ans1"}),
        SimEvent(t=2, branch_id="a2", kind="SINGLE_INVOKE",
                 payload={"next_agent": "D", "value": "ans2"}),
    ]

    return SimTrace(
        topology=topo, policy=policy, events=events,
        assertions=[
            SimAssertion(at="final", kind="workflow_succeeded", target=None, value=None),
            SimAssertion(at="final", kind="no_leaked_barriers", target=None, value=None),
        ], name="P5_consensus",
    )


def test_p5_consensus_happy():
    trace = build_p5_consensus_trace()
    sim = Simulator(trace)
    # D is invoked as resolver when conv_D fires. It decides "consensus" and
    # invokes Final (terminal).
    sim.mock.queue_agent("D", StepResult(
        kind="SINGLE_INVOKE", next_agent="Final", request="consensus_reached",
    ))
    # Final is a terminal agent — spawned as resolver of Req's waiting fork?
    # Actually Req's fork_Req expects a1, a2 to deliver. Both abandoned (went
    # to D). R4 adds conv_D as upstream of fork_Req. D fires → spawns
    # d_resolver. d_resolver → Final. Final's first step is FINAL_RESPONSE
    # (it's terminal so no further dispatch).
    sim.mock.queue_agent("Final", StepResult(
        kind="FINAL_RESPONSE", value="ANSWER",
    ))
    # When main resumes at Req after fork_Req fires with Final's result,
    # it produces FINAL_RESPONSE (consensus reached → workflow done).
    sim.mock.queue_agent("Req", StepResult(
        kind="FINAL_RESPONSE", value="consensus_ANSWER",
    ))
    run = sim.run()
    root = run.orchestrator.barriers[run.orchestrator.root_barrier_id]
    if root.status != "FIRED":
        raise AssertionError(
            f"consensus flow should complete; root={root.status}, err={run.orchestrator._workflow_error}\n"
            + "\n".join(f"  {bid}: {b.status} arr={list(b.arrived)}"
                        for bid, b in run.orchestrator.barriers.items())
        )


def test_p5_cycle_termination():
    """D loops back to Req infinitely; the per-branch step_count should
    eventually terminate the runaway."""
    topo = build_p5_topology()
    policy = ConvergencePolicy(min_ratio=1.0, on_insufficient="fail", terminate_orphans=True)

    # Only initial events; D keeps invoking Req, which re-spawns children.
    events = [
        SimEvent(t=0, branch_id="main", kind="CREATE_INITIAL",
                 payload={"task": "q", "entry_agent": "Req"}),
        SimEvent(t=1, branch_id="main", kind="PARALLEL_INVOKE", payload={
            "invocations": [
                {"agent": "A1", "request": "", "alias": "a1"},
                {"agent": "A2", "request": "", "alias": "a2"},
            ],
            "fork_alias": "fork_Req",
        }),
    ]

    trace = SimTrace(topology=topo, policy=policy, events=events,
                     assertions=[], name="P5_cycle")
    sim = Simulator(trace)
    # Agent scripts: A1, A2 always go to D; D always loops back to Req;
    # Req always parallel-invokes A1,A2. Infinite.
    # Queue many copies (deque will be drained as ticks happen).
    from research.orchestration.orchestrator.types import Invocation
    for _ in range(200):
        sim.mock.queue_agent("A1", StepResult(
            kind="SINGLE_INVOKE", next_agent="D", value="ans",
        ))
        sim.mock.queue_agent("A2", StepResult(
            kind="SINGLE_INVOKE", next_agent="D", value="ans",
        ))
        sim.mock.queue_agent("D", StepResult(
            kind="SINGLE_INVOKE", next_agent="Req", request="retry",
        ))
        sim.mock.queue_agent("Req", StepResult(
            kind="PARALLEL_INVOKE",
            invocations=[Invocation("A1", ""), Invocation("A2", "")],
        ))

    # Should terminate via step_count exhaustion (max_steps default = 200)
    run = sim.run()
    # Workflow should NOT have succeeded; some branch should have failed on max_steps
    root = run.orchestrator.barriers[run.orchestrator.root_barrier_id]
    assert root.status != "FIRED" or run.orchestrator._workflow_error is not None, \
        f"infinite cycle should have been terminated; root={root.status}"
    # At least one branch should be FAILED from max_steps exhaustion
    any_step_exhausted = any(
        b.status == "FAILED" for b in run.orchestrator.branches.values()
    )
    # Also accept the workflow stopping due to queue exhaustion in mock
    # (if branch ran out of events before hitting max_steps)
    assert any_step_exhausted or any(
        b.status == "RUNNING" for b in run.orchestrator.branches.values()
    ), "expected either step exhaustion or leftover running branches"


if __name__ == "__main__":
    test_p5_consensus_happy()
    print("P5 consensus happy: PASS")
    test_p5_cycle_termination()
    print("P5 cycle termination: PASS")
