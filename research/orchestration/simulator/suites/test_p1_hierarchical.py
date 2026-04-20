"""P1: Hierarchical parent-convergence (2 levels).

Topology:
    A (entry)
    A  -> [B1, B2]  (parallel)
    B1 -> [B11, B12] (parallel)
    B2 -> [B21, B22] (parallel)
    B11, B12 -> B1
    B21, B22 -> B2
    B1, B2 -> A

Rationale: nested parent-convergence. The baseline pattern for nearly all
real multi-agent systems. If this doesn't work cleanly, nothing else will.

We run one trace and check:
  - 3 fork barriers fire (bar_f_A, bar_f_B1, bar_f_B2)
  - the root barrier fires
  - final_response aggregates the four leaf outcomes into A
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


def build_p1_topology():
    return build_topology(
        nodes=[
            SimNode("A", is_entry=True),
            SimNode("B1"),
            SimNode("B2"),
            SimNode("B11"),
            SimNode("B12"),
            SimNode("B21"),
            SimNode("B22"),
        ],
        flows=[
            "A -> B1", "A -> B2",
            "B1 -> B11", "B1 -> B12",
            "B2 -> B21", "B2 -> B22",
            "B11 -> B1", "B12 -> B1",
            "B21 -> B2", "B22 -> B2",
            "B1 -> A", "B2 -> A",
        ],
    )


def build_p1_happy_trace() -> SimTrace:
    topo = build_p1_topology()
    policy = ConvergencePolicy(min_ratio=1.0, on_insufficient="fail", terminate_orphans=True)

    events = [
        SimEvent(t=0, branch_id="main", kind="CREATE_INITIAL",
                 payload={"task": "the-task", "entry_agent": "A"}),

        # t=1: A invokes B1, B2 in parallel
        SimEvent(t=1, branch_id="main", kind="PARALLEL_INVOKE", payload={
            "invocations": [
                {"agent": "B1", "request": "to_B1", "alias": "b1"},
                {"agent": "B2", "request": "to_B2", "alias": "b2"},
            ],
            "fork_alias": "fork_A",
        }),

        # t=2: B1 and B2 each do their own parallel invoke
        SimEvent(t=2, branch_id="b1", kind="PARALLEL_INVOKE", payload={
            "invocations": [
                {"agent": "B11", "request": "to_B11", "alias": "b11"},
                {"agent": "B12", "request": "to_B12", "alias": "b12"},
            ],
            "fork_alias": "fork_B1",
        }),
        SimEvent(t=2, branch_id="b2", kind="PARALLEL_INVOKE", payload={
            "invocations": [
                {"agent": "B21", "request": "to_B21", "alias": "b21"},
                {"agent": "B22", "request": "to_B22", "alias": "b22"},
            ],
            "fork_alias": "fork_B2",
        }),

        # t=3: leaves finish
        SimEvent(t=3, branch_id="b11", kind="FINAL_RESPONSE", payload={"value": "B11_result"}),
        SimEvent(t=3, branch_id="b12", kind="FINAL_RESPONSE", payload={"value": "B12_result"}),
        SimEvent(t=3, branch_id="b21", kind="FINAL_RESPONSE", payload={"value": "B21_result"}),
        SimEvent(t=3, branch_id="b22", kind="FINAL_RESPONSE", payload={"value": "B22_result"}),

        # t=4: fork_B1 and fork_B2 have fired at t=3 when their last child delivered;
        # B1 and B2 resume. They each produce FINAL_RESPONSE for A.
        SimEvent(t=4, branch_id="b1", kind="FINAL_RESPONSE", payload={"value": "B1_result"}),
        SimEvent(t=4, branch_id="b2", kind="FINAL_RESPONSE", payload={"value": "B2_result"}),

        # t=5: fork_A has fired; main (A) resumes and produces final response
        SimEvent(t=5, branch_id="main", kind="FINAL_RESPONSE", payload={"value": "A_final"}),
    ]

    assertions = [
        # After t=3 all four leaves have delivered; fork_B1 and fork_B2 fired
        SimAssertion(at=3, kind="barrier_status", target="fork_B1", value="FIRED"),
        SimAssertion(at=3, kind="barrier_status", target="fork_B2", value="FIRED"),
        SimAssertion(at=3, kind="barrier_arrived", target="fork_B1", value={"b11", "b12"}),
        SimAssertion(at=3, kind="barrier_arrived", target="fork_B2", value={"b21", "b22"}),

        # After t=4 B1 and B2 have delivered to fork_A which then fires
        SimAssertion(at=4, kind="barrier_status", target="fork_A", value="FIRED"),
        SimAssertion(at=4, kind="barrier_arrived", target="fork_A", value={"b1", "b2"}),

        # After t=5 ROOT fires
        SimAssertion(at=5, kind="barrier_status", target="ROOT", value="FIRED"),
        SimAssertion(at=5, kind="workflow_succeeded", target=None, value=None),
        SimAssertion(at=5, kind="workflow_final_response", target=None, value="A_final"),

        # No deadlock or leaked barriers
        SimAssertion(at="final", kind="no_deadlock", target=None, value=None),
        SimAssertion(at="final", kind="no_leaked_barriers", target=None, value=None),
        SimAssertion(at="final", kind="barrier_fired_count", target=None, value=4),  # 3 forks + root
    ]

    return SimTrace(
        topology=topo,
        policy=policy,
        events=events,
        assertions=assertions,
        name="P1_happy",
    )


# pytest entry point
def test_p1_happy():
    trace = build_p1_happy_trace()
    run = run_trace(trace, verbose=True)
    if not run.passed:
        msg = "\n".join(f"  {f.assertion.kind} target={f.assertion.target}: {f.message}" for f in run.failures)
        raise AssertionError(f"P1 happy path assertions failed:\n{msg}\n\nstep_log: {run.step_log[:50]}")


if __name__ == "__main__":
    test_p1_happy()
    print("P1 happy: PASS")
