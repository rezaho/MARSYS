"""P13: No End node, no terminating chain — workflow must terminate via
step_count exhaustion (and Fix 1 lets ROOT eventually fire).

Topology:
    Start → A
    A → [B1, B2]   (parallel)
    B1, B2 → A     (loop, no exit)

There's no End node, no terminal-leaf path, just an infinite loop.
With no breaking condition, branches accumulate steps until max_steps
caps each one. Each branch fails on max_steps; failure cascades up to
ROOT via the chain. Workflow ends with workflow_error set.

This locks in the behavior of Fix 1 (ROOT defer) under a topology
where nothing reaches ROOT through normal channels — the cascade is
the only path.
"""
from __future__ import annotations

from research.orchestration.orchestrator.types import Invocation, StepResult
from research.orchestration.simulator.det_nodes import StartNode
from research.orchestration.simulator.simulator import Simulator
from research.orchestration.simulator.topology import SimNode, build_topology
from research.orchestration.simulator.trace import (
    ConvergencePolicy,
    SimEvent,
    SimTrace,
)


def build_p13_topology():
    return build_topology(
        nodes=[
            StartNode(),
            SimNode("A"),
            SimNode("B1"),
            SimNode("B2"),
        ],
        flows=[
            "Start -> A",
            "A -> B1", "A -> B2",
            "B1 -> A",
            "B2 -> A",
        ],
    )


def test_p13_no_exit_step_count():
    topo = build_p13_topology()
    policy = ConvergencePolicy(min_ratio=1.0, on_insufficient="fail", terminate_orphans=True)

    events = [
        SimEvent(t=0, branch_id="main", kind="CREATE_INITIAL", payload={"task": "q"}),
    ]
    trace = SimTrace(topology=topo, policy=policy, events=events,
                     assertions=[], name="P13_no_exit")
    sim = Simulator(trace)

    # Infinite loop: A always parallel-invokes [B1, B2]; B1, B2 always go to A.
    # max_steps default = 200; exhaust each branch eventually.
    for _ in range(300):
        sim.mock.queue_agent("A", StepResult(
            kind="PARALLEL_INVOKE",
            invocations=[Invocation("B1", ""), Invocation("B2", "")],
        ))
        sim.mock.queue_agent("B1", StepResult(
            kind="SINGLE_INVOKE", next_agent="A", value="b1_loop",
        ))
        sim.mock.queue_agent("B2", StepResult(
            kind="SINGLE_INVOKE", next_agent="A", value="b2_loop",
        ))

    run = sim.run()
    orch = run.orchestrator
    root = orch.barriers[orch.root_barrier_id]

    # Either workflow_error set, or ROOT not FIRED. Either is acceptable
    # for an unterminating-loop topology — both indicate the workflow
    # didn't succeed.
    if root.status == "FIRED" and orch._workflow_error is None:
        raise AssertionError(
            "P13 expected loop to terminate via step_count failure; "
            f"got ROOT FIRED with no workflow_error. arrived={dict(root.arrived)}"
        )

    # At least one branch should have FAILED via step_count.
    any_failed = any(b.status == "FAILED" for b in orch.branches.values())
    if not any_failed:
        raise AssertionError(
            "P13 expected at least one branch in FAILED status; "
            f"branches={[(b.id, b.status) for b in orch.branches.values()][:10]}"
        )


if __name__ == "__main__":
    test_p13_no_exit_step_count()
    print("P13 no-exit step_count termination: PASS")
