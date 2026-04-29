"""Failure-mode tests: F1 (strict-fail cascade), F-step-count (max_steps).

These exercise the ROOT-failure cascade: when a barrier fires-with-failure,
the failure travels up the wait-graph until ROOT sets _workflow_error, and
the workflow ends with success=False.
"""
from __future__ import annotations

from marsys.coordination.execution.deterministic_runtime import DeterministicRuntime
from marsys.coordination.execution.orchestrator import Orchestrator
from marsys.coordination.execution.orchestrator_types import (
    ConvergencePolicy,
    Invocation,
    StepResult,
    reset_ids,
)

from ._helpers import build_topology


def test_f1_strict_fail_cascade():
    """One sibling fails under strict policy → cascade to ROOT."""
    reset_ids()
    topo = build_topology(
        nodes=["Start", "A", "B1", "B2"],
        flows=["Start -> A", "A -> B1", "A -> B2", "B1 -> A", "B2 -> A"],
    )

    runtime = DeterministicRuntime()
    runtime.queue_agent("A", StepResult(
        kind="PARALLEL_INVOKE",
        invocations=[Invocation("B1"), Invocation("B2")],
    ))
    runtime.queue_agent("B1", StepResult(kind="FINAL_RESPONSE", value="b1_ok"))
    runtime.queue_agent("B2", StepResult(kind="FAIL", error="b2_died"))

    policy = ConvergencePolicy(min_ratio=1.0, on_insufficient="fail", terminate_orphans=True)
    orch = Orchestrator(topo, runtime, policy)
    result = orch.run(task="q")

    assert not result.success
    assert "ROOT failure" in (result.error or "")


def test_f_step_count_termination():
    """Infinite loop without End — terminates via step_count exhaustion."""
    reset_ids()
    topo = build_topology(
        nodes=["Start", "A", "B1", "B2"],
        flows=["Start -> A", "A -> B1", "A -> B2", "B1 -> A", "B2 -> A"],
    )

    runtime = DeterministicRuntime()
    # Queue enough infinite-loop scripts that the orchestrator hits max_steps
    for _ in range(300):
        runtime.queue_agent("A", StepResult(
            kind="PARALLEL_INVOKE",
            invocations=[Invocation("B1"), Invocation("B2")],
        ))
        runtime.queue_agent("B1", StepResult(kind="SINGLE_INVOKE", next_agent="A", value="b1_loop"))
        runtime.queue_agent("B2", StepResult(kind="SINGLE_INVOKE", next_agent="A", value="b2_loop"))

    orch = Orchestrator(topo, runtime, ConvergencePolicy())
    result = orch.run(task="q")

    # Either workflow_error set, or ROOT not FIRED — both indicate the loop
    # didn't succeed.
    root = result.barriers[orch.root_barrier_id]
    assert (root.status != "FIRED") or (orch._workflow_error is not None)
    # At least one branch should have FAILED via step_count.
    any_failed = any(b.status == "FAILED" for b in result.branches.values())
    assert any_failed
