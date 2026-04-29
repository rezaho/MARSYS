"""P1: hierarchical parent-convergence (2 levels).

Topology:
    Start -> A
    A -> [B1, B2]
    B1 -> [B11, B12]; B11, B12 -> B1
    B2 -> [B21, B22]; B21, B22 -> B2
    B1, B2 -> A

The fundamental nested fork-rejoin pattern. If this regresses, nothing
else will work.
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


def test_p1_hierarchical_two_levels():
    reset_ids()
    topo = build_topology(
        nodes=["Start", "A", "B1", "B2", "B11", "B12", "B21", "B22"],
        flows=[
            "Start -> A",
            "A -> B1", "A -> B2",
            "B1 -> B11", "B1 -> B12",
            "B2 -> B21", "B2 -> B22",
            "B11 -> B1", "B12 -> B1",
            "B21 -> B2", "B22 -> B2",
            "B1 -> A", "B2 -> A",
        ],
    )

    runtime = DeterministicRuntime()
    runtime.queue_agent("A", StepResult(
        kind="PARALLEL_INVOKE",
        invocations=[Invocation("B1", "to_b1"), Invocation("B2", "to_b2")],
    ))
    runtime.queue_agent("B1", StepResult(
        kind="PARALLEL_INVOKE",
        invocations=[Invocation("B11"), Invocation("B12")],
    ))
    runtime.queue_agent("B2", StepResult(
        kind="PARALLEL_INVOKE",
        invocations=[Invocation("B21"), Invocation("B22")],
    ))
    runtime.queue_agent("B11", StepResult(kind="FINAL_RESPONSE", value="b11_done"))
    runtime.queue_agent("B12", StepResult(kind="FINAL_RESPONSE", value="b12_done"))
    runtime.queue_agent("B21", StepResult(kind="FINAL_RESPONSE", value="b21_done"))
    runtime.queue_agent("B22", StepResult(kind="FINAL_RESPONSE", value="b22_done"))
    runtime.queue_agent("B1", StepResult(kind="FINAL_RESPONSE", value="b1_aggregate"))
    runtime.queue_agent("B2", StepResult(kind="FINAL_RESPONSE", value="b2_aggregate"))
    runtime.queue_agent("A", StepResult(kind="FINAL_RESPONSE", value="a_final"))

    orch = Orchestrator(topo, runtime, ConvergencePolicy())
    result = orch.run(task="task")

    assert result.success, result.error
    assert result.final_response == "a_final"
    # 3 fork barriers + 1 root = 4 barriers; all FIRED
    fired = [b for b in result.barriers.values() if b.status == "FIRED"]
    assert len(fired) == 4, [(b.id, b.kind, b.status) for b in result.barriers.values()]
