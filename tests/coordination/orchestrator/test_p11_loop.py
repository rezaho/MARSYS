"""P11: loop topology terminating via End det-node.

Topology:
    Start -> A
    A -> [B1, B2]
    B1, B2 -> C
    C -> A         (loop back)
    C -> End       (terminate via End)

C is auto-rendezvous (incoming B1, B2 with no reciprocal outgoings).
A is also auto-rendezvous (incoming Start, C with outgoings B1, B2 — the
diff is {Start, C} which has size 2). Each loop iteration creates fresh
barrier instances. Iteration 1's resolver loops back; iteration 2's
resolver invokes End.
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


def test_p11_loop_terminating_via_end():
    reset_ids()
    topo = build_topology(
        nodes=["Start", "A", "B1", "B2", "C", "End"],
        flows=[
            "Start -> A",
            "A -> B1", "A -> B2",
            "B1 -> C", "B2 -> C",
            "C -> A",      # loop
            "C -> End",    # terminate
        ],
    )

    runtime = DeterministicRuntime()
    # A always parallel-invokes [B1, B2]; queue enough iterations
    for _ in range(3):
        runtime.queue_agent("A", StepResult(
            kind="PARALLEL_INVOKE",
            invocations=[Invocation("B1"), Invocation("B2")],
        ))
    for _ in range(4):
        runtime.queue_agent("B1", StepResult(kind="SINGLE_INVOKE", next_agent="C", value="b1"))
        runtime.queue_agent("B2", StepResult(kind="SINGLE_INVOKE", next_agent="C", value="b2"))
    # C: loop on iteration 1, terminate on iteration 2
    runtime.queue_agent("C", StepResult(kind="SINGLE_INVOKE", next_agent="A", request="retry"))
    runtime.queue_agent("C", StepResult(kind="SINGLE_INVOKE", next_agent="End", value="ANSWER"))

    orch = Orchestrator(topo, runtime, ConvergencePolicy())
    result = orch.run(task="q")

    assert result.success, result.error
    assert result.final_response == "ANSWER"
