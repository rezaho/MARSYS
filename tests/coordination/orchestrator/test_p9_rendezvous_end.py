"""P9: rendezvous resolver invokes End to deliver the final answer.

Topology:
    Start -> A
    A -> [B1, B2]
    B1, B2 -> C       (auto-rendezvous on C)
    C -> End

Confirms the End det-node's deliver_to_root path: when the rendezvous
resolver invokes End, value flows directly to ROOT.
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


def test_p9_rendezvous_to_end():
    reset_ids()
    topo = build_topology(
        nodes=["Start", "A", "B1", "B2", "C", "End"],
        flows=[
            "Start -> A",
            "A -> B1", "A -> B2",
            "B1 -> C", "B2 -> C",
            "C -> End",
        ],
    )

    runtime = DeterministicRuntime()
    runtime.queue_agent("A", StepResult(
        kind="PARALLEL_INVOKE",
        invocations=[Invocation("B1"), Invocation("B2")],
    ))
    runtime.queue_agent("B1", StepResult(kind="SINGLE_INVOKE", next_agent="C", value="b1_ans"))
    runtime.queue_agent("B2", StepResult(kind="SINGLE_INVOKE", next_agent="C", value="b2_ans"))
    runtime.queue_agent("C", StepResult(kind="SINGLE_INVOKE", next_agent="End", value="FINAL"))

    orch = Orchestrator(topo, runtime, ConvergencePolicy())
    result = orch.run(task="q")

    assert result.success, result.error
    assert result.final_response == "FINAL"
