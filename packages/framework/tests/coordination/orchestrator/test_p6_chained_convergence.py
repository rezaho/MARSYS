"""P6: chained convergence (cross-group rendezvous — the hard case).

Topology:
    Start -> A
    A -> [B1, B2]
    B1 -> [C1, C2]
    B2 -> [D, C2]   # B2 dispatches to both D and C2
    C1 -> D
    C2 -> D         # C2 is auto-rendezvous (incoming from B1, B2)
    D -> End        # D is auto-rendezvous (incoming from B2, C1, C2)

Cross-group rendezvous: C2 receives from both B1 and B2 (different fork
groups). D receives from B2's branch, C1, and the C2 resolver. Both C2
and D are auto-detected via reciprocal-edge subtraction.
"""
from __future__ import annotations

import pytest

from marsys.coordination.execution.deterministic_runtime import DeterministicRuntime
from marsys.coordination.execution.orchestrator import Orchestrator
from marsys.coordination.execution.orchestrator_types import (
    ConvergencePolicy,
    Invocation,
    StepResult,
    reset_ids,
)

from ._helpers import build_topology


@pytest.mark.asyncio
async def test_p6_chained_convergence():
    reset_ids()
    topo = build_topology(
        nodes=["Start", "A", "B1", "B2", "C1", "C2", "D", "End"],
        flows=[
            "Start -> A",
            "A -> B1", "A -> B2",
            "B1 -> C1", "B1 -> C2",
            "B2 -> D", "B2 -> C2",
            "C1 -> D",
            "C2 -> D",
            "D -> End",
        ],
    )

    # C2 and D are auto-rendezvous
    assert topo.is_convergence("C2"), "C2 should be auto-rendezvous"
    assert topo.is_convergence("D"), "D should be auto-rendezvous"
    # D's predecessor convergence is C2 (cycle break for the C2->D edge)
    assert topo.predecessor_convergences("D") == frozenset({"C2"})

    runtime = DeterministicRuntime()
    runtime.queue_agent("A", StepResult(
        kind="PARALLEL_INVOKE",
        invocations=[Invocation("B1"), Invocation("B2")],
    ))
    runtime.queue_agent("B1", StepResult(
        kind="PARALLEL_INVOKE",
        invocations=[Invocation("C1"), Invocation("C2")],
    ))
    runtime.queue_agent("B2", StepResult(
        kind="PARALLEL_INVOKE",
        invocations=[Invocation("D"), Invocation("C2")],
    ))
    runtime.queue_agent("C1", StepResult(kind="SINGLE_INVOKE", next_agent="D", value="c1_val"))
    # C2 is the rendezvous resolver — invoked when both B1, B2 paths land
    runtime.queue_agent("C2", StepResult(kind="SINGLE_INVOKE", next_agent="D", value="c2_val"))
    # D is the rendezvous resolver — invoked when all upstream contributions land
    runtime.queue_agent("D", StepResult(kind="SINGLE_INVOKE", next_agent="End", value="answer"))

    orch = Orchestrator(topo, runtime, ConvergencePolicy())
    result = await orch.run(task="q")

    assert result.success, result.error
    assert result.final_response == "answer"
