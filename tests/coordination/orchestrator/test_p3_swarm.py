"""P3: swarm (one-to-many fan-out, many-to-one fan-in).

Topology:
    Start -> Coord
    Coord -> [W1, W2, W3]; W1, W2, W3 -> Coord
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


def test_p3_swarm():
    reset_ids()
    topo = build_topology(
        nodes=["Start", "Coord", "W1", "W2", "W3"],
        flows=[
            "Start -> Coord",
            "Coord -> W1", "Coord -> W2", "Coord -> W3",
            "W1 -> Coord", "W2 -> Coord", "W3 -> Coord",
        ],
    )

    runtime = DeterministicRuntime()
    runtime.queue_agent("Coord", StepResult(
        kind="PARALLEL_INVOKE",
        invocations=[Invocation("W1"), Invocation("W2"), Invocation("W3")],
    ))
    for w in ("W1", "W2", "W3"):
        runtime.queue_agent(w, StepResult(kind="FINAL_RESPONSE", value=f"{w}_done"))
    runtime.queue_agent("Coord", StepResult(kind="FINAL_RESPONSE", value="aggregated"))

    orch = Orchestrator(topo, runtime, ConvergencePolicy())
    result = orch.run(task="task")

    assert result.success, result.error
    assert result.final_response == "aggregated"
