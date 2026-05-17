"""Direct marsys repro — isolate Spren from the framework for WF-BUG-RUN-3e.

User hypothesis: the marsys framework DOES return the final answer for the
exact Start->Agent->End workflow Spren runs; Spren is either invoking the
framework non-canonically or not reading the result correctly.

This bypasses the FastAPI sidecar entirely. It builds the SAME
WorkflowDefinition the probe uses, then runs it three ways, dumping the
*full* OrchestraResult each time (every field + every branch_result, since
the framework hides the real per-branch error behind result.error):

  A) Spren's exact path: materialize_run() -> Orchestra(...).execute(...)
     (mirrors packages/spren/src/spren/runs/lifecycle.py:186-307)
  B) Same materialized bundle, canonical entrypoint: Orchestra.run(...)
  C) Hand-built minimal marsys (default ExecutionConfig, plain Agent) via
     Orchestra.run(...) — the "pure framework / test_mas.py" path.

Post-ADR-008: the framework resolves credentials per-provider. Export the
standard ``ANTHROPIC_API_KEY`` (no Spren-prefixed variable). Usage:
    set -a && . .env && set +a && \
      uv run python scripts/scenarios/marsys_direct_probe.py
"""
from __future__ import annotations

import asyncio
import dataclasses
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

MODEL = "claude-haiku-4-5-20251001"
TASK = "Say hello."
AGENT_INSTRUCTION = "You are a helpful assistant. Answer concisely."


def _dump_result(label: str, result: Any) -> None:
    print(f"\n========== {label} ==========", flush=True)
    print(f"type: {type(result)!r}", flush=True)
    for attr in ("success", "error", "total_steps", "total_duration"):
        print(f"  {attr} = {getattr(result, attr, '<MISSING>')!r}", flush=True)
    fr = getattr(result, "final_response", "<MISSING>")
    print(f"  final_response = {fr!r}", flush=True)
    if hasattr(result, "get_final_response_as_text"):
        try:
            print(f"  get_final_response_as_text() = {result.get_final_response_as_text()!r}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  get_final_response_as_text() raised {exc!r}", flush=True)
    md = getattr(result, "metadata", None)
    print(f"  metadata = {md!r}", flush=True)
    branches = getattr(result, "branch_results", None)
    if branches is not None:
        print(f"  branch_results ({len(branches)}):", flush=True)
        for br in branches:
            if dataclasses.is_dataclass(br):
                print(f"    - {dataclasses.asdict(br)!r}", flush=True)
            else:
                print(
                    "    - "
                    + repr({
                        k: getattr(br, k, None)
                        for k in ("branch_id", "branch_name", "success", "error", "input", "output", "final_response")
                        if hasattr(br, k)
                    }),
                    flush=True,
                )


def _definition():
    from spren.models import AgentSpec, ModelConfigSpec, WorkflowDefinition
    from spren.models.topology import EdgeSpec, NodeKind, NodeSpec, TopologySpec

    return WorkflowDefinition(
        topology=TopologySpec(
            nodes=[
                NodeSpec(name="Start", kind=NodeKind.START),
                NodeSpec(name="assistant", kind=NodeKind.AGENT, agent_ref="assistant", is_convergence_point=True),
                NodeSpec(name="End", kind=NodeKind.END),
            ],
            edges=[
                EdgeSpec(source="Start", target="assistant", edge_type="invoke"),
                EdgeSpec(source="assistant", target="End", edge_type="invoke"),
            ],
        ),
        agents={
            "assistant": AgentSpec(
                agent_model=ModelConfigSpec(type="api", name=MODEL, provider="anthropic"),
                name="assistant",
                goal="Answer the user's question.",
                instruction=AGENT_INSTRUCTION,
                tools=[],
                memory_retention="session",
                allowed_peers=[],
            ),
        },
    )


async def variant_a_spren_path() -> None:
    from marsys.agents.registry import AgentRegistry
    from marsys.coordination.orchestra import Orchestra
    from marsys.coordination.state.storage import FileStorageBackend
    from spren.runs.materialize import materialize_run

    AgentRegistry.clear()
    bundle = materialize_run(definition=_definition(), enable_aggui=True)
    tmp = Path(tempfile.mkdtemp(prefix="marsys-direct-A-"))
    orchestra = Orchestra(
        agent_registry=AgentRegistry,
        execution_config=bundle.execution_config,
        storage_backend=FileStorageBackend(tmp / "data" / "runs"),
    )
    result = await orchestra.execute(
        task=TASK, topology=bundle.topology, context={"session_id": "direct-A"}
    )
    _dump_result("A: Spren materialize_run + Orchestra(...).execute()", result)
    AgentRegistry.clear()


async def variant_b_canonical_same_bundle() -> None:
    from marsys.agents.registry import AgentRegistry
    from marsys.coordination.orchestra import Orchestra
    from spren.runs.materialize import materialize_run

    AgentRegistry.clear()
    bundle = materialize_run(definition=_definition(), enable_aggui=True)
    result = await Orchestra.run(
        task=TASK,
        topology=bundle.topology,
        agent_registry=AgentRegistry,
        execution_config=bundle.execution_config,
        context={"session_id": "direct-B"},
    )
    _dump_result("B: same bundle via canonical Orchestra.run()", result)
    AgentRegistry.clear()


async def variant_c_handbuilt_pure() -> None:
    """Pure framework: build it by hand the way a marsys test would.

    Post-ADR-008: Start/End are plain ``Node(kind=...)`` (no
    StartNode/EndNode instances); the framework resolves the key
    per-provider from the standard ``ANTHROPIC_API_KEY`` (no api_key arg,
    no Spren-prefixed variable).
    """
    from marsys.agents.agents import Agent
    from marsys.agents.registry import AgentRegistry
    from marsys.coordination.orchestra import Orchestra
    from marsys.coordination.topology.core import Edge, Node, NodeKind, Topology
    from marsys.models.models import ModelConfig

    AgentRegistry.clear()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n[C] SKIPPED — ANTHROPIC_API_KEY not in env", flush=True)
        return
    Agent(
        model_config=ModelConfig(type="api", name=MODEL, provider="anthropic"),
        goal="Answer the user's question.",
        instruction=AGENT_INSTRUCTION,
        name="assistant",
    )
    topology = Topology(
        nodes=[
            Node(name="Start", kind=NodeKind.START),
            Node(name="assistant", kind=NodeKind.AGENT, agent_ref="assistant"),
            Node(name="End", kind=NodeKind.END),
        ],
        edges=[
            Edge(source="Start", target="assistant"),
            Edge(source="assistant", target="End"),
        ],
    )
    result = await Orchestra.run(task=TASK, topology=topology, agent_registry=AgentRegistry, context={"session_id": "direct-C"})
    _dump_result("C: hand-built pure marsys via Orchestra.run() (default ExecutionConfig)", result)
    AgentRegistry.clear()


async def main() -> int:
    for fn in (variant_a_spren_path, variant_b_canonical_same_bundle, variant_c_handbuilt_pure):
        try:
            await fn()
        except Exception as exc:  # noqa: BLE001
            import traceback
            print(f"\n!!! {fn.__name__} raised: {exc!r}", flush=True)
            traceback.print_exc()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
