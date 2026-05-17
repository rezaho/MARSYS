"""Session-08 real-world test: build → serialize-to-JSON → load-from-JSON → run.

This exercises exactly what Session 08 changed: a workflow whose topology
contains **explicit `Start` / `End` deterministic nodes** is serialized to a
canonical JSON document, written to disk, read back from disk, hydrated into a
runnable topology, and executed.

Pre-Session-08 this was impossible: `workflow_to_pydantic` raised
`NonSerializableTopologyError` the moment it saw a det-node, and `NodeSpec`
only had `node_type ∈ {user,agent,system,tool}` (no Start/End). Post-Session-08
the wire is total over `NodeKind = {agent,start,end,user}` and the JSON
round-trips losslessly.

Requires `ANTHROPIC_API_KEY` (loaded from the repo-root `.env` if not already
exported). Uses Claude Haiku — a full run is a few cents.

Run:

    source .venv/bin/activate
    python packages/framework/examples/session08_workflow_roundtrip.py
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path


# --- Load ANTHROPIC_API_KEY from the repo-root .env (framework doesn't auto-load) ---
def _load_repo_dotenv() -> None:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / ".env"
        if candidate.is_file():
            for raw in candidate.read_text().splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
            return


_load_repo_dotenv()
if not os.environ.get("ANTHROPIC_API_KEY"):
    sys.exit("ANTHROPIC_API_KEY not set and not found in a repo-root .env.")

from marsys.agents.registry import AgentRegistry
from marsys.agents.serialize import AgentSpec
from marsys.agents.utils import init_agent_logging
from marsys.coordination import Orchestra
from marsys.coordination.config import ExecutionConfig, StatusConfig
from marsys.models.serialize import ModelConfigSpec
from marsys.coordination.topology.serialize import (
    EdgeSpec,
    NodeSpec,
    TopologySpec,
    WorkflowDefinition,
    pydantic_to_topology,
    workflow_definition_schema,
)

OUT_DIR = Path(__file__).parent / "_session08_demo_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)
WORKFLOW_JSON = OUT_DIR / "my_workflow.json"

init_agent_logging(level=logging.ERROR, log_file=str(OUT_DIR / "_run.log"),
                   file_level=logging.DEBUG)

haiku = ModelConfigSpec(
    type="api",
    provider="anthropic",
    name="claude-haiku-4-5-20251001",
    temperature=0.3,
    max_tokens=2000,
)

# =========================================================================
# STAGE 1 — CREATE the workflow definition.
# Topology: Start -> Planner -> Writer -> End
#   * Start / End are EXPLICIT deterministic nodes (kind="start"/"end") —
#     the Session-08 capability. They have no agent_ref.
#   * Planner / Writer are agent nodes (kind="agent").
# =========================================================================
planner = AgentSpec(
    name="Planner",
    goal="Produce a one-paragraph outline for a short explainer.",
    instruction=(
        "Given a topic, write a 3-bullet outline (one line each). Then hand "
        "off to the Writer by calling `invoke_agent` with target='Writer' and "
        "the outline as the request."
    ),
    agent_model=haiku,
    memory_retention="single_run",
)
writer = AgentSpec(
    name="Writer",
    goal="Write the final explainer from the outline.",
    instruction=(
        "Given an outline, write a tight ~5-sentence explainer. When finished, "
        "call `terminate_workflow` with the explainer as the final answer."
    ),
    agent_model=haiku,
    memory_retention="single_run",
)

workflow = WorkflowDefinition(
    topology=TopologySpec(
        nodes=[
            NodeSpec(name="Start", kind="start"),                 # det-node
            NodeSpec(name="Planner", kind="agent", agent_ref="Planner"),
            NodeSpec(name="Writer", kind="agent", agent_ref="Writer"),
            NodeSpec(name="End", kind="end"),                     # det-node
        ],
        edges=[
            EdgeSpec(source="Start", target="Planner", edge_type="invoke"),
            EdgeSpec(source="Planner", target="Writer", edge_type="invoke"),
            EdgeSpec(source="Writer", target="End", edge_type="invoke"),
        ],
    ),
    agents={"Planner": planner, "Writer": writer},
)

# =========================================================================
# STAGE 2 — SERIALIZE to a JSON document and write it to disk.
# This is the canonical wire shape. Note `"kind": "start"` / `"end"` in the
# output — pre-Session-08 those node kinds did not exist and a det-node made
# serialization raise.
# =========================================================================
WORKFLOW_JSON.write_text(workflow.model_dump_json(indent=2))
print(f"STAGE 2  serialized → {WORKFLOW_JSON}")
_doc = json.loads(WORKFLOW_JSON.read_text())
print("         node kinds on the wire:",
      [(n["name"], n["kind"]) for n in _doc["topology"]["nodes"]])
print("         wire schema version:",
      workflow_definition_schema()["x-wire-schema-version"])

# =========================================================================
# STAGE 3 — READ the JSON back from disk into a WorkflowDefinition.
# =========================================================================
reloaded = WorkflowDefinition.model_validate_json(WORKFLOW_JSON.read_text())
assert [n.kind.value for n in reloaded.topology.nodes] == [
    "start", "agent", "agent", "end",
], "node kinds did not survive the JSON round-trip"
print("STAGE 3  reloaded from disk; det-node kinds intact ✓")

# =========================================================================
# STAGE 4 — HYDRATE into a runnable Topology.
# tool_registry / handler_registry are the DI seams (empty here: no custom
# tools, no explicit per-USER-node handler).
# =========================================================================
topology = pydantic_to_topology(reloaded, tool_registry={}, handler_registry={})
print(f"STAGE 4  hydrated runnable Topology: "
      f"{[ (n.name, n.kind.value) for n in topology.nodes ]}")


# =========================================================================
# STAGE 5 — RUN it through the orchestrator (the component chain S08 touched:
# pydantic_to_topology → analyzer materializes Start/End det-nodes → run).
# =========================================================================
async def main():
    try:
        result = await Orchestra.run(
            task="Explain why the sky is blue.",
            topology=topology,
            agent_registry=AgentRegistry,
            execution_config=ExecutionConfig(
                status=StatusConfig.from_verbosity(1),
                step_timeout=60.0,
            ),
            max_steps=12,
        )
        print(f"\n{'='*60}")
        print(f"STAGE 5  Success: {result.success}  steps={result.total_steps}  "
              f"{result.total_duration:.1f}s")
        print(f"Final answer:\n{str(result.final_response)[:600]}")
        assert result.success is True, f"run failed: {result.error}"
        print("\nALL STAGES PASSED ✓  (build → JSON → disk → load → hydrate → run)")
    finally:
        AgentRegistry.clear()


if __name__ == "__main__":
    asyncio.run(main())
