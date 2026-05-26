"""
Canonical-version twin of ``test_parallel_tracing.py`` — **standard Anthropic
API** variant (``provider="anthropic"`` + ``ANTHROPIC_API_KEY``), not OAuth.

Identical workflow, agents, and tracing to
``test_parallel_tracing_canonical.py``; the only difference is the model
provider. ``test_parallel_tracing_canonical.py`` uses ``anthropic-oauth``,
which structurally bypasses the credential/base-url validators — so it never
exercised the path that the Session-07 fix repaired. THIS script uses a
standard API-key provider, which is exactly that path: pre-fix it produced
an unrunnable ``ModelConfig`` (``base_url=None``/``api_key=None``); post-fix
the canonical run resolves the env credential and runs identically to the
string-notation baseline.

Requires ``ANTHROPIC_API_KEY``. This script loads it from the repo-root
``.env`` if not already exported (the framework does not auto-load ``.env``;
``python-dotenv`` is not a dependency, so a tiny inline loader is used).

Run it the same way the string-notation test is run:

    python packages/framework/benchmarks/GAIA/test_parallel_tracing_canonical_anthropic.py
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path


# --- Load ANTHROPIC_API_KEY from the repo-root .env -------------------------
# The framework intentionally does not auto-load .env and python-dotenv is not
# a dependency. An already-exported env var wins over the .env file (standard
# dotenv precedence). Without this, the canonical run-path correctly raises
# "API key for provider 'anthropic' not found" at materialization.
def _load_repo_dotenv() -> None:
    here = Path(__file__).resolve()
    for parent in here.parents:
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
    sys.exit(
        "ANTHROPIC_API_KEY is not set and was not found in a repo-root .env. "
        "Export it or add it to .env, then re-run. (This is the correct "
        "post-Session-07 behavior: the canonical run-path resolves the env "
        "credential exactly like a directly-constructed ModelConfig.)"
    )

from marsys.agents.registry import AgentRegistry
from marsys.agents.utils import init_agent_logging
from marsys.coordination import Orchestra
from marsys.coordination.config import ExecutionConfig, StatusConfig
from marsys.coordination.tracing import TracingConfig
from marsys.agents.serialize import AgentSpec
from marsys.models.serialize import ModelConfigSpec
from marsys.coordination.topology.serialize import (
    EdgeSpec,
    NodeSpec,
    TopologySpec,
    WorkflowDefinition,
    pydantic_to_topology,
)

TRACE_DIR = str(Path(__file__).parent / "results" / "traces_parallel_canonical_anthropic_test")
_TMP_LOG_FILE = str(Path(TRACE_DIR) / "_run.log")
Path(TRACE_DIR).mkdir(parents=True, exist_ok=True)

init_agent_logging(
    level=logging.ERROR,
    log_file=_TMP_LOG_FILE,
    file_level=logging.DEBUG,
)

print(f"Traces will be written to: {TRACE_DIR}")

# --- Canonical model spec: standard Anthropic API (NOT oauth) ---
# api_key is deliberately absent from ModelConfigSpec (storage boundary); the
# Session-07 runnable-by-default contract resolves ANTHROPIC_API_KEY from the
# environment, identical to a string-notation ModelConfig(provider="anthropic").
#
# Use the canonical dated model ID, NOT the "claude-haiku-4.5" friendly alias.
# The anthropic-oauth adapter has a MODEL_ALIASES table that resolves the
# alias; the standard `anthropic` adapter does not — it passes the name
# verbatim to the Anthropic Messages API, which 404s on the alias. This is
# the real API ID the oauth alias resolves to (anthropic_oauth.py:70).

haiku_model_spec = ModelConfigSpec(
    type="api",
    provider="anthropic",
    name="claude-haiku-4-5-20251001",
    temperature=0.3,
    max_tokens=4000,
)

# --- Canonical agent specs (mirror of the string test's Agent(...) calls) ---

coordinator_spec = AgentSpec(
    name="Coordinator",
    goal="Coordinate research by dispatching to workers in parallel.",
    instruction=(
        "You are a coordinator. On your FIRST response, dispatch to BOTH "
        "Researcher and FactChecker in parallel by calling `invoke_agent` "
        "with two invocations (one per worker). When you receive their "
        "results, synthesize a final answer and call `terminate_workflow` "
        "with the synthesized answer."
    ),
    model=haiku_model_spec,
    memory_retention="session",
)

researcher_spec = AgentSpec(
    name="Researcher",
    goal="Research a topic and provide findings.",
    instruction=(
        "You are a researcher. When given a topic, provide a brief 2-3 sentence "
        "research summary. Be concise. When done, return your findings to "
        "Coordinator by calling `invoke_agent` with target='Coordinator' and "
        "your findings as the request."
    ),
    model=haiku_model_spec,
    memory_retention="single_run",
)

fact_checker_spec = AgentSpec(
    name="FactChecker",
    goal="Verify claims and check facts.",
    instruction=(
        "You are a fact checker. When given a topic, provide 2-3 key facts "
        "that should be verified. Be concise. When done, return your facts to "
        "Coordinator by calling `invoke_agent` with target='Coordinator' and "
        "your facts as the request."
    ),
    model=haiku_model_spec,
    memory_retention="single_run",
)

# --- Canonical topology spec (mirror of the string-notation dict) ---
#
# String-notation equivalent was:
#   {
#     "agents": ["Coordinator", "Researcher", "FactChecker"],
#     "flows": ["Coordinator -> Researcher", "Coordinator -> FactChecker",
#               "Researcher -> Coordinator", "FactChecker -> Coordinator"],
#     "entry_point": "Coordinator",
#     "exit_points": ["Coordinator"],
#   }
# StringNotationConverter puts entry_point/exit_points into topology.metadata
# (string_converter.py:54-65); the canonical mirror does the same via
# TopologySpec.metadata, which pydantic_to_topology copies onto the Topology.

topology_spec = TopologySpec(
    nodes=[
        NodeSpec(name="Coordinator", kind="agent", agent_ref="Coordinator"),
        NodeSpec(name="Researcher", kind="agent", agent_ref="Researcher"),
        NodeSpec(name="FactChecker", kind="agent", agent_ref="FactChecker"),
    ],
    edges=[
        EdgeSpec(source="Coordinator", target="Researcher", edge_type="invoke"),
        EdgeSpec(source="Coordinator", target="FactChecker", edge_type="invoke"),
        EdgeSpec(source="Researcher", target="Coordinator", edge_type="invoke"),
        EdgeSpec(source="FactChecker", target="Coordinator", edge_type="invoke"),
    ],
    metadata={"entry_point": "Coordinator", "exit_points": ["Coordinator"]},
)

workflow = WorkflowDefinition(
    topology=topology_spec,
    agents={
        "Coordinator": coordinator_spec,
        "Researcher": researcher_spec,
        "FactChecker": fact_checker_spec,
    },
)

# Hydrate the canonical spec into a runnable Topology. No user tools in this
# workflow, so an empty tool registry is correct. This is the line that, pre
# Session-07, silently produced agents with base_url=None/api_key=None for
# provider="anthropic"; post-fix it resolves ANTHROPIC_API_KEY from the env.
# pydantic_to_topology is async (ADR-009 / S09 B′); this top-level script
# owns its own loop via asyncio.run (the caller-side contract).
topology = asyncio.run(pydantic_to_topology(workflow, tool_registry={}))


async def main():
    try:
        result = await Orchestra.run(
            task="What is the speed of light and who first measured it?",
            topology=topology,
            agent_registry=AgentRegistry,
            execution_config=ExecutionConfig(
                status=StatusConfig.from_verbosity(2),
                step_timeout=60.0,
                tracing=TracingConfig(
                    enabled=True,
                    output_dir=TRACE_DIR,
                    include_message_content=True,
                ),
            ),
            max_steps=20,
        )

        print(f"\n{'='*60}")
        print(f"Success: {result.success}")
        print(f"Total steps: {result.total_steps}")
        print(f"Duration: {result.total_duration:.1f}s")
        print(f"Final response: {str(result.final_response)[:200]}")

        # The point of this script is to prove the canonical (WorkflowDefinition)
        # run-path reaches the same successful outcome as the string-notation
        # baseline. Assert it so a regression fails loudly (non-zero exit)
        # instead of printing a failed run and exiting 0.
        assert result.success is True, (
            f"canonical workflow did not succeed: success={result.success}"
        )
        final_text = str(result.final_response).strip()
        assert final_text, "canonical workflow produced an empty final response"

        # Find and print trace structure from latest file
        trace_dir = Path(TRACE_DIR)
        trace_files = sorted(trace_dir.glob("*.json"), key=lambda f: f.stat().st_mtime)
        if trace_files:
            latest = trace_files[-1]
            with open(latest) as f:
                trace = json.load(f)

            # Rename log file to match trace session ID
            session_id = trace.get("session_id", "unknown")
            final_log = Path(TRACE_DIR) / f"{session_id}.log"
            tmp_log = Path(_TMP_LOG_FILE)
            if tmp_log.exists():
                tmp_log.rename(final_log)
                print(f"Log: {final_log}")

            print(f"\n{'='*60}")
            print(f"Trace: {latest.name}")
            print_tree(trace["root_span"])

            # Convergence check
            print(f"\n{'='*60}")
            print("CONVERGENCE CHECK:")
            check_convergence(trace["root_span"])

    finally:
        AgentRegistry.clear()


def print_tree(span, indent=0):
    prefix = "  " * indent
    kind = span.get("kind", "?")
    name = span.get("name", "?")
    status = span.get("status", "?")
    attrs = span.get("attributes", {})
    children = span.get("children", [])
    events = span.get("events", [])
    links = span.get("links", [])

    extra = ""
    if kind == "branch":
        extra = f" src={attrs.get('source_agent')} tgt={attrs.get('target_agents')} trigger={attrs.get('trigger_type')}"
    elif kind == "generation":
        extra = f" model={attrs.get('model_name','')} in={attrs.get('prompt_tokens')} out={attrs.get('completion_tokens')}"
    elif kind == "tool":
        extra = f" tool={attrs.get('tool_name')}"

    print(f"{prefix}[{kind}] {name} (status={status} ch={len(children)} ev={len(events)} lnk={len(links)}){extra}")

    # Show links if present
    for link in links:
        rel = link.get("relationship", "?")
        linked_id = link.get("linked_span_id", "?")[:8]
        link_attrs = link.get("attributes", {})
        print(f"{prefix}  >> LINK: {rel} -> span {linked_id} {link_attrs}")

    # Show events if present
    for event in events:
        ev_name = event.get("name", "?")
        ev_attrs = event.get("attributes", {})
        print(f"{prefix}  ** EVENT: {ev_name} {ev_attrs}")

    for child in children:
        print_tree(child, indent + 1)


def check_convergence(root_span):
    """Check if convergence relationships are captured in the trace."""
    branch_spans = []
    all_spans = []
    convergence_steps = []

    def walk(span, parent_kind=None):
        all_spans.append(span)
        kind = span.get("kind")
        if kind == "branch":
            branch_spans.append(span)
        # A step directly under root that comes AFTER branch spans = convergence candidate
        if kind == "step" and parent_kind == "execution":
            attrs = span.get("attributes", {})
            req = attrs.get("request_summary", "")
            if req and "original_invocations" in str(req):
                convergence_steps.append(span)
        for child in span.get("children", []):
            walk(child, kind)

    walk(root_span)

    print(f"  Branch spans found: {len(branch_spans)}")
    for b in branch_spans:
        attrs = b.get("attributes", {})
        links = b.get("links", [])
        events = b.get("events", [])
        print(f"    - {b['name']} (id={b['span_id'][:8]}) links={len(links)} events={len(events)}")

    print(f"  Convergence step candidates: {len(convergence_steps)}")
    for s in convergence_steps:
        links = s.get("links", [])
        events = s.get("events", [])
        print(f"    - {s['name']} (id={s['span_id'][:8]}) links={len(links)} events={len(events)}")
        if not links:
            print(f"      !! NO LINKS - cannot trace which branches fed into this step")

    if branch_spans and not any(s.get("links") for s in all_spans):
        print("\n  RESULT: Convergence NOT captured in trace.")
        print("  The trace shows branches ran in parallel, but there is no")
        print("  link connecting the convergence step back to its source branches.")
    elif any(s.get("links") for s in all_spans):
        print("\n  RESULT: Convergence IS captured via links.")


if __name__ == "__main__":
    asyncio.run(main())
