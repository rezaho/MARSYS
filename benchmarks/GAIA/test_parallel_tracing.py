"""
Minimal parallel multi-agent test to verify tracing branch spans and convergence.

Topology: Coordinator dispatches to Researcher + FactChecker in parallel,
they converge back at ReportWriter who synthesizes a final report.

    Coordinator ──┬──> Researcher  ──┬──> ReportWriter ──> Coordinator
                  └──> FactChecker ──┘
"""

import asyncio
import json
import logging
from pathlib import Path

from marsys.agents import Agent
from marsys.agents.registry import AgentRegistry
from marsys.agents.utils import init_agent_logging
from marsys.coordination import Orchestra
from marsys.coordination.config import ExecutionConfig, StatusConfig
from marsys.coordination.tracing import TracingConfig
from marsys.models.models import ModelConfig

TRACE_DIR = str(Path(__file__).parent / "results" / "traces_parallel_test")
_TMP_LOG_FILE = str(Path(TRACE_DIR) / "_run.log")
Path(TRACE_DIR).mkdir(parents=True, exist_ok=True)

init_agent_logging(
    level=logging.ERROR,
    log_file=_TMP_LOG_FILE,
    file_level=logging.DEBUG,
)

print(f"Traces will be written to: {TRACE_DIR}")

haiku_config = ModelConfig(
    type="api",
    provider="anthropic-oauth",
    name="claude-haiku-4.5",
    oauth_profile="marsys-2",
    temperature=0.3,
    max_tokens=4000,
)

# --- Agents ---

coordinator = Agent(
    model_config=haiku_config,
    name="Coordinator",
    goal="Coordinate research by dispatching to workers in parallel.",
    instruction=(
        "You are a coordinator. When given a research task, you MUST dispatch "
        "to BOTH Researcher and FactChecker in parallel by using parallel_invoke. "
        "When you receive their results, write a final answer.\n\n"
        "IMPORTANT: On your FIRST response, always use parallel_invoke to send "
        "the task to both Researcher and FactChecker simultaneously."
    ),
    memory_retention="session",
)

researcher = Agent(
    model_config=haiku_config,
    name="Researcher",
    goal="Research a topic and provide findings.",
    instruction=(
        "You are a researcher. When given a topic, provide a brief 2-3 sentence "
        "research summary. Be concise. Respond with a final_response."
    ),
    memory_retention="single_run",
)

fact_checker = Agent(
    model_config=haiku_config,
    name="FactChecker",
    goal="Verify claims and check facts.",
    instruction=(
        "You are a fact checker. When given a topic, provide 2-3 key facts "
        "that should be verified. Be concise. Respond with a final_response."
    ),
    memory_retention="single_run",
)

# --- Topology ---

topology = {
    "agents": ["Coordinator", "Researcher", "FactChecker"],
    "flows": [
        "Coordinator -> Researcher",
        "Coordinator -> FactChecker",
        "Researcher -> Coordinator",
        "FactChecker -> Coordinator",
    ],
    "entry_point": "Coordinator",
    "exit_points": ["Coordinator"],
}


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
                    detail_level="verbose",
                    include_message_content=True,
                    max_content_length=2000,
                ),
            ),
            max_steps=20,
        )

        print(f"\n{'='*60}")
        print(f"Success: {result.success}")
        print(f"Total steps: {result.total_steps}")
        print(f"Duration: {result.total_duration:.1f}s")
        print(f"Final response: {str(result.final_response)[:200]}")

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
