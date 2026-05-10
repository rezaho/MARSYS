"""Full-payload capture + OTel export — live-run smoke check.

PURPOSE
  End-to-end verification of the Phase-1 capture layer (full LLM payloads
  via the model-wrapper context manager) and the Phase-2 OTel exporter
  (LangSmith / Phoenix / Langfuse fan-out). Five agents collaborate to
  discover the secret word "MARS" so the trace exercises the full
  spectrum of span kinds — execution, branch, step, generation,
  compaction (when triggered), tool, with convergence links.

EXERCISES
  * coordination.tracing.capture.capture_llm_call — emits LLMRequestEvent
    / LLMResponseEvent with full messages + tool schemas + sampling +
    response content
  * coordination.tracing.collector.TraceCollector — _handle_llm_request /
    _handle_llm_response wire LLM payloads into generation spans on the
    streaming path
  * coordination.tracing.writers.ndjson_writer.NDJSONTraceWriter — local
    source-of-truth file
  * coordination.tracing.writers.otel_writer.OtelTraceWriter — opt-in OTel
    export to LangSmith via OTLP/HTTP using gen_ai.* semconv
  * coordination.orchestra.Orchestra.run — TracingConfig.sinks plumbing,
    finalize-block close timeout

TOPOLOGY
  Orchestrator -+--> AgentA ---------------+--> Orchestrator
                +--> AgentB ---------------+
                +--> AgentC --> AgentD ----+

  AgentA holds "M". AgentB calls a tool to discover "A". AgentC holds
  "R" and forwards to AgentD. AgentD holds "S", combines C's letter
  with its own, and returns both to the Orchestrator. Orchestrator
  dispatches A/B/C in parallel, then assembles "MARS" from the three
  incoming branches (A, B, D).

RUN
  cd packages/framework && source ../../.venv/bin/activate
  python live_tests/tracing/secret_word_pipeline.py --output-dir /tmp/marsys_runs/secret-001

KEY ARGS
  --task TASK         User task sent to the Orchestrator. Default: "Collect
                      the secret letters and assemble the word." Keep
                      stable for comparable traces.
  --output-dir DIR    Where run.log, the .ndjson trace, and the messages/
                      sidecar dir land. REQUIRED for automation; auto-
                      generated under ./_runs/secret_word/<timestamp>/.
  --max-steps N       Orchestra max_steps. Default 30.
  --print-tree        Pretty-print the reconstructed TraceTree.
  --langsmith         Enable OTel export to LangSmith. Reads LANGSMITH_API_KEY
                      / LANGSMITH_PROJECT / LANGSMITH_OTEL_ENDPOINT from env.
                      Without this flag, OtelTraceWriter is not wired —
                      NDJSON-only run.
  --provider {openrouter,azure}
                      LLM provider. Default 'openrouter'. 'azure' reads
                      AZURE_OPENAI_{API_KEY,ENDPOINT,DEPLOYMENT} and routes
                      through the OpenAI adapter (endpoint must be the v1
                      inference path: .../openai/v1/).
  --model-name NAME   OpenRouter model id, or — when --provider=azure —
                      the deployment name. Defaults to the corresponding
                      env var (OPEN_ROUTER_MODEL / AZURE_OPENAI_DEPLOYMENT).

OUTPUTS
  <output-dir>/run.log                DEBUG-level framework log
  <output-dir>/<trace_id>.ndjson      Streaming trace file
  <output-dir>/messages/<sha>.json    Content-addressed message blobs
  Last stdout line                     Single-line JSON summary
  Exit code                            0 = all checks passed, 1 otherwise

  Summary keys: output_dir, trace_file, log_file, all_checks_passed,
  checks (per-check bool), capture_stats, langsmith_enabled.

REQUIREMENTS
  * For --provider=openrouter (default): OPENROUTER_API_KEY (or
    OPEN_ROUTER_API_KEY); network access to openrouter.ai.
  * For --provider=azure: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT
    (must be the v1 path, e.g. https://<resource>.openai.azure.com/openai/v1/),
    and AZURE_OPENAI_DEPLOYMENT (or pass --model-name explicitly).
  * LANGSMITH_API_KEY only when --langsmith is passed.
  * Network access to api.smith.langchain.com when --langsmith is on.
  * pip install 'marsys[tracing-otel]' for the OtelTraceWriter import.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from marsys.agents import Agent
from marsys.agents.registry import AgentRegistry
from marsys.agents.utils import init_agent_logging
from marsys.coordination import Orchestra
from marsys.coordination.config import ExecutionConfig, StatusConfig
from marsys.coordination.tracing import (
    NDJSONTraceReader,
    TraceTree,
    TracingConfig,
)
from marsys.models.models import ModelConfig


DEFAULT_TASK = (
    "Collect all secret letters from the agents and assemble the secret word."
)
DEFAULT_MODEL = "anthropic/claude-haiku-4.5"
SUPPORTED_PROVIDERS = ("openrouter", "azure")


# ── Tools ──────────────────────────────────────────────────────────────────


def reveal_secret_letter() -> str:
    """Reveal the secret letter assigned to this agent.

    Returns:
        The single-character secret letter.
    """
    return "A"


# ── Topology ───────────────────────────────────────────────────────────────


def _build_agents(model_config: ModelConfig) -> Dict[str, Agent]:
    orchestrator = Agent(
        model_config=model_config,
        name="Orchestrator",
        goal="Dispatch AgentA/B/C in parallel, then assemble the secret word from their replies.",
        instruction=(
            "You run in TWO turns.\n"
            "TURN 1 (first call, no prior results): make a single `invoke_agent` "
            "tool call containing THREE invocations — one for AgentA, one for AgentB, "
            "one for AgentC. Each invocation's `request` field must be "
            "'Provide your secret letter.' Do NOT write any reasoning. "
            "Do NOT create plans. Call `invoke_agent` immediately.\n"
            "TURN 2 (resume, aggregated results present in your input): extract every "
            "letter from the aggregated results (letters from AgentA, AgentB, and AgentD), "
            "rearrange them to form a real English word, then call `terminate_workflow` "
            "with the response 'The secret word is: <word>'. Do NOT call any other tools."
        ),
        memory_retention="single_run",
    )
    agent_a = Agent(
        model_config=model_config,
        name="AgentA",
        goal="Return letter M to the Orchestrator.",
        instruction=(
            "Your secret letter is M. Your ONLY action: make a single `invoke_agent` "
            "call with one invocation: agent_name='Orchestrator', "
            "request='Letter from AgentA: M'. Do NOT write any reasoning. "
            "Do NOT create plans. Call `invoke_agent` immediately."
        ),
        memory_retention="single_run",
    )
    agent_b = Agent(
        model_config=model_config,
        name="AgentB",
        goal="Discover your letter via tool, then return it to the Orchestrator.",
        instruction=(
            "You MUST end EVERY turn with a tool call. Plain-text replies are forbidden.\n"
            "TURN 1: call the `reveal_secret_letter` tool (no arguments). After the tool "
            "returns, your NEXT response MUST be a `invoke_agent` call. Do NOT reply with "
            "plain text saying you already revealed the letter.\n"
            "TURN 2: call `invoke_agent` with one invocation — agent_name='Orchestrator', "
            "request='Letter from AgentB: <the letter the tool returned>'. "
            "Two tool calls total: one `reveal_secret_letter`, then one `invoke_agent`. "
            "Never reply with plain text. If you find yourself writing prose, STOP and "
            "issue the next tool call instead."
        ),
        tools={"reveal_secret_letter": reveal_secret_letter},
        memory_retention="single_run",
    )
    agent_c = Agent(
        model_config=model_config,
        name="AgentC",
        goal="Forward your letter R to AgentD.",
        instruction=(
            "Your secret letter is R. Your ONLY action: make a single `invoke_agent` "
            "call with one invocation — agent_name='AgentD', "
            "request='Letter from AgentC: R. Append your letter and forward the "
            "combined letters to the Orchestrator.' "
            "Do NOT write any reasoning. Do NOT create plans. Do NOT invoke yourself. "
            "Call `invoke_agent` immediately."
        ),
        memory_retention="single_run",
    )
    agent_d = Agent(
        model_config=model_config,
        name="AgentD",
        goal="Combine AgentC's letter with your own and return both to the Orchestrator.",
        instruction=(
            "Your secret letter is S. Your incoming request contains AgentC's letter (R). "
            "Your ONLY action: make a single `invoke_agent` call with one invocation — "
            "agent_name='Orchestrator', "
            "request='Letters: R (from AgentC), S (from AgentD)'. "
            "Do NOT write any reasoning. Do NOT create plans. Do NOT invoke yourself. "
            "Do NOT invoke AgentC. Call `invoke_agent` immediately."
        ),
        memory_retention="single_run",
    )
    return {
        "Orchestrator": orchestrator,
        "AgentA": agent_a,
        "AgentB": agent_b,
        "AgentC": agent_c,
        "AgentD": agent_d,
    }


_TOPOLOGY = {
    "agents": ["Start", "Orchestrator", "AgentA", "AgentB", "AgentC", "AgentD", "End"],
    "flows": [
        "Start -> Orchestrator",
        "Orchestrator -> AgentA",
        "Orchestrator -> AgentB",
        "Orchestrator -> AgentC",
        "AgentA -> Orchestrator",
        "AgentB -> Orchestrator",
        "AgentC -> AgentD",
        "AgentD -> Orchestrator",
        "Orchestrator -> End",
    ],
    "rules": ["timeout(180)"],
}


# ── Model-config builders ──────────────────────────────────────────────────


def _build_model_config(args: argparse.Namespace) -> "ModelConfig":
    """Construct the ``ModelConfig`` for the chosen provider.

    ``azure`` is not a first-class provider — it routes through the
    stock OpenAI adapter, which is compatible with the Azure v1
    inference endpoint (``.../openai/v1/``). Raises ``SystemExit`` with
    a clear message when required env vars are missing.
    """
    if args.provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPEN_ROUTER_API_KEY")
        if not api_key:
            print("ERROR: --provider=openrouter requires OPENROUTER_API_KEY (or OPEN_ROUTER_API_KEY).")
            raise SystemExit(1)
        # CLI override > env > hardcoded fallback.
        model_name = args.model_name or os.getenv("OPEN_ROUTER_MODEL") or DEFAULT_MODEL
        return ModelConfig(
            type="api",
            provider="openrouter",
            name=model_name,
            api_key=api_key,
            temperature=0.2,
            max_tokens=2000,
        )

    if args.provider == "azure":
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        # No fallback to OPEN_ROUTER_MODEL — Azure deployment names and
        # OpenRouter model ids are separate namespaces.
        deployment = args.model_name or os.getenv("AZURE_OPENAI_DEPLOYMENT")

        missing = [
            k for k, v in (
                ("AZURE_OPENAI_API_KEY", api_key),
                ("AZURE_OPENAI_ENDPOINT", endpoint),
                ("AZURE_OPENAI_DEPLOYMENT (or --model-name)", deployment),
            ) if not v
        ]
        if missing:
            print(f"ERROR: --provider=azure requires env vars: {', '.join(missing)}")
            raise SystemExit(1)

        if "/openai/v1" not in endpoint:
            print(
                "WARNING: AZURE_OPENAI_ENDPOINT should point at the v1 "
                "inference path (.../openai/v1/)."
            )

        # provider="openai" — the CLI flag is just an env-var-source
        # selector; the v1 endpoint speaks the same Responses API.
        return ModelConfig(
            type="api",
            provider="openai",
            name=deployment,
            base_url=endpoint,
            api_key=api_key,
            temperature=0.2,
            max_tokens=2000,
        )

    # argparse `choices=` should make this unreachable.
    print(f"ERROR: unsupported --provider={args.provider!r}; expected one of {SUPPORTED_PROVIDERS}.")
    raise SystemExit(1)


# ── OTel sink builder (opt-in) ──────────────────────────────────────────────


def _build_otel_sink() -> Any:
    """Build an OtelTraceWriter from LangSmith env vars. Returns None
    when LANGSMITH_API_KEY is missing — callers should treat that as
    "OTel export disabled."
    """
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        return None
    project = os.getenv("LANGSMITH_PROJECT", "default")
    endpoint = os.getenv(
        "LANGSMITH_OTEL_ENDPOINT",
        "https://api.smith.langchain.com/otel/v1/traces",
    )
    # Lazy import: only require the OTel extra when --langsmith is set.
    from marsys.coordination.tracing import OtelTraceWriter

    # service.name MUST match LANGSMITH_PROJECT — a mismatch silently
    # routes spans to an auto-created side-project.
    return OtelTraceWriter(
        endpoint=endpoint,
        headers={
            "x-api-key": api_key,
            "Langsmith-Project": project,
        },
        service_name=project,
    )


# ── Inspection helpers ─────────────────────────────────────────────────────


def print_tree(span_dict: Dict[str, Any], indent: int = 0) -> None:
    prefix = "  " * indent
    kind = span_dict.get("kind", "?")
    name = span_dict.get("name", "?")
    status = span_dict.get("status", "?")
    attrs = span_dict.get("attributes", {})
    children = span_dict.get("children", [])
    extra = ""
    if kind == "branch":
        extra = (
            f" src={attrs.get('source_agent')} tgt={attrs.get('target_agents')}"
        )
    elif kind in ("generation", "compaction"):
        sampling = attrs.get("sampling_params") or {}
        extra = (
            f" model={attrs.get('model_name','?')} "
            f"temp={sampling.get('temperature','?')} "
            f"msgs={len(attrs.get('input_messages') or [])}"
        )
    elif kind == "tool":
        extra = f" tool={attrs.get('tool_name')}"
    print(f"{prefix}[{kind}] {name} (status={status}){extra}")
    for child in children:
        print_tree(child, indent + 1)


def collect_capture_stats(root: Dict[str, Any]) -> Dict[str, Any]:
    """Walk the tree; count generation spans and check the new full-payload
    fields landed (input_messages, response_content, sampling_params)."""
    gen_spans: List[Dict[str, Any]] = []
    compaction_spans: List[Dict[str, Any]] = []
    branch_spans: List[Dict[str, Any]] = []
    tool_spans: List[Dict[str, Any]] = []

    def walk(s: Dict[str, Any]) -> None:
        kind = s.get("kind")
        if kind == "generation":
            gen_spans.append(s)
        elif kind == "compaction":
            compaction_spans.append(s)
        elif kind == "branch":
            branch_spans.append(s)
        elif kind == "tool":
            tool_spans.append(s)
        for c in s.get("children", []):
            walk(c)

    walk(root)

    def _has_attr(s: Dict[str, Any], key: str) -> bool:
        v = (s.get("attributes") or {}).get(key)
        return v is not None and v != [] and v != {}

    return {
        "branch_count": len(branch_spans),
        "tool_count": len(tool_spans),
        "generation_count": len(gen_spans),
        "compaction_count": len(compaction_spans),
        "gens_with_input_messages": sum(
            1 for s in gen_spans if _has_attr(s, "input_messages")
        ),
        "gens_with_response_content": sum(
            1 for s in gen_spans if _has_attr(s, "response_content")
        ),
        "gens_with_sampling_params": sum(
            1 for s in gen_spans if _has_attr(s, "sampling_params")
        ),
        "gens_with_tools_advertised": sum(
            1 for s in gen_spans if _has_attr(s, "tools")
        ),
    }


# ── Verification ───────────────────────────────────────────────────────────


def verify_run(
    *,
    result: Any,
    trace_file: Optional[Path],
    langsmith_enabled: bool,
) -> Dict[str, Any]:
    """Build a structured pass/fail report covering Phase 1 + Phase 2."""
    checks: Dict[str, Dict[str, Any]] = {}

    checks["orchestration_succeeded"] = {
        "ok": bool(result and result.success),
        "detail": (
            f"success={getattr(result, 'success', None)} "
            f"steps={getattr(result, 'total_steps', None)} "
            f"error={getattr(result, 'error', None)}"
        ),
    }

    final = str(getattr(result, "final_response", "") or "")
    checks["final_response_contains_secret_word"] = {
        "ok": "MARS" in final.upper(),
        "detail": f"final_response={final[:200]!r}",
    }

    checks["ndjson_file_written"] = {
        "ok": trace_file is not None and trace_file.exists(),
        "detail": str(trace_file) if trace_file else "no .ndjson in output dir",
    }

    if trace_file is None or not trace_file.exists():
        return {"checks": checks, "capture_stats": None}

    reader = NDJSONTraceReader(trace_file)
    tree = reader.to_tree()
    root = tree.to_dict()["root_span"]
    stats = collect_capture_stats(root)

    checks["completion_marker_present"] = {
        "ok": reader.completion_status == "complete",
        "detail": f"status={reader.completion_status}",
    }
    checks["root_is_execution_span"] = {
        "ok": root.get("kind") == "execution",
        "detail": f"kind={root.get('kind')}",
    }
    checks["parallel_branches_present"] = {
        "ok": stats["branch_count"] >= 3,  # AgentA, AgentB, AgentC fan-out
        "detail": f"branch_count={stats['branch_count']}",
    }
    checks["tool_call_captured"] = {
        "ok": stats["tool_count"] >= 1,  # AgentB's reveal_secret_letter
        "detail": f"tool_count={stats['tool_count']}",
    }

    # Phase-1 capture invariants — these are the new full-payload fields.
    checks["generation_spans_present"] = {
        "ok": stats["generation_count"] >= 5,  # 5 agents, each gens at least once
        "detail": f"generation_count={stats['generation_count']}",
    }
    checks["all_gens_have_input_messages"] = {
        "ok": stats["generation_count"] > 0
        and stats["gens_with_input_messages"] == stats["generation_count"],
        "detail": (
            f"with={stats['gens_with_input_messages']}/"
            f"{stats['generation_count']} generation spans"
        ),
    }
    checks["all_gens_have_response_content"] = {
        "ok": stats["generation_count"] > 0
        and stats["gens_with_response_content"] == stats["generation_count"],
        "detail": (
            f"with={stats['gens_with_response_content']}/"
            f"{stats['generation_count']}"
        ),
    }
    checks["all_gens_have_sampling_params"] = {
        "ok": stats["generation_count"] > 0
        and stats["gens_with_sampling_params"] == stats["generation_count"],
        "detail": (
            f"with={stats['gens_with_sampling_params']}/"
            f"{stats['generation_count']}"
        ),
    }

    # Phase-2 OTel: only assertable here as "no exception logged"; real
    # vendor verification (visible trace in LangSmith UI) is manual.
    if langsmith_enabled:
        checks["langsmith_export_attempted"] = {
            "ok": True,  # presence of OtelTraceWriter is verified via wiring
            "detail": "OtelTraceWriter wired; check LangSmith UI manually",
        }

    return {"checks": checks, "capture_stats": stats}


# ── Main entry ─────────────────────────────────────────────────────────────


def _load_env_early() -> None:
    """Load .env before argparse computes its defaults.

    Argparse defaults call ``os.getenv``, so the .env load has to
    happen first. ``override=True`` keeps stale shell-injected vars
    from shadowing the .env file (the authoritative source for live
    tests).
    """
    here = Path(__file__).resolve()
    for candidate in (here.parents[2], here.parents[3], here.parents[4]):
        env_file = candidate / ".env"
        if env_file.exists():
            load_dotenv(env_file, override=True)
            print(f"Loaded env: {env_file}")
            return


def parse_args() -> argparse.Namespace:
    _load_env_early()
    p = argparse.ArgumentParser(
        description="Secret-word pipeline live-run smoke check (Phase 1 capture + Phase 2 OTel).",
    )
    p.add_argument("--output-dir", default=None,
                   help="Where to write run.log, .ndjson, messages/.")
    p.add_argument("--task", default=DEFAULT_TASK,
                   help="User task sent to the Orchestrator.")
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--provider", choices=SUPPORTED_PROVIDERS, default="openrouter",
                   help="LLM provider. Default 'openrouter'. With 'azure', reads "
                        "AZURE_OPENAI_API_KEY / AZURE_OPENAI_ENDPOINT / "
                        "AZURE_OPENAI_DEPLOYMENT from env (endpoint must be the "
                        "v1 inference path: .../openai/v1/).")
    p.add_argument("--model-name", default=None,
                   help="For openrouter: the OpenRouter model id (defaults to "
                        "OPEN_ROUTER_MODEL env, then hardcoded fallback). For azure: "
                        "the deployment name (defaults to AZURE_OPENAI_DEPLOYMENT env). "
                        "Resolved per --provider; passing this explicitly overrides "
                        "both env defaults.")
    p.add_argument("--langsmith", action="store_true",
                   help="Enable OTel export to LangSmith. Requires LANGSMITH_API_KEY in env.")
    p.add_argument("--print-tree", action="store_true",
                   help="Pretty-print the reconstructed TraceTree.")
    return p.parse_args()


def _resolve_output_dir(arg: Optional[str]) -> Path:
    if arg:
        out = Path(arg).expanduser().resolve()
    else:
        from datetime import datetime
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = (Path.cwd() / "_runs" / "secret_word" / stamp).resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


async def run(args: argparse.Namespace) -> int:
    # .env already loaded by parse_args → _load_env_early.
    output_dir = _resolve_output_dir(args.output_dir)
    log_file = output_dir / "run.log"

    init_agent_logging(level=logging.ERROR, log_file=str(log_file), file_level=logging.DEBUG)
    print(f"Output dir:      {output_dir}")
    print(f"Log file:        {log_file}")
    print(f"Task:            {args.task!r}")
    print(f"Provider:        {args.provider}")
    print(f"LangSmith:       {args.langsmith}")

    model_config = _build_model_config(args)
    print(f"Model:           {model_config.name}")
    if model_config.base_url and args.provider == "azure":
        print(f"Endpoint:        {model_config.base_url}")

    sinks: List[Any] = []
    otel_sink = None
    if args.langsmith:
        otel_sink = _build_otel_sink()
        if otel_sink is None:
            print("WARNING: --langsmith passed but LANGSMITH_API_KEY not set; skipping OTel export.")
        else:
            sinks.append(otel_sink)
            print(
                f"OTel -> LangSmith enabled (project={os.getenv('LANGSMITH_PROJECT', 'default')!r})"
            )

    _ = _build_agents(model_config)
    # Agents register themselves on AgentRegistry via __init__.

    result = None
    trace_file: Optional[Path] = None
    try:
        result = await Orchestra.run(
            task=args.task,
            topology=_TOPOLOGY,
            agent_registry=AgentRegistry,
            execution_config=ExecutionConfig(
                status=StatusConfig.from_verbosity(2),
                step_timeout=60.0,
                tracing=TracingConfig(
                    enabled=True,
                    output_dir=str(output_dir),
                    capture_full_input=True,        # Phase-1 full payload
                    include_message_content=True,
                    sinks=sinks,                    # Phase-2 OTel (opt-in)
                ),
            ),
            max_steps=args.max_steps,
        )

        print()
        print("=" * 60)
        print(f"Success: {result.success}")
        print(f"Total steps: {result.total_steps}")
        print(f"Duration: {result.total_duration:.1f}s")
        if result.final_response is not None:
            print(f"Final response: {str(result.final_response)[:200]}")

        ndjson_files = sorted(output_dir.glob("*.ndjson"), key=lambda f: f.stat().st_mtime)
        if ndjson_files:
            trace_file = ndjson_files[-1]
            print(f"Trace file: {trace_file.name}")

        if args.print_tree and trace_file is not None:
            tree = TraceTree.from_ndjson(trace_file)
            print()
            print("=" * 60)
            print_tree(tree.to_dict()["root_span"])

    finally:
        AgentRegistry.clear()

    report = verify_run(
        result=result,
        trace_file=trace_file,
        langsmith_enabled=otel_sink is not None,
    )
    print()
    print("=" * 60)
    print("VERIFICATION:")
    for name, check in report["checks"].items():
        mark = "PASS" if check["ok"] else "FAIL"
        print(f"  [{mark}] {name}: {check['detail']}")
    if report.get("capture_stats"):
        print(f"\nCapture stats: {report['capture_stats']}")

    summary = {
        "output_dir": str(output_dir),
        "trace_file": str(trace_file) if trace_file else None,
        "log_file": str(log_file),
        "langsmith_enabled": otel_sink is not None,
        "all_checks_passed": all(c["ok"] for c in report["checks"].values()),
        "checks": {k: v["ok"] for k, v in report["checks"].items()},
        "capture_stats": report.get("capture_stats"),
    }
    print()
    print("=" * 60)
    print("SUMMARY (machine-readable JSON, single line):")
    print(json.dumps(summary, default=str))

    return 0 if summary["all_checks_passed"] else 1


def main() -> None:
    args = parse_args()
    exit_code = asyncio.run(run(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()