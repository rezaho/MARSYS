"""Deep-research multi-agent workflow — tracing stress test.

PURPOSE
  Adapted from ``examples/example_01_Deep_Research.py`` (User node
  removed for reproducibility) with tracing + Azure + OTel export wired
  in. Produces orders of magnitude more events than
  ``secret_word_pipeline.py`` and exercises tracing dimensions that
  smoke test doesn't: AgentPool fan-out, compaction spans (large
  fetched pages trip the compaction threshold), sustained event-bus
  pressure, real failure modes (bot-blocked fetches, timeouts), and
  cross-pool MessageStore dedup of overlapping system prompts.

TOPOLOGY
  Start -> OrchestratorAgent -+-> RetrievalAgent -+-> WebSearchAgent
                              |                   +-> BrowserAgent (pool)
                              +-> SynthesizerAgent
                              +-> End

RUN
  cd packages/framework && source ../../.venv/bin/activate
  python live_tests/tracing/deep_research.py --provider azure --langsmith \\
      --output-dir _runs/deep-research-001

KEY ARGS
  --task TOPIC           Research topic.
  --output-dir DIR       Run output (NDJSON, run.log, scratch_pad.jsonl,
                         report.md). Default ./_runs/deep_research/<ts>/.
  --max-steps N          Orchestra max_steps. Default 100.
  --browser-pool-size N  BrowserAgent instances. Default 5.
  --search-tools CSV     Default: "arxiv,semantic_scholar" (both keyless).
                         "google" needs GOOGLE_SEARCH_API_KEY +
                         GOOGLE_CSE_ID_GENERIC; "duckduckgo" is keyless
                         but rate-limited.
  --provider {openrouter,azure}, --model-name NAME, --langsmith,
  --print-tree           See secret_word_pipeline.py for shared semantics.

REQUIREMENTS
  * Provider env vars (see secret_word_pipeline.py).
  * BrowserAgent needs Playwright: ``pip install marsys[browser]`` and
    ``playwright install chromium``.
  * LANGSMITH_API_KEY when --langsmith is set.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from marsys.agents import Agent, BrowserAgent, WebSearchAgent
from marsys.agents.agent_pool import AgentPool
from marsys.agents.registry import AgentRegistry
from marsys.agents.utils import init_agent_logging
from marsys.coordination import Orchestra
from marsys.coordination.config import ExecutionConfig, StatusConfig
from marsys.coordination.tracing import (
    NDJSONTraceReader,
    TraceTree,
    TracingConfig,
)
from marsys.environment.file_operations import create_file_operation_tools
from marsys.models.models import ModelConfig


DEFAULT_TASK = (
    "Find and summarize recent academic papers on retrieval-augmented "
    "generation (RAG) for long-context LLMs from 2024-2025."
)
DEFAULT_MODEL = "anthropic/claude-haiku-4.5"
SUPPORTED_PROVIDERS = ("openrouter", "azure")


# ── Model-config builder (mirrors secret_word_pipeline.py) ─────────────────


def _build_model_config(args: argparse.Namespace) -> ModelConfig:
    if args.provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPEN_ROUTER_API_KEY")
        if not api_key:
            print("ERROR: --provider=openrouter requires OPENROUTER_API_KEY.")
            raise SystemExit(1)
        model_name = args.model_name or os.getenv("OPEN_ROUTER_MODEL") or DEFAULT_MODEL
        return ModelConfig(
            type="api",
            provider="openrouter",
            name=model_name,
            api_key=api_key,
            temperature=0.2,
            max_tokens=12000,
        )

    if args.provider == "azure":
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
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
        return ModelConfig(
            type="api",
            provider="openai",
            name=deployment,
            base_url=endpoint,
            api_key=api_key,
            temperature=0.2,
            max_tokens=12000,
        )

    print(f"ERROR: unsupported --provider={args.provider!r}")
    raise SystemExit(1)


def _build_otel_sink() -> Any:
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        return None
    project = os.getenv("LANGSMITH_PROJECT", "default")
    endpoint = os.getenv(
        "LANGSMITH_OTEL_ENDPOINT",
        "https://api.smith.langchain.com/otel/v1/traces",
    )
    from marsys.coordination.tracing import OtelTraceWriter
    return OtelTraceWriter(
        endpoint=endpoint,
        headers={"x-api-key": api_key, "Langsmith-Project": project},
        service_name=project,
    )


# ── Agent / topology setup (adapted from example_01_Deep_Research.py) ──────


async def _build_agents_and_topology(
    *,
    model_config: ModelConfig,
    output_dir: Path,
    browser_pool_size: int,
    search_tools: List[str],
) -> Dict[str, Any]:
    scratch_pad_file = str(output_dir / "scratch_pad.jsonl")
    report_file = str(output_dir / "research_report.md")

    file_tools = create_file_operation_tools()

    orchestrator = Agent(
        model_config=model_config,
        name="OrchestratorAgent",
        goal="Manage research workflow and coordinate agents",
        instruction=(
            "You manage a research workflow.\n\n"
            "WORKFLOW:\n"
            "1. Delegate to RetrievalAgent with the user's research query.\n"
            f"   Scratch pad file: {scratch_pad_file}\n"
            "2. After RetrievalAgent returns the collected sources, "
            "delegate to SynthesizerAgent to create the report.\n"
            f"   Report file: {report_file}\n"
            "3. Once the report is saved, deliver a brief confirmation as the final answer."
        ),
        memory_retention="single_run",
    )
    retrieval_agent = Agent(
        model_config=model_config,
        name="RetrievalAgent",
        goal="Find and collect research sources from the web",
        instruction=(
            "Find relevant sources using WebSearchAgent.\n"
            "IMPORTANT: Only retrieve content from the TOP 5 most relevant URLs.\n"
            "For each URL, invoke BrowserAgent in parallel with:\n"
            "1. The URL to fetch\n"
            "2. The search query (so it can extract only relevant content)\n"
            f"3. The scratch_pad_file path: {scratch_pad_file}\n"
            "Return to OrchestratorAgent when all URLs have been processed."
        ),
        memory_retention="single_run",
    )
    web_search_agent = WebSearchAgent(
        model_config=model_config,
        name="WebSearchAgent",
        goal="Find relevant sources using web and academic search",
        instruction="Search for information using available search tools. Return relevant URLs and summaries.",
        enabled_tools=search_tools,
        memory_retention="single_run",
    )

    browser_pool = await AgentPool.create_async(
        agent_class=BrowserAgent,
        num_instances=browser_pool_size,
        model_config=model_config,
        name="BrowserAgent",
        mode="primitive",
        headless=True,
        memory_retention="single_run",
        tools={"write_file": file_tools["write_file"], "read_file": file_tools["read_file"]},
        instruction=(
            "You are a browser agent that fetches web content and saves it to a scratch pad file.\n\n"
            "WORKFLOW:\n"
            "1. Extract the content of the given URL.\n"
            "2. Clean and filter the content to keep only information relevant to the search query.\n"
            f"3. Append the result to {scratch_pad_file} using mode='append'.\n"
            "Write a single JSON line: "
            '{"url": "...", "title": "...", "content": "...", "timestamp": "..."}\n'
            "Return to RetrievalAgent after saving."
        ),
    )
    AgentRegistry.register_pool(browser_pool)

    synthesizer = Agent(
        model_config=model_config,
        name="SynthesizerAgent",
        goal="Create research reports from collected sources",
        tools={"read_file": file_tools["read_file"], "write_file": file_tools["write_file"]},
        instruction=(
            f"Read all content from {scratch_pad_file}. "
            "Synthesize a markdown report with executive summary, main findings, "
            f"detailed analysis with citations, and a references section. Save to {report_file}. "
            "Return to OrchestratorAgent."
        ),
        memory_retention="single_run",
    )

    topology = {
        "agents": [
            "Start", "OrchestratorAgent", "RetrievalAgent", "WebSearchAgent",
            "BrowserAgent", "SynthesizerAgent", "End",
        ],
        "flows": [
            "Start -> OrchestratorAgent",
            "OrchestratorAgent -> RetrievalAgent",
            "OrchestratorAgent -> SynthesizerAgent",
            "RetrievalAgent -> OrchestratorAgent",
            "RetrievalAgent -> WebSearchAgent",
            "RetrievalAgent -> BrowserAgent",
            "WebSearchAgent -> RetrievalAgent",
            "BrowserAgent -> RetrievalAgent",
            "SynthesizerAgent -> OrchestratorAgent",
            "OrchestratorAgent -> End",
        ],
        "rules": ["timeout(600)"],
    }
    return {
        "topology": topology,
        "scratch_pad_file": scratch_pad_file,
        "report_file": report_file,
    }


# ── Inspection / verification (adapted from secret_word_pipeline.py) ───────


def print_tree(span_dict: Dict[str, Any], indent: int = 0) -> None:
    prefix = "  " * indent
    kind = span_dict.get("kind", "?")
    name = span_dict.get("name", "?")
    status = span_dict.get("status", "?")
    attrs = span_dict.get("attributes", {})
    children = span_dict.get("children", [])
    extra = ""
    if kind == "branch":
        extra = f" src={attrs.get('source_agent')} tgt={attrs.get('target_agents')}"
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
    gen_spans: List[Dict[str, Any]] = []
    compaction_spans: List[Dict[str, Any]] = []
    branch_spans: List[Dict[str, Any]] = []
    tool_spans: List[Dict[str, Any]] = []
    error_spans: List[Dict[str, Any]] = []
    pool_branch_agents: Dict[str, int] = {}

    def walk(s: Dict[str, Any]) -> None:
        kind = s.get("kind")
        if kind == "generation":
            gen_spans.append(s)
        elif kind == "compaction":
            compaction_spans.append(s)
        elif kind == "branch":
            branch_spans.append(s)
            tgt = (s.get("attributes") or {}).get("target_agents")
            if isinstance(tgt, list):
                for t in tgt:
                    pool_branch_agents[t] = pool_branch_agents.get(t, 0) + 1
        elif kind == "tool":
            tool_spans.append(s)
        if s.get("status") == "error":
            error_spans.append(s)
        for c in s.get("children", []):
            walk(c)

    walk(root)

    def _has_attr(s: Dict[str, Any], key: str) -> bool:
        v = (s.get("attributes") or {}).get(key)
        return v is not None and v != [] and v != {}

    # A "produced output" span has either text content or tool calls (or both).
    # A pure tool-call response correctly carries content=None — that's the
    # OpenAI shape a chat-UI renders as the AI-message bubble. Counting such
    # turns as "missing payload" is a verifier bug, not a capture bug.
    def _produced_output(s: Dict[str, Any]) -> bool:
        return _has_attr(s, "response_content") or _has_attr(s, "response_tool_calls")

    return {
        "branch_count": len(branch_spans),
        "tool_count": len(tool_spans),
        "generation_count": len(gen_spans),
        "compaction_count": len(compaction_spans),
        "error_span_count": len(error_spans),
        "branches_per_agent": pool_branch_agents,
        "gens_with_input_messages": sum(1 for s in gen_spans if _has_attr(s, "input_messages")),
        "gens_with_response_content": sum(1 for s in gen_spans if _has_attr(s, "response_content")),
        "gens_with_response_tool_calls": sum(1 for s in gen_spans if _has_attr(s, "response_tool_calls")),
        "gens_with_output": sum(1 for s in gen_spans if _produced_output(s)),
        "gens_with_sampling_params": sum(1 for s in gen_spans if _has_attr(s, "sampling_params")),
        "gens_with_tools_advertised": sum(1 for s in gen_spans if _has_attr(s, "tools")),
    }


def verify_run(
    *,
    result: Any,
    trace_file: Optional[Path],
    otel_enabled: bool,
    browser_pool_size: int,
) -> Dict[str, Any]:
    checks: Dict[str, Dict[str, Any]] = {}

    checks["orchestration_succeeded"] = {
        "ok": bool(result and result.success),
        "detail": (
            f"success={getattr(result, 'success', None)} "
            f"steps={getattr(result, 'total_steps', None)} "
            f"error={getattr(result, 'error', None)}"
        ),
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
    # Pool fan-out: at least one BrowserAgent dispatch is required for the
    # research workflow to actually do anything.
    browser_dispatches = stats["branches_per_agent"].get("BrowserAgent", 0)
    checks["browser_pool_fanout"] = {
        "ok": browser_dispatches >= 1,
        "detail": f"BrowserAgent branches={browser_dispatches} (pool_size={browser_pool_size})",
    }
    # Tool spans: file_writes + browser fetches + web search.
    checks["tool_spans_present"] = {
        "ok": stats["tool_count"] >= 3,
        "detail": f"tool_count={stats['tool_count']}",
    }
    # Generation spans: at minimum one per agent that ran.
    checks["generation_spans_present"] = {
        "ok": stats["generation_count"] >= 5,
        "detail": f"generation_count={stats['generation_count']}",
    }
    # Phase-1 invariants: full payload on every generation.
    checks["all_gens_have_input_messages"] = {
        "ok": stats["generation_count"] > 0
        and stats["gens_with_input_messages"] == stats["generation_count"],
        "detail": f"with={stats['gens_with_input_messages']}/{stats['generation_count']}",
    }
    # Every generation must produce *some* output — either text content or
    # tool calls. A pure tool-call response legitimately has content=None.
    checks["all_gens_have_output"] = {
        "ok": stats["generation_count"] > 0
        and stats["gens_with_output"] == stats["generation_count"],
        "detail": (
            f"with={stats['gens_with_output']}/{stats['generation_count']} "
            f"(text={stats['gens_with_response_content']}, "
            f"tool_calls={stats['gens_with_response_tool_calls']})"
        ),
    }
    checks["all_gens_have_sampling_params"] = {
        "ok": stats["generation_count"] > 0
        and stats["gens_with_sampling_params"] == stats["generation_count"],
        "detail": f"with={stats['gens_with_sampling_params']}/{stats['generation_count']}",
    }

    if otel_enabled:
        checks["otel_export_attempted"] = {
            "ok": True,
            "detail": "OtelTraceWriter wired; check the backend UI manually",
        }

    return {"checks": checks, "capture_stats": stats}


# ── Main entry ─────────────────────────────────────────────────────────────


def _load_env_early() -> None:
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
        description="Deep-research multi-agent tracing stress test.",
    )
    p.add_argument("--task", default=DEFAULT_TASK, help="Research topic.")
    p.add_argument("--output-dir", default=None,
                   help="Where to write run.log, .ndjson, messages/, scratch_pad, report.")
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--browser-pool-size", type=int, default=5,
                   help="Number of BrowserAgent instances in the pool.")
    p.add_argument("--search-tools", default="arxiv,semantic_scholar",
                   help="Comma-separated search tools enabled in WebSearchAgent. "
                        "Default 'arxiv,semantic_scholar' is keyless. Other options: "
                        "'google' (needs GOOGLE_SEARCH_API_KEY + GOOGLE_CSE_ID_GENERIC), "
                        "'duckduckgo' (keyless, rate-limited).")
    p.add_argument("--provider", choices=SUPPORTED_PROVIDERS, default="openrouter")
    p.add_argument("--model-name", default=None,
                   help="OpenRouter model id or Azure deployment name.")
    p.add_argument("--langsmith", action="store_true")
    p.add_argument("--print-tree", action="store_true")
    return p.parse_args()


def _resolve_output_dir(arg: Optional[str]) -> Path:
    if arg:
        out = Path(arg).expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = (Path.cwd() / "_runs" / "deep_research" / stamp).resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


async def run(args: argparse.Namespace) -> int:
    output_dir = _resolve_output_dir(args.output_dir)
    log_file = output_dir / "run.log"

    init_agent_logging(level=logging.ERROR, log_file=str(log_file), file_level=logging.DEBUG)
    print(f"Output dir:        {output_dir}")
    print(f"Log file:          {log_file}")
    print(f"Task:              {args.task!r}")
    print(f"Provider:          {args.provider}")
    print(f"Browser pool size: {args.browser_pool_size}")
    print(f"Search tools:      {args.search_tools}")
    print(f"LangSmith:         {args.langsmith}")

    model_config = _build_model_config(args)
    print(f"Model:             {model_config.name}")
    if model_config.base_url and args.provider == "azure":
        print(f"Endpoint:          {model_config.base_url}")

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

    search_tools = [t.strip() for t in args.search_tools.split(",") if t.strip()]

    setup = await _build_agents_and_topology(
        model_config=model_config,
        output_dir=output_dir,
        browser_pool_size=args.browser_pool_size,
        search_tools=search_tools,
    )

    result = None
    trace_file: Optional[Path] = None
    try:
        result = await Orchestra.run(
            task=args.task,
            topology=setup["topology"],
            agent_registry=AgentRegistry,
            execution_config=ExecutionConfig(
                status=StatusConfig.from_verbosity(2),
                user_interaction="none",       # non-interactive: we removed the User node
                step_timeout=120.0,
                tracing=TracingConfig(
                    enabled=True,
                    output_dir=str(output_dir),
                    include_message_content=True,
                    sinks=sinks,
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
            print(f"Final response: {str(result.final_response)[:300]}")

        ndjson_files = sorted(output_dir.glob("*.ndjson"), key=lambda f: f.stat().st_mtime)
        if ndjson_files:
            trace_file = ndjson_files[-1]
            size_kb = trace_file.stat().st_size / 1024
            print(f"Trace file: {trace_file.name} ({size_kb:.1f} KB)")

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
        otel_enabled=otel_sink is not None,
        browser_pool_size=args.browser_pool_size,
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
        "otel_export_enabled": otel_sink is not None,
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