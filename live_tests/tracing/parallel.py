"""Parallel multi-agent NDJSON tracing — live-run smoke check.

PURPOSE
  End-to-end verification that streaming NDJSON tracing works correctly
  against a real Anthropic Haiku-4.5 model under a parallel-fan-out
  topology. The topology is shaped to produce a non-trivial trace
  (branch spans, convergence links, multiple step + tool spans).

EXERCISES
  * coordination.tracing.collector.TraceCollector — span tree, fan-out
  * coordination.tracing.writers.ndjson_writer.NDJSONTraceWriter — streaming
  * coordination.tracing.sink.TelemetrySink — the new ABC
  * coordination.tracing.redactor.SecretRedactor — chokepoint redaction
    (only with --inject-secret; otherwise the task is benign and the
    redactor finds nothing to scrub)
  * coordination.orchestra.Orchestra.run — TracingConfig.sinks plumbing,
    finalize-block close timeout, _collect_tracing_metadata

TOPOLOGY
  Coordinator --+--> Researcher  --+--> Coordinator
                +--> FactChecker --+

RUN
  cd packages/framework && source ../../.venv/bin/activate
  python live_tests/tracing/parallel.py --output-dir /tmp/marsys_runs/run-001

KEY ARGS
  --task TASK            User task sent to the Coordinator. Default: a
                         speed-of-light question. Keep stable across runs
                         when you want comparable traces; vary it when
                         you want to probe a different LLM behaviour.
  --inject-secret        Add a noop `record_credential` tool to the
                         Coordinator and prompt it to call the tool with
                         a fake api_key BEFORE dispatching workers. The
                         verification step then asserts the redactor
                         scrubbed it from the on-disk trace and the
                         literal injected token does NOT appear in the
                         file. Without this flag the run is benign and
                         no redactions are expected.
  --output-dir DIR       Where run.log and <trace_id>.ndjson land.
                         REQUIRED for automation; auto-generated under
                         ./_runs/parallel_tracing/<timestamp>/ otherwise.
  --max-steps N          Orchestra max_steps. Default 20.
  --print-tree           Pretty-print the reconstructed TraceTree.
  --oauth-profile NAME   Anthropic OAuth profile. Default 'marsys-2'.

OUTPUTS
  <output-dir>/run.log                DEBUG-level framework log
  <output-dir>/<trace_id>.ndjson      Streaming trace file
  Last stdout line                     Single-line JSON summary
  Exit code                            0 = all checks passed, 1 otherwise

  Summary keys: output_dir, trace_file, log_file, all_checks_passed,
  checks (per-check bool), tree_stats, redaction_count, inject_secret.

REQUIREMENTS
  * A working `anthropic-oauth` profile (default 'marsys-2'); see
    packages/framework/research/anthropic-oauth.md if you don't have one.
  * Network access to api.anthropic.com.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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


DEFAULT_TASK = "What is the speed of light and who first measured it?"
INJECTED_SECRET = "sk-FAKE-INJECTED-XYZ123ABC"  # never a real key; used only with --inject-secret


# ── Tools ──────────────────────────────────────────────────────────────────
# Defined at module scope so the framework's tool-schema generator (which
# inspects __module__ + __qualname__) can serialize it cleanly.


def record_credential(api_key: str, note: str = "") -> str:
    """Record a credential token. Use exactly once when explicitly told to.

    The framework's TraceCollector records this call's arguments dict in the
    tool span's attributes. The SecretRedactor running at the fan-out
    chokepoint should scrub the api_key field before any sink sees it.

    Args:
        api_key: The API key string. Stored as-is on the server.
        note: Optional human-readable description of what this credential is for.

    Returns:
        Acknowledgement string with the credential's character length.
    """
    return f"acknowledged: stored credential ({len(api_key)} chars, note={note!r})"


# ── Topology builders ──────────────────────────────────────────────────────


def _build_agents(model_config: ModelConfig, *, inject_secret: bool) -> Dict[str, Agent]:
    if inject_secret:
        coord_instruction = (
            "You are a coordinator. Your turn has THREE steps, in this exact order:\n"
            "  1. FIRST RESPONSE: Call `record_credential` exactly once with "
            f"     api_key='{INJECTED_SECRET}' and note='setup token'. Wait for "
            "     the acknowledgement.\n"
            "  2. SECOND RESPONSE (after the acknowledgement comes back): "
            "     dispatch to BOTH Researcher and FactChecker in parallel by "
            "     calling `invoke_agent` with two invocations (one per worker).\n"
            "  3. THIRD RESPONSE (after both workers return): synthesize a "
            "     final answer and call `terminate_workflow` with the answer.\n"
            "Do not skip steps. Do not call `record_credential` more than once."
        )
        coord_tools: Optional[Dict[str, Any]] = {"record_credential": record_credential}
    else:
        coord_instruction = (
            "You are a coordinator. On your FIRST response, dispatch to BOTH "
            "Researcher and FactChecker in parallel by calling `invoke_agent` "
            "with two invocations (one per worker). When you receive their "
            "results, synthesize a final answer and call `terminate_workflow` "
            "with the synthesized answer."
        )
        coord_tools = None

    coordinator = Agent(
        model_config=model_config,
        name="Coordinator",
        goal="Coordinate research by dispatching to workers in parallel.",
        instruction=coord_instruction,
        memory_retention="session",
        tools=coord_tools,
    )
    researcher = Agent(
        model_config=model_config,
        name="Researcher",
        goal="Research a topic and provide findings.",
        instruction=(
            "You are a researcher. When given a topic, provide a brief 2-3 sentence "
            "research summary. Be concise. When done, return your findings to "
            "Coordinator by calling `invoke_agent` with target='Coordinator' and "
            "your findings as the request."
        ),
        memory_retention="single_run",
    )
    fact_checker = Agent(
        model_config=model_config,
        name="FactChecker",
        goal="Verify claims and check facts.",
        instruction=(
            "You are a fact checker. When given a topic, provide 2-3 key facts "
            "that should be verified. Be concise. When done, return your facts to "
            "Coordinator by calling `invoke_agent` with target='Coordinator' and "
            "your facts as the request."
        ),
        memory_retention="single_run",
    )
    return {"Coordinator": coordinator, "Researcher": researcher, "FactChecker": fact_checker}


_TOPOLOGY = {
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


# ── Inspection helpers ─────────────────────────────────────────────────────


def print_tree(span_dict: Dict[str, Any], indent: int = 0) -> None:
    prefix = "  " * indent
    kind = span_dict.get("kind", "?")
    name = span_dict.get("name", "?")
    status = span_dict.get("status", "?")
    attrs = span_dict.get("attributes", {})
    children = span_dict.get("children", [])
    events = span_dict.get("events", [])
    links = span_dict.get("links", [])

    extra = ""
    if kind == "branch":
        extra = (
            f" src={attrs.get('source_agent')} tgt={attrs.get('target_agents')} "
            f"trigger={attrs.get('trigger_type')}"
        )
    elif kind == "generation":
        extra = (
            f" model={attrs.get('model_name','')} "
            f"in={attrs.get('prompt_tokens')} out={attrs.get('completion_tokens')}"
        )
    elif kind == "tool":
        extra = f" tool={attrs.get('tool_name')}"

    print(
        f"{prefix}[{kind}] {name} (status={status} ch={len(children)} "
        f"ev={len(events)} lnk={len(links)}){extra}"
    )

    for link in links:
        rel = link.get("relationship", "?")
        linked_id = link.get("linked_span_id", "?")[:8]
        link_attrs = link.get("attributes", {})
        print(f"{prefix}  >> LINK: {rel} -> span {linked_id} {link_attrs}")

    for event in events:
        ev_name = event.get("name", "?")
        ev_attrs = event.get("attributes", {})
        print(f"{prefix}  ** EVENT: {ev_name} {ev_attrs}")

    for child in children:
        print_tree(child, indent + 1)


def check_convergence(root_span_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Walk the tree and report convergence-evidence stats."""
    branch_spans: List[Dict[str, Any]] = []
    all_spans: List[Dict[str, Any]] = []

    def walk(span: Dict[str, Any], parent_kind: Optional[str] = None) -> None:
        all_spans.append(span)
        kind = span.get("kind")
        if kind == "branch":
            branch_spans.append(span)
        for child in span.get("children", []):
            walk(child, kind)

    walk(root_span_dict)
    spans_with_links = [s for s in all_spans if s.get("links")]
    return {
        "total_spans": len(all_spans),
        "branch_spans": len(branch_spans),
        "spans_with_links": len(spans_with_links),
        "convergence_captured": bool(spans_with_links),
    }


def find_redacted_keys(span_dict: Dict[str, Any]) -> List[str]:
    """Walk the trace; report every dotted-path whose value is the redaction marker."""
    found: List[str] = []
    REDACTED = "[REDACTED]"

    def walk_attrs(attrs: Dict[str, Any], path: str) -> None:
        for key, value in attrs.items():
            here = f"{path}.{key}" if path else key
            if isinstance(value, dict):
                walk_attrs(value, here)
            elif value == REDACTED:
                found.append(here)

    def walk_span(s: Dict[str, Any], path: str) -> None:
        kind = s.get("kind", "?")
        name = s.get("name", "?")
        span_path = f"{path}/{kind}:{name}"
        attrs = s.get("attributes", {})
        if isinstance(attrs, dict):
            walk_attrs(attrs, f"{span_path}.attributes")
        for ev in s.get("events", []):
            ev_attrs = ev.get("attributes", {})
            if isinstance(ev_attrs, dict):
                walk_attrs(ev_attrs, f"{span_path}.events[{ev.get('name','?')}]")
        for ln in s.get("links", []):
            ln_attrs = ln.get("attributes", {})
            if isinstance(ln_attrs, dict):
                walk_attrs(ln_attrs, f"{span_path}.links")
        for child in s.get("children", []):
            walk_span(child, span_path)

    walk_span(span_dict, "")
    return found


# ── Verification ───────────────────────────────────────────────────────────


def verify_run(
    *,
    result: Any,
    trace_file: Optional[Path],
    inject_secret: bool,
    expected_branch_count_min: int = 2,
) -> Dict[str, Any]:
    """Inspect run + trace; build a structured report. Computes a pass/fail per check."""
    checks: Dict[str, Dict[str, Any]] = {}

    checks["orchestration_succeeded"] = {
        "ok": bool(result and result.success),
        "detail": (
            f"success={getattr(result, 'success', None)} "
            f"steps={getattr(result, 'total_steps', None)} "
            f"error={getattr(result, 'error', None)}"
        ),
    }

    tracing_meta = {}
    if result is not None and isinstance(getattr(result, "metadata", None), dict):
        tracing_meta = result.metadata.get("tracing", {})

    checks["tracing_metadata_populated"] = {
        "ok": bool(tracing_meta) and tracing_meta.get("total_spans", 0) > 0,
        "detail": json.dumps(tracing_meta, default=str),
    }
    checks["sink_not_disabled"] = {
        "ok": tracing_meta.get("disabled", False) is False,
        "detail": (
            f"disabled={tracing_meta.get('disabled')} "
            f"disk_errors={tracing_meta.get('disk_error_count')} "
            f"dropped={tracing_meta.get('dropped_span_count')}"
        ),
    }
    checks["ndjson_file_written"] = {
        "ok": trace_file is not None and trace_file.exists(),
        "detail": str(trace_file) if trace_file else "no .ndjson file in output dir",
    }

    if trace_file is None or not trace_file.exists():
        return {"checks": checks, "tree_stats": None, "redactions": []}

    reader = NDJSONTraceReader(trace_file)
    tree = reader.to_tree()
    tree_dict = tree.to_dict()
    root = tree_dict["root_span"]
    stats = check_convergence(root)
    redactions = find_redacted_keys(root)

    checks["completion_marker_present"] = {
        "ok": reader.completion_status == "complete",
        "detail": f"status={reader.completion_status} truncated_lines={reader.truncated_line_count}",
    }
    checks["root_is_execution_span"] = {
        "ok": root.get("kind") == "execution",
        "detail": f"kind={root.get('kind')}",
    }
    checks["parallel_branches_present"] = {
        "ok": stats["branch_spans"] >= expected_branch_count_min,
        "detail": f"branch_spans={stats['branch_spans']}",
    }
    checks["convergence_links_captured"] = {
        "ok": stats["convergence_captured"],
        "detail": f"spans_with_links={stats['spans_with_links']}",
    }

    # Inject-secret-specific checks
    if inject_secret:
        raw_text = trace_file.read_text(encoding="utf-8")
        secret_in_file = INJECTED_SECRET in raw_text
        checks["injected_secret_redacted"] = {
            "ok": (not secret_in_file) and len(redactions) > 0,
            "detail": (
                f"secret_in_file={secret_in_file} "
                f"redaction_count={len(redactions)} "
                f"sample_paths={redactions[:3]}"
            ),
        }
    else:
        checks["no_unexpected_redactions"] = {
            "ok": len(redactions) == 0,
            "detail": (
                f"redaction_count={len(redactions)} "
                f"(benign task; redaction_count > 0 means the deny-list "
                f"matched a framework attribute it shouldn't have)"
            ),
        }

    return {"checks": checks, "tree_stats": stats, "redactions": redactions}


# ── Main entry ─────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Parallel multi-agent NDJSON tracing live-run smoke check.",
    )
    p.add_argument("--output-dir", default=None, help="Where to write run.log and the .ndjson trace.")
    p.add_argument("--task", default=DEFAULT_TASK, help="User task. Keep stable for comparable traces.")
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--oauth-profile", default="marsys-2")
    p.add_argument("--inject-secret", action="store_true",
                   help="Inject a fake api_key via a tool call; verify the redactor scrubs it.")
    p.add_argument("--print-tree", action="store_true",
                   help="Pretty-print the reconstructed TraceTree to stdout.")
    return p.parse_args()


def _resolve_output_dir(arg: Optional[str]) -> Path:
    if arg:
        out = Path(arg).expanduser().resolve()
    else:
        from datetime import datetime
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = (Path.cwd() / "_runs" / "parallel_tracing" / stamp).resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


async def run(args: argparse.Namespace) -> int:
    output_dir = _resolve_output_dir(args.output_dir)
    log_file = output_dir / "run.log"

    init_agent_logging(level=logging.ERROR, log_file=str(log_file), file_level=logging.DEBUG)
    print(f"Output dir:      {output_dir}")
    print(f"Log file:        {log_file}")
    print(f"Task:            {args.task!r}")
    print(f"Inject secret:   {args.inject_secret}")

    model_config = ModelConfig(
        type="api",
        provider="anthropic-oauth",
        name="claude-haiku-4.5",
        oauth_profile=args.oauth_profile,
        temperature=0.3,
        max_tokens=4000,
    )

    _ = _build_agents(model_config, inject_secret=args.inject_secret)
    # Agents register themselves via their __init__ on AgentRegistry.

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
                    include_message_content=True,
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
        inject_secret=args.inject_secret,
    )
    print()
    print("=" * 60)
    print("VERIFICATION:")
    for name, check in report["checks"].items():
        mark = "PASS" if check["ok"] else "FAIL"
        print(f"  [{mark}] {name}: {check['detail']}")

    if report.get("tree_stats"):
        print(f"\nTree stats: {report['tree_stats']}")
    if report.get("redactions"):
        print(f"\nRedacted attribute paths ({len(report['redactions'])}):")
        for path in report["redactions"][:20]:
            print(f"  {path}")

    summary = {
        "output_dir": str(output_dir),
        "trace_file": str(trace_file) if trace_file else None,
        "log_file": str(log_file),
        "inject_secret": args.inject_secret,
        "all_checks_passed": all(c["ok"] for c in report["checks"].values()),
        "checks": {k: v["ok"] for k, v in report["checks"].items()},
        "tree_stats": report.get("tree_stats"),
        "redaction_count": len(report.get("redactions") or []),
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
