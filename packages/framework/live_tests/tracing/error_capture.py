"""Structured ErrorEvent (Phase 1) — live-run smoke check.

PURPOSE
  End-to-end verification that real API errors surface in the NDJSON trace
  as structured ``ErrorEvent`` data — not just a bare-string ``error``
  attribute. Triggers a real 4xx by configuring an agent with an invalid
  model name; expects ``ModelAPIError`` to propagate, the step span to
  carry ``error_class``, ``error_classification``, ``recoverable``,
  ``retry_count``, and a non-empty ``traceback`` event payload.

  This is the failure-path complement to ``parallel.py`` and
  ``full_input_capture.py``, both of which exercise the success path.

EXERCISES
  * coordination.status.events.ErrorEvent — structured exception payload
  * coordination.execution.step_executor.StepExecutor._emit_error_event —
    helper that fires ErrorEvent on caught exceptions
  * coordination.tracing.collector.TraceCollector._handle_error — adds
    structured event to span.events + mirror attributes for filterability
  * coordination.tracing.collector.TraceCollector._handle_agent_complete —
    prefers structured event over legacy bare-string error attribute

TOPOLOGY
  Researcher (single agent, single branch — keep the trace minimal so
  the failure-path verification is unambiguous).

RUN
  cd packages/framework && source ../../.venv/bin/activate
  python live_tests/tracing/error_capture.py --output-dir /tmp/marsys_runs/run-003

KEY ARGS
  --bad-model NAME       Invalid model name to provoke a 4xx. Default is
                         'claude-this-model-does-not-exist'.
  --output-dir DIR       Where run.log and <trace_id>.ndjson land.
  --max-steps N          Orchestra max_steps. Default 3.
  --oauth-profile NAME   Anthropic OAuth profile. Default 'marsys-2'.

OUTPUTS
  <output-dir>/run.log               DEBUG framework log
  <output-dir>/<trace_id>.ndjson     Streaming trace file
  Last stdout line                   Single-line JSON summary
  Exit code                          0 = all checks passed, 1 otherwise

  Summary keys: output_dir, trace_file, log_file, all_checks_passed,
  checks (per-check bool), error_event_count, error_classifications.

REQUIREMENTS
  * A working `anthropic-oauth` profile (default 'marsys-2') so the
    request reaches Anthropic's API and gets rejected on model name —
    not on auth.
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
    TracingConfig,
)
from marsys.models.models import ModelConfig


DEFAULT_TASK = "Hello — this should fail before the model responds."
DEFAULT_BAD_MODEL = "claude-this-model-does-not-exist"


def _build_agents(model_config: ModelConfig) -> Dict[str, Agent]:
    researcher = Agent(
        model_config=model_config,
        name="Researcher",
        goal="Smoke-test failure-path tracing.",
        instruction=(
            "You are a researcher. Reply with a single sentence. "
            "(In this run the model name is invalid, so we don't expect a reply.)"
        ),
        memory_retention="single_run",
    )
    return {"Researcher": researcher}


_TOPOLOGY = {
    "agents": ["Researcher"],
    "flows": [],
    "entry_point": "Researcher",
    "exit_points": ["Researcher"],
}


# ── Trace inspection ──────────────────────────────────────────────────────


def collect_step_spans(root: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    def walk(s: Dict[str, Any]) -> None:
        if s.get("kind") == "step":
            out.append(s)
        for c in s.get("children", []):
            walk(c)
    walk(root)
    return out


def collect_error_events(spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for s in spans:
        for ev in s.get("events", []) or []:
            if ev.get("name") == "error":
                out.append({"span": s, "event": ev})
    return out


# ── Verification ──────────────────────────────────────────────────────────


def verify_run(
    *,
    result: Any,
    trace_file: Optional[Path],
) -> Dict[str, Any]:
    checks: Dict[str, Dict[str, Any]] = {}

    # Failure path: orchestration is expected to fail (or the agent step is)
    # — we don't gate on `result.success`. We do require *some* result, even
    # if it's a failed one, so the orchestrator's own teardown ran.
    checks["result_returned"] = {
        "ok": result is not None,
        "detail": (
            f"success={getattr(result, 'success', None)} "
            f"error={str(getattr(result, 'error', '') or '')[:200]}"
        ),
    }
    checks["ndjson_file_written"] = {
        "ok": trace_file is not None and trace_file.exists(),
        "detail": str(trace_file) if trace_file else "no .ndjson file in output dir",
    }

    if trace_file is None or not trace_file.exists():
        return {"checks": checks}

    reader = NDJSONTraceReader(trace_file)
    tree = reader.to_tree()
    root = tree.to_dict()["root_span"]

    step_spans = collect_step_spans(root)
    error_records = collect_error_events(step_spans)

    checks["error_event_present"] = {
        "ok": len(error_records) > 0,
        "detail": f"error_event_count={len(error_records)}",
    }

    # Take the first ErrorEvent for shape validation.
    first = error_records[0]["event"] if error_records else None
    first_attrs = (first or {}).get("attributes", {})

    checks["error_class_populated"] = {
        "ok": bool(first_attrs.get("error_class")),
        "detail": f"error_class={first_attrs.get('error_class')!r}",
    }
    checks["error_classification_set"] = {
        # Bad model name → APIErrorClassification.INVALID_MODEL or similar.
        # We don't pin the exact value (provider classifiers vary) — just
        # require *some* classification when the underlying error was a
        # ModelAPIError.
        "ok": (
            first_attrs.get("error_class") != "ModelAPIError"
            or bool(first_attrs.get("classification"))
        ),
        "detail": (
            f"error_class={first_attrs.get('error_class')} "
            f"classification={first_attrs.get('classification')}"
        ),
    }
    checks["recoverable_field_typed"] = {
        "ok": isinstance(first_attrs.get("recoverable"), bool),
        "detail": f"recoverable={first_attrs.get('recoverable')!r}",
    }
    checks["traceback_present"] = {
        "ok": bool(first_attrs.get("traceback")),
        "detail": (
            f"traceback length={len(first_attrs.get('traceback') or '')} "
            f"chars (truncated to 4096 max by step_executor)"
        ),
    }

    # Span-level mirror attributes (fast filter surface).
    spans_with_error_class = [
        s for s in step_spans if s.get("attributes", {}).get("error_class")
    ]
    checks["span_attribute_mirror"] = {
        "ok": len(spans_with_error_class) > 0,
        "detail": (
            f"step_spans_with_error_class={len(spans_with_error_class)} / "
            f"total={len(step_spans)}"
        ),
    }

    # Span ``status`` should be 'error' for the failed step.
    failed_step_status = [
        s.get("status") for s in step_spans
        if s.get("attributes", {}).get("error_class")
    ]
    checks["span_status_error"] = {
        "ok": all(st == "error" for st in failed_step_status) and len(failed_step_status) > 0,
        "detail": f"statuses={failed_step_status}",
    }

    classifications = sorted({
        rec["event"].get("attributes", {}).get("classification")
        for rec in error_records
        if rec["event"].get("attributes", {}).get("classification")
    })

    return {
        "checks": checks,
        "error_event_count": len(error_records),
        "error_classifications": classifications,
        "first_error_class": first_attrs.get("error_class"),
    }


# ── Main entry ────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 1 ErrorEvent live-run smoke check (deliberate failure).",
    )
    p.add_argument("--output-dir", default=None)
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--bad-model", default=DEFAULT_BAD_MODEL,
                   help="Invalid model name used to provoke a 4xx.")
    p.add_argument("--max-steps", type=int, default=3)
    p.add_argument("--oauth-profile", default="marsys-2")
    return p.parse_args()


def _resolve_output_dir(arg: Optional[str]) -> Path:
    if arg:
        out = Path(arg).expanduser().resolve()
    else:
        from datetime import datetime
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = (Path.cwd() / "_runs" / "error_capture" / stamp).resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


async def run(args: argparse.Namespace) -> int:
    output_dir = _resolve_output_dir(args.output_dir)
    log_file = output_dir / "run.log"

    init_agent_logging(level=logging.ERROR, log_file=str(log_file), file_level=logging.DEBUG)
    print(f"Output dir:      {output_dir}")
    print(f"Log file:        {log_file}")
    print(f"Bad model:       {args.bad_model!r}")

    # Tight retries so the failure surfaces fast (no waiting for 5s exponential
    # backoff). The bad-model case returns a 4xx (non-retryable) so the loop
    # would short-circuit anyway, but be explicit for the cheap-CI use case.
    from marsys.coordination.config import ErrorHandlingConfig

    model_config = ModelConfig(
        type="api",
        provider="anthropic-oauth",
        name=args.bad_model,  # invalid → triggers 4xx
        oauth_profile=args.oauth_profile,
        temperature=0.3,
        max_tokens=200,
    )

    _ = _build_agents(model_config)

    result = None
    trace_file: Optional[Path] = None
    try:
        try:
            result = await Orchestra.run(
                task=args.task,
                topology=_TOPOLOGY,
                agent_registry=AgentRegistry,
                execution_config=ExecutionConfig(
                    status=StatusConfig.from_verbosity(1),
                    step_timeout=30.0,
                    error_handling=ErrorHandlingConfig(
                        max_retries=1, base_delay=0.1, jitter=0.0, max_delay=1.0,
                    ),
                    tracing=TracingConfig(
                        enabled=True,
                        output_dir=str(output_dir),
                    ),
                ),
                max_steps=args.max_steps,
            )
        except Exception as e:  # noqa: BLE001
            # The bad-model case may propagate the exception out through
            # Orchestra.run; capture it as a synthetic failed result so
            # verification can still inspect the trace.
            print()
            print(f"Orchestra.run raised (expected): {type(e).__name__}: {e}")
            class _FakeResult:
                success = False
                error = f"{type(e).__name__}: {e}"
                total_steps = 0
                total_duration = 0.0
                final_response = None
                metadata = {}
            result = _FakeResult()

        print()
        print("=" * 60)
        print(f"Success:        {getattr(result, 'success', None)}")
        print(f"Total steps:    {getattr(result, 'total_steps', None)}")
        print(f"Error:          {str(getattr(result, 'error', '') or '')[:200]}")

        ndjson_files = sorted(output_dir.glob("*.ndjson"), key=lambda f: f.stat().st_mtime)
        if ndjson_files:
            trace_file = ndjson_files[-1]
            print(f"Trace file:     {trace_file.name}")

    finally:
        AgentRegistry.clear()

    report = verify_run(result=result, trace_file=trace_file)

    print()
    print("=" * 60)
    print("VERIFICATION:")
    for name, check in report["checks"].items():
        mark = "PASS" if check["ok"] else "FAIL"
        print(f"  [{mark}] {name}: {check['detail']}")

    summary = {
        "output_dir": str(output_dir),
        "trace_file": str(trace_file) if trace_file else None,
        "log_file": str(log_file),
        "all_checks_passed": all(c["ok"] for c in report["checks"].values()),
        "checks": {k: v["ok"] for k, v in report["checks"].items()},
        "error_event_count": report.get("error_event_count"),
        "error_classifications": report.get("error_classifications"),
        "first_error_class": report.get("first_error_class"),
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
