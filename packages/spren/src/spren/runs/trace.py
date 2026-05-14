"""Run-trace adapter.

Thin wrapper over the framework-shipped ``NDJSONTraceReader`` +
``TraceTree.from_ndjson`` (at
``packages/framework/src/marsys/coordination/tracing/{readers/ndjson_reader.py,types.py}``).

Spren intentionally does NOT re-implement the parser. Per Session 05
decision §16, the framework writes one ``{trace_id}.ndjson`` per run
into ``<data-dir>/data/runs/{run_id}/`` (per-run ``output_dir`` set by
``materialize.py``), so glob-resolution finds exactly one file per run.

The 50MB Spren-side response cap (decision §10.12) lives here — purely a
JSON-response guardrail; storage is uncapped.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from spren.models import RunTrace, RunTraceCompletionStatus, SpanNode

logger = logging.getLogger(__name__)


# §10.12 response-size cap. Storage is uncapped — this guards the
# bound-JSON response only. If trace.ndjson exceeds the cap, the
# adapter still parses it, but truncates the response and surfaces a
# truncation banner on the client.
RESPONSE_SIZE_CAP_BYTES = 50 * 1024 * 1024  # 50 MB


class TraceNotAvailableError(Exception):
    """No trace.ndjson file exists for this run yet."""


def find_trace_file(*, data_dir: Path, run_id: str) -> Path | None:
    """Resolve the per-run NDJSON trace file, or ``None`` if not present.

    Per-run ``output_dir`` is ``<data-dir>/data/runs/{run_id}``; the
    framework writes a single ``{trace_id}.ndjson`` there. Glob → single
    match in the steady state. If no file is present, the run hasn't
    emitted a trace yet (or tracing is disabled).
    """
    run_dir = data_dir / "data" / "runs" / run_id
    if not run_dir.is_dir():
        return None
    matches = sorted(run_dir.glob("*.ndjson"))
    if not matches:
        return None
    if len(matches) > 1:
        # Multiple traces shouldn't happen with per-run output_dir; pick
        # the most recent and log so an operator notices.
        logger.warning(
            "find_trace_file: %d ndjson files in %s; using most recent",
            len(matches),
            run_dir,
        )
        matches.sort(key=lambda p: p.stat().st_mtime)
    return matches[-1]


def build_run_trace(*, data_dir: Path, run_id: str) -> RunTrace:
    """Read + parse the run's trace; return a ``RunTrace`` Pydantic.

    Raises ``TraceNotAvailableError`` when no NDJSON file is on disk.

    The framework's reader handles all parsing edge cases (truncated
    final line, missing terminal marker = crashed, malformed lines).
    Spren's adapter:

    1. Glob-resolves the file.
    2. Calls ``NDJSONTraceReader(path).to_tree()`` — gets ``TraceTree``.
    3. Inspects ``reader.completion_status`` + ``reader.truncated_line_count``.
    4. Walks the framework ``Span`` tree → Spren ``SpanNode`` Pydantic.
    5. Counts total spans (root + descendants + orphans).
    6. Enforces the 50MB response cap.
    """
    # Local import — keep the framework dep contained at the call site.
    from marsys.coordination.tracing.readers.ndjson_reader import NDJSONTraceReader

    path = find_trace_file(data_dir=data_dir, run_id=run_id)
    if path is None:
        raise TraceNotAvailableError(
            f"no trace file present for run {run_id}; tracing may be disabled or run hasn't started"
        )

    reader = NDJSONTraceReader(path)
    tree = reader.to_tree()

    completion = RunTraceCompletionStatus(reader.completion_status)
    truncated_line_count = reader.truncated_line_count

    # Convert framework Span → Spren SpanNode. Walk the root + each orphan.
    root_node = _to_span_node(tree.root_span)
    spans: List[SpanNode] = [root_node]
    spans.extend(_to_span_node(o) for o in tree.orphans)

    total_spans = _count_spans(spans)

    # Apply the response-size cap. We compute approximate JSON-size by
    # serializing one node at a time; below the cap we just return everything.
    truncated, truncation_reason = _apply_response_cap(spans)

    return RunTrace(
        run_id=run_id,
        completion_status=completion,
        truncated_line_count=truncated_line_count,
        total_spans=total_spans,
        spans=spans,
        truncated=truncated,
        truncation_reason=truncation_reason,
    )


def _to_span_node(span: object) -> SpanNode:
    """Convert one framework ``Span`` (recursive) to ``SpanNode``."""
    return SpanNode(
        span_id=getattr(span, "span_id"),
        parent_span_id=getattr(span, "parent_span_id"),
        trace_id=getattr(span, "trace_id"),
        name=getattr(span, "name"),
        kind=getattr(span, "kind"),
        start_time=getattr(span, "start_time"),
        end_time=getattr(span, "end_time"),
        duration_ms=getattr(span, "duration_ms"),
        status=getattr(span, "status", "ok"),
        attributes=dict(getattr(span, "attributes") or {}),
        events=list(getattr(span, "events") or []),
        links=list(getattr(span, "links") or []),
        children=[_to_span_node(c) for c in (getattr(span, "children") or [])],
    )


def _count_spans(nodes: List[SpanNode]) -> int:
    """Total span count = sum across the forest."""
    total = 0
    stack: list[SpanNode] = list(nodes)
    while stack:
        node = stack.pop()
        total += 1
        stack.extend(node.children)
    return total


def _apply_response_cap(spans: List[SpanNode]) -> tuple[bool, Optional[str]]:
    """Best-effort response-size guard.

    Walks the encoded JSON length of each top-level subtree; if the
    cumulative size crosses the cap, the offending subtree (and any
    later siblings) are dropped from the wire output. Returns
    ``(truncated, reason)``.

    For typical v0.3 sizes (≤10MB), this loop runs once per subtree
    and returns ``(False, None)`` immediately.
    """
    cumulative = 0
    cutoff_index: int | None = None
    for idx, node in enumerate(spans):
        encoded_len = len(node.model_dump_json())
        cumulative += encoded_len
        if cumulative > RESPONSE_SIZE_CAP_BYTES:
            cutoff_index = idx
            break
    if cutoff_index is None:
        return False, None
    # Drop the offending subtree + remaining siblings.
    del spans[cutoff_index:]
    return True, "trace_size_cap"
