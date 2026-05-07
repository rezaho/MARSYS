"""
Streaming NDJSON trace reader.

Two distinct use cases are supported:

* **Live consumer** — call ``stream(follow=True)`` to tail-follow a file as
  spans arrive. Designed for SSE backends, CLI inspectors, hosted UIs.
* **Post-mortem reconstruction** — call ``to_tree()`` on a complete or
  crashed file to materialize a ``TraceTree``. Use ``TraceTree.from_ndjson``
  for the symmetric public surface.

The reader tolerates a truncated trailing line, surfaces missing
``stream_completed`` markers as ``completion_status == "crashed"``, and
ignores unknown top-level fields per the additive-only schema-evolution
policy. ``schema_version`` higher than ``SUPPORTED_SCHEMA_VERSION`` raises
``NDJSONVersionError``.
"""

import json
import logging
import pathlib
import time
from typing import Iterator, Literal, Optional

logger = logging.getLogger(__name__)


class NDJSONVersionError(ValueError):
    """Reader encountered a ``schema_version`` higher than ``SUPPORTED_SCHEMA_VERSION``."""


class NDJSONTraceReader:
    """Streaming NDJSON trace reader with tail-follow support.

    Not Spren-specific. Any consumer that wants line-by-line access to a
    streaming trace uses this reader.
    """

    SUPPORTED_SCHEMA_VERSION = 1
    DEFAULT_POLL_INTERVAL = 0.1

    def __init__(self, path: pathlib.Path):
        self.path = pathlib.Path(path)
        self._completion_status: Literal["complete", "truncated", "crashed"] = "crashed"
        self._truncated_line_count = 0
        self._streamed = False  # whether stream() has been called once

    @property
    def completion_status(self) -> Literal["complete", "truncated", "crashed"]:
        """File completion state. Meaningful after ``stream()`` returns.

        * ``complete`` — file ended with a ``stream_completed`` marker.
        * ``truncated`` — file ended with a partial line (writer crashed
          mid-write or the file was copied during a write).
        * ``crashed`` — file ended cleanly but no marker was written
          (process died after the last full line).
        """
        return self._completion_status

    @property
    def truncated_line_count(self) -> int:
        """Count of unparseable / truncated lines encountered."""
        return self._truncated_line_count

    def stream(
        self,
        follow: bool = False,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
    ) -> Iterator[dict]:
        """Yield span dicts in file order.

        Marker / diagnostic lines (``kind == "stream_completed"``,
        ``kind == "stream_event"``) are not yielded but DO update
        ``completion_status``. With ``follow=True`` the reader polls for
        new bytes after EOF and stops only when it sees the
        ``stream_completed`` marker.
        """
        self._streamed = True
        # Reset state in case stream() is called twice.
        self._completion_status = "crashed"
        self._truncated_line_count = 0

        with self.path.open("r", encoding="utf-8") as f:
            while True:
                position_before = f.tell()
                line = f.readline()

                if not line:
                    # EOF. In follow mode, sleep and re-poll. In one-shot
                    # mode, exit (status is already set).
                    if follow:
                        time.sleep(poll_interval)
                        # Reopen via seek/tell pattern so any newly-flushed
                        # bytes become visible. f.readline() after EOF on
                        # a regular file will start returning data again
                        # if the file grew (POSIX semantics).
                        continue
                    return

                if not line.endswith("\n"):
                    # Partial line — the writer was mid-flush or process
                    # died. Don't yield, don't raise. Bookmark and exit
                    # in one-shot mode; in follow mode, rewind and retry
                    # so we re-read once the line completes.
                    self._truncated_line_count += 1
                    self._completion_status = "truncated"
                    if follow:
                        f.seek(position_before)
                        time.sleep(poll_interval)
                        continue
                    return

                stripped = line.rstrip("\n")
                if not stripped:
                    continue

                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError as e:
                    logger.warning(
                        "NDJSONTraceReader: malformed line skipped (%s): %s",
                        e, stripped[:120],
                    )
                    self._truncated_line_count += 1
                    continue

                version = payload.get("schema_version", 1)
                if version > self.SUPPORTED_SCHEMA_VERSION:
                    raise NDJSONVersionError(
                        f"schema_version={version} exceeds supported={self.SUPPORTED_SCHEMA_VERSION}"
                    )

                kind = payload.get("kind")
                if kind == "stream_completed":
                    self._completion_status = "complete"
                    return
                if kind == "stream_event":
                    # Diagnostic — don't yield to consumers, but log so
                    # operators can see overflow / disable events.
                    attrs = payload.get("attributes", {})
                    logger.info("NDJSONTraceReader stream_event: %s", attrs)
                    continue

                yield payload

    def to_tree(self) -> "TraceTree":
        """Materialize a hierarchical ``TraceTree`` from the file's spans.

        Two-pass: load all spans into a flat dict by ``span_id``, then attach
        each span to its parent. The execution-kind span with
        ``parent_span_id is None`` becomes the root. Spans whose parent is
        unknown go to ``TraceTree.orphans``.
        """
        # Local imports to avoid circular dependencies.
        from ..types import Span, TraceTree

        spans_by_id: dict = {}
        order: list = []  # preserve file order so children list is stable

        for payload in self.stream(follow=False):
            try:
                span = Span(
                    span_id=payload["span_id"],
                    parent_span_id=payload.get("parent_span_id"),
                    trace_id=payload["trace_id"],
                    name=payload["name"],
                    kind=payload["kind"],
                    start_time=payload["start_time"],
                    end_time=payload.get("end_time"),
                    duration_ms=payload.get("duration_ms"),
                    status=payload.get("status", "ok"),
                    attributes=payload.get("attributes") or {},
                    events=payload.get("events") or [],
                    links=payload.get("links") or [],
                )
            except KeyError as e:
                logger.warning("NDJSONTraceReader: skipping span missing %s", e)
                continue
            spans_by_id[span.span_id] = span
            order.append(span.span_id)

        root: Optional["Span"] = None
        orphans: list = []

        for span_id in order:
            span = spans_by_id[span_id]
            if span.parent_span_id is None:
                if span.kind == "execution" and root is None:
                    root = span
                else:
                    orphans.append(span)
                continue
            parent = spans_by_id.get(span.parent_span_id)
            if parent is None:
                orphans.append(span)
            else:
                parent.children.append(span)

        if root is None:
            # No execution span found. Synthesize a stub so callers can
            # walk something — but treat all non-root spans as orphans.
            trace_id = spans_by_id[order[0]].trace_id if order else ""
            root = Span(
                span_id="",
                parent_span_id=None,
                trace_id=trace_id,
                name="(no root span — file may be incomplete)",
                kind="execution",
                start_time=0.0,
            )
            orphans = list(spans_by_id.values())

        return TraceTree(
            trace_id=root.trace_id,
            session_id="",  # session_id is not on the wire format; left empty
            root_span=root,
            metadata={},
            orphans=orphans,
        )
