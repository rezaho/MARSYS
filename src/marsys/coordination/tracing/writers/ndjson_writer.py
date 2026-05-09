"""
Streaming NDJSON trace writer.

One JSON object per closed span, written line-by-line as spans close, plus a
``stream_completed`` marker on close. Implements ``TelemetrySink``; hooked into
``TraceCollector`` via ``publish_span``. The collector remains the single
``EventBus`` subscriber and sole tree builder.

Wire format per line::

    {
        "schema_version": 1,
        "ts": <float epoch seconds — emission time>,
        "span_id": "<ULID>",
        "parent_span_id": "<ULID>" | null,
        "trace_id": "<ULID>",
        "name": "...",
        "kind": "execution|branch|step|generation|tool",
        "start_time": <float>,
        "end_time": <float>,
        "duration_ms": <float>,
        "status": "ok|error",
        "attributes": {...},
        "events": [...]   # optional
        "links": [...]    # optional
    }

Diagnostic lines (queue overflow, writer self-disable) carry
``"kind": "stream_event"`` and an ``"attributes"`` payload describing the event.
The terminal marker carries ``"kind": "stream_completed"``.
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, Optional, TYPE_CHECKING

from ..sink import TelemetrySink

if TYPE_CHECKING:
    from ..config import TracingConfig
    from ..types import Span

logger = logging.getLogger(__name__)


class _DrainSentinel:
    """Singleton sentinel signalling the drain task to stop after current items."""


_SENTINEL = _DrainSentinel()


class NDJSONTraceWriter(TelemetrySink):
    """Streaming NDJSON trace sink.

    Each closed span is enqueued from ``TraceCollector`` and serialized to disk
    by a single dedicated drain task. ``publish_span`` is non-blocking on the
    disk path. Drop-oldest on queue overflow.

    On disk errors the writer self-disables after
    ``DISK_ERROR_DISABLE_THRESHOLD`` consecutive failures so a permanent
    failure (read-only mount, full disk) does not flood logs.

    Lifecycle:

    * ``publish_span(span)`` — enqueue (lazy file open on first call; the
      filename is derived from ``span.trace_id``).
    * ``close()`` — drain queue, write ``stream_completed`` marker, fsync,
      close file. Idempotent.
    """

    SCHEMA_VERSION = 1
    DISK_ERROR_DISABLE_THRESHOLD = 100
    DISK_ERROR_LOG_PERIOD = 100
    DEFAULT_QUEUE_MAXSIZE = 10000
    CLOSE_TIMEOUT_SECONDS = 5.0

    def __init__(
        self,
        config: 'TracingConfig',
        *,
        fsync_per_span: bool = False,
        queue_maxsize: int = DEFAULT_QUEUE_MAXSIZE,
    ):
        self.config = config
        self.fsync_per_span = fsync_per_span
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=queue_maxsize)
        self._drain_task: Optional[asyncio.Task] = None
        self._file: Optional[Any] = None
        self._trace_id: Optional[str] = None
        self._closed = False
        self._disabled = False

        self._total_spans = 0
        self._disk_error_count = 0
        self._consecutive_disk_errors = 0
        self._dropped_span_count = 0
        self._disabled_dropped_count = 0

    # ── Introspection ───────────────────────────────────────────────────
    @property
    def total_spans(self) -> int:
        return self._total_spans

    @property
    def disk_error_count(self) -> int:
        return self._disk_error_count

    @property
    def consecutive_disk_errors(self) -> int:
        return self._consecutive_disk_errors

    @property
    def dropped_span_count(self) -> int:
        return self._dropped_span_count

    @property
    def disabled_dropped_count(self) -> int:
        return self._disabled_dropped_count

    @property
    def disabled(self) -> bool:
        return self._disabled

    # ── Lifecycle ───────────────────────────────────────────────────────
    async def publish_span(self, span: 'Span') -> None:
        """Enqueue a closed span for serialization. Non-blocking on disk."""
        if self._closed:
            return

        # Lazy file open on first span; trace_id is the filename key.
        if self._trace_id is None:
            self._open_file(span.trace_id)
            self._drain_task = asyncio.create_task(self._drain_loop())
        elif span.trace_id != self._trace_id:
            raise ValueError(
                f"NDJSONTraceWriter is per-trace; received second trace_id "
                f"{span.trace_id!r} (already bound to {self._trace_id!r})"
            )

        self._enqueue_or_drop_oldest(span)

    async def close(self) -> None:
        """Drain queue, write ``stream_completed`` marker, fsync, close file.

        Idempotent: subsequent calls return immediately.
        """
        if self._closed:
            return
        self._closed = True

        # Signal the drain task to exit after processing remaining items.
        if self._drain_task is not None:
            self._enqueue_or_drop_oldest(_SENTINEL)
            try:
                await asyncio.wait_for(
                    self._drain_task, timeout=self.CLOSE_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "NDJSONTraceWriter drain task did not finish within %.1fs; cancelling",
                    self.CLOSE_TIMEOUT_SECONDS,
                )
                self._drain_task.cancel()
                try:
                    await self._drain_task
                except (asyncio.CancelledError, Exception):
                    pass
            except Exception as e:
                logger.warning("NDJSONTraceWriter drain task raised on close: %s", e)
            self._drain_task = None

        # Best-effort terminal marker. Failures here are logged, not raised.
        if self._file is not None:
            try:
                marker = {
                    "schema_version": self.SCHEMA_VERSION,
                    "ts": time.time(),
                    "kind": "stream_completed",
                    "attributes": {
                        "total_spans": self._total_spans,
                        "disk_error_count": self._disk_error_count,
                        "dropped_span_count": self._dropped_span_count,
                        "disabled_dropped_count": self._disabled_dropped_count,
                        "disabled": self._disabled,
                    },
                }
                self._file.write(json.dumps(marker, default=str) + "\n")
                self._file.flush()
                try:
                    os.fsync(self._file.fileno())
                except OSError:
                    pass
            except OSError as e:
                logger.warning(
                    "NDJSONTraceWriter failed to write stream_completed marker: %s", e
                )
            try:
                self._file.close()
            except OSError:
                pass
            self._file = None

    # ── Internals ───────────────────────────────────────────────────────
    def _open_file(self, trace_id: str) -> None:
        """Create the trace file at ``{output_dir}/{trace_id}.ndjson``."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        path = os.path.join(self.config.output_dir, f"{trace_id}.ndjson")
        # ``newline=''`` disables Python's universal newline translation in
        # text mode; line terminator stays ``\n`` on every platform (no \r\n
        # on Windows). UTF-8 without BOM.
        self._file = open(path, mode="a", encoding="utf-8", newline="")
        self._trace_id = trace_id
        logger.info("NDJSONTraceWriter opened %s", path)

    def _enqueue_or_drop_oldest(self, item: Any) -> None:
        """Put ``item`` on the queue; drop the oldest item if full."""
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            else:
                if not isinstance(item, _DrainSentinel):
                    self._dropped_span_count += 1
                    self._emit_diagnostic(
                        {"event": "dropped_span",
                         "dropped_span_count": self._dropped_span_count}
                    )
            try:
                self._queue.put_nowait(item)
            except asyncio.QueueFull:
                # Should not happen after a get_nowait drain; if it does, drop.
                if not isinstance(item, _DrainSentinel):
                    self._dropped_span_count += 1

    async def _drain_loop(self) -> None:
        """Single dedicated task: pop spans off the queue and write to disk."""
        while True:
            try:
                item = await self._queue.get()
            except asyncio.CancelledError:
                return

            if isinstance(item, _DrainSentinel):
                return

            if self._disabled:
                self._disabled_dropped_count += 1
                continue

            line = self._serialize_span(item)
            self._write_line(line)

    def _serialize_span(self, span: 'Span') -> str:
        """Build the per-line wire payload from a Span. ``children`` is dropped
        because each child is its own line in the NDJSON stream."""
        payload: Dict[str, Any] = {
            "schema_version": self.SCHEMA_VERSION,
            "ts": time.time(),
            "span_id": span.span_id,
            "parent_span_id": span.parent_span_id,
            "trace_id": span.trace_id,
            "name": span.name,
            "kind": span.kind,
            "start_time": span.start_time,
            "end_time": span.end_time,
            "duration_ms": span.duration_ms,
            "status": span.status,
            "attributes": span.attributes,
        }
        if span.events:
            payload["events"] = span.events
        if span.links:
            payload["links"] = span.links
        return json.dumps(payload, default=str)

    def _write_line(self, line: str) -> None:
        """Append a line to the trace file with flush/fsync. On OSError,
        increment the disk-error counter and self-disable past the threshold."""
        if self._file is None:
            return
        try:
            self._file.write(line + "\n")
            self._file.flush()
            if self.fsync_per_span:
                os.fsync(self._file.fileno())
            self._total_spans += 1
            self._consecutive_disk_errors = 0
        except OSError as e:
            self._disk_error_count += 1
            self._consecutive_disk_errors += 1
            if self._disk_error_count == 1:
                logger.warning(
                    "NDJSONTraceWriter disk write failed (first error)",
                    exc_info=e,
                )
            elif self._disk_error_count % self.DISK_ERROR_LOG_PERIOD == 0:
                logger.warning(
                    "NDJSONTraceWriter disk write failed (%d total errors): %s",
                    self._disk_error_count, e,
                )
            if (
                not self._disabled
                and self._consecutive_disk_errors >= self.DISK_ERROR_DISABLE_THRESHOLD
            ):
                self._disabled = True
                logger.warning(
                    "NDJSONTraceWriter self-disabled after %d consecutive disk errors",
                    self._consecutive_disk_errors,
                )
                self._emit_diagnostic(
                    {"event": "writer_disabled",
                     "consecutive_failures": self._consecutive_disk_errors}
                )

    def _emit_diagnostic(self, attributes: Dict[str, Any]) -> None:
        """Best-effort diagnostic line (overflow, self-disable). Errors are swallowed."""
        if self._file is None:
            return
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "ts": time.time(),
            "kind": "stream_event",
            "attributes": attributes,
        }
        try:
            self._file.write(json.dumps(payload, default=str) + "\n")
            self._file.flush()
        except OSError:
            pass
