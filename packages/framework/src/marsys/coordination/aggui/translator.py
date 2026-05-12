"""AGGUITranslator — EventBus subscriber that produces AG-UI events.

Peer subscriber to ``TraceCollector``. Subscribes to every event class in
``mapping.DISPATCH`` at construction; unsubscribes in :py:meth:`close`. Maps each
event to zero or more AG-UI ``BaseEvent`` instances and enqueues them on a
bounded ``asyncio.Queue``.

Backpressure: drop-newest. When the queue is full, the new event is dropped and
``_lagged_count`` is incremented. On the next successful enqueue, a
``Custom("marsys.stream.lagged")`` event prefixes the next event so the consumer
sees how many events were lost. Drop-oldest is wrong here — it would break
AG-UI's ``TextMessageStart`` / ``Content`` / ``End`` ordering invariant by
dropping the start while keeping a later content event.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Optional, Tuple

from ag_ui.core import BaseEvent

from . import mapping as _mapping
from .config import AGGUIConfig
from .custom_events import validate_custom_value
from .state import MarsysRunState

if TYPE_CHECKING:
    from ag_ui.core import CustomEvent

    from ..event_bus import EventBus


logger = logging.getLogger(__name__)


def _make_lagged_custom(count: int) -> "CustomEvent":
    from ag_ui.core import CustomEvent  # local import to keep optional
    validate_custom_value("marsys.stream.lagged", {"count": count})
    return CustomEvent(name="marsys.stream.lagged", value={"count": count})


class AGGUITranslator:
    """EventBus subscriber that produces AG-UI events into a bounded queue.

    Lifecycle:
      * ``__init__`` — subscribes to every event class in ``DISPATCH``.
      * ``close()`` — unsubscribes and marks the stream as closed.

    The corresponding :class:`AGUIEventStream` iterator consumes the queue.
    """

    def __init__(
        self,
        event_bus: "EventBus",
        config: AGGUIConfig,
    ) -> None:
        self.event_bus = event_bus
        self.config = config
        self.queue: asyncio.Queue[BaseEvent] = asyncio.Queue(
            maxsize=config.queue_max_size
        )
        self.state = MarsysRunState()
        self._lagged_count = 0
        self._closed = False
        # Set by map_assistant_message; consumed by map_tool_call for
        # parent_message_id linkage.
        self.last_assistant_message_id: Optional[str] = None
        # Per-(branch_id, tool_name, step_number) → tool_call_id. Maintains
        # id continuity across the started/completed/failed lifecycle.
        self.tool_call_ids: Dict[Tuple[Optional[str], str, Optional[int]], str] = {}
        self._subscribe()

    def mark_terminal(self) -> None:
        """Called by terminal mappers (RunFinished / RunError) to flag close.

        The iterator drains remaining queued events first; once the queue is
        empty AND ``_closed`` is set, ``__anext__`` raises ``StopAsyncIteration``.
        """
        self._closed = True

    def _subscribe(self) -> None:
        for event_cls in _mapping.DISPATCH:
            self.event_bus.subscribe(event_cls.__name__, self._handle)

    async def _handle(self, event: Any) -> None:
        try:
            for aggui_event in _mapping.map_event(event, self):
                self._enqueue(aggui_event)
        except Exception:
            # AG-UI translation must never break the run; mapper failures stay
            # local. Other subscribers (TraceCollector) keep running on the same
            # event because EventBus.emit catches per-listener exceptions.
            logger.exception(
                "AGGUI mapper for %s failed", type(event).__name__,
            )

    def _enqueue(self, aggui_event: BaseEvent) -> None:
        # Catch-up notification: before this event, surface the cumulative
        # drop count from prior overflow(s) and reset.
        if self._lagged_count > 0:
            try:
                self.queue.put_nowait(_make_lagged_custom(self._lagged_count))
                self._lagged_count = 0
            except asyncio.QueueFull:
                # Still full — keep counting; try again next time.
                pass
        try:
            self.queue.put_nowait(aggui_event)
        except asyncio.QueueFull:
            self._lagged_count += 1

    async def close(self) -> None:
        """Unsubscribe and mark closed. Idempotent."""
        was_closed = self._closed
        self._closed = True
        if was_closed:
            # EventBus.unsubscribe is idempotent, but skipping the loop on
            # repeat-close avoids unnecessary list mutations.
            return
        for event_cls in _mapping.DISPATCH:
            self.event_bus.unsubscribe(event_cls.__name__, self._handle)


class AGUIEventStream:
    """Async iterator that yields AG-UI events from a translator's queue.

    Takes the translator directly — no Orchestra/run_id indirection. Orchestra
    exposes the translator as a plain attribute (``orchestra.aggui_translator``).
    """

    def __init__(self, translator: AGGUITranslator) -> None:
        self.translator = translator

    def __aiter__(self) -> "AGUIEventStream":
        return self

    async def __anext__(self) -> BaseEvent:
        # Drain remaining events even after terminal, then stop.
        while True:
            if not self.translator.queue.empty():
                return self.translator.queue.get_nowait()
            if self.translator._closed:
                raise StopAsyncIteration
            # Wait for the next event or for the translator to close.
            try:
                return await asyncio.wait_for(
                    self.translator.queue.get(), timeout=0.1
                )
            except asyncio.TimeoutError:
                # Loop and recheck closed/queue state.
                continue
