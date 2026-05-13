"""SSE wire-format helper for AG-UI events.

Thin wrapper around ``ag_ui.encoder.EventEncoder``. Saves consumers (Spren,
MARSYS Cloud, Studio, third-party AG-UI clients) from re-importing the encoder.
"""

from __future__ import annotations

from ag_ui.core import BaseEvent
from ag_ui.encoder import EventEncoder

_encoder = EventEncoder()


def aggui_event_to_sse(event: BaseEvent) -> str:
    """Encode an AG-UI event to its SSE wire form (``data: {json}\\n\\n``)."""
    return _encoder.encode(event)
