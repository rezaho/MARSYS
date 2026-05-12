"""AG-UI event stream translator.

Subscribes to ``EventBus`` and emits AG-UI protocol events as an async iterator
so any UI that speaks AG-UI can render a live MARSYS run.

Public exports:

* :class:`AGGUITranslator` — the EventBus subscriber. Constructed by ``Orchestra``
  when ``ExecutionConfig.aggui.enabled`` is ``True``; exposed as
  ``orchestra.aggui_translator``.
* :class:`AGUIEventStream` — the async iterator. ``AGUIEventStream(translator)``.
* :func:`aggui_event_to_sse` — thin wrapper around ``ag_ui.encoder.EventEncoder``.
* :class:`AGGUIConfig` — configuration dataclass (``enabled``, ``queue_max_size``).
* :class:`MarsysRunState` — the typed state model carried by ``StateSnapshot`` /
  ``StateDelta`` events.

Optional-dependency isolation: only ``AGGUIConfig`` is imported eagerly (its
module has no heavy deps). The other public symbols are lazy via PEP 562 ``__getattr__`` —
they hit ``ag-ui-protocol`` / ``jsonpatch`` only when actually accessed.
Framework imports of ``coordination.config`` (which imports ``AGGUIConfig``) stay
clean for users who haven't installed ``pip install 'marsys[aggui]'``.
"""

from typing import TYPE_CHECKING

from .config import AGGUIConfig

if TYPE_CHECKING:
    # Re-exports for type-checkers; runtime resolution goes through __getattr__
    from .sse import aggui_event_to_sse  # noqa: F401
    from .state import MarsysRunState  # noqa: F401
    from .translator import AGGUITranslator, AGUIEventStream  # noqa: F401


__all__ = [
    "AGGUIConfig",
    "AGGUITranslator",
    "AGUIEventStream",
    "MarsysRunState",
    "aggui_event_to_sse",
]


def __getattr__(name: str):
    """PEP 562 lazy access for symbols that pull the optional deps."""
    if name == "AGGUITranslator":
        from .translator import AGGUITranslator
        return AGGUITranslator
    if name == "AGUIEventStream":
        from .translator import AGUIEventStream
        return AGUIEventStream
    if name == "aggui_event_to_sse":
        from .sse import aggui_event_to_sse
        return aggui_event_to_sse
    if name == "MarsysRunState":
        from .state import MarsysRunState
        return MarsysRunState
    raise AttributeError(f"module 'marsys.coordination.aggui' has no attribute {name!r}")
