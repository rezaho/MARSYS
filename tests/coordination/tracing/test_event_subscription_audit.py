"""Audit test: every StatusEvent subclass is either subscribed or explicitly ignored.

The collector dispatches events by ``type(event).__name__``. A renamed or
added event class silently stops being traced with no compile-time signal;
this test is the gate that fails CI instead of letting the trace degrade.

For each concrete StatusEvent subclass the test asserts the class name is
either:
  - bound to a handler in the collector's subscription map, OR
  - listed in ``TraceCollector.IGNORED_EVENTS`` with an intentional reason.

When a new event is added, the contributor must choose one of the two —
adding a handler or explicitly declaring "not for tracing."
"""
from __future__ import annotations

import importlib
import pkgutil

import pytest

from marsys.coordination.event_bus import EventBus
from marsys.coordination.status.events import StatusEvent
from marsys.coordination.tracing.collector import TraceCollector
from marsys.coordination.tracing.config import TracingConfig


def _import_all_event_modules() -> None:
    """Walk the coordination package so every ``StatusEvent`` subclass is
    registered on ``StatusEvent.__subclasses__``. Without this walk,
    ``__subclasses__`` returns only the classes imported by code that has
    already run — making the audit incomplete on a cold test process.
    """
    import marsys.coordination as pkg
    for module_info in pkgutil.walk_packages(pkg.__path__, prefix=f"{pkg.__name__}."):
        try:
            importlib.import_module(module_info.name)
        except Exception:
            # Some submodules are optional (e.g. extras-gated); skipping
            # them is fine — they can't define events we don't already
            # see through the modules that do import.
            pass


def _concrete_status_event_names() -> set[str]:
    """Return the names of every concrete (non-abstract) ``StatusEvent`` subclass.

    Recursive: also walks subclasses-of-subclasses (e.g. tracing events that
    extend ``StatusEvent`` directly land at depth 1; future intermediate
    bases would land deeper).
    """
    _import_all_event_modules()
    seen: set[type] = set()
    stack: list[type] = list(StatusEvent.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        stack.extend(cls.__subclasses__())
    # The base ``StatusEvent`` itself is abstract by convention (never
    # emitted directly); concrete-only here means "leaf classes the bus
    # would actually dispatch on by class name". We treat every non-base
    # subclass as concrete — there are no intermediate abstract bases
    # under ``StatusEvent`` today.
    return {cls.__name__ for cls in seen}


@pytest.fixture
def collector(tmp_path):
    cfg = TracingConfig(enabled=True, output_dir=str(tmp_path))
    return TraceCollector(EventBus(), cfg)


def test_every_status_event_is_subscribed_or_explicitly_ignored(collector):
    """The audit. Fails when a new ``StatusEvent`` subclass appears that
    isn't in either set — forcing the contributor to make a choice.
    """
    all_events = _concrete_status_event_names()
    subscribed = set(collector.event_bus.listeners.keys())
    ignored = TraceCollector.IGNORED_EVENTS

    unhandled = all_events - subscribed - ignored
    assert not unhandled, (
        "These StatusEvent subclasses are neither subscribed by "
        "TraceCollector nor listed in TraceCollector.IGNORED_EVENTS — "
        "the trace will silently drop them. Add a handler or extend "
        f"IGNORED_EVENTS with a one-line reason: {sorted(unhandled)}"
    )


def test_ignored_events_do_not_collide_with_subscribed(collector):
    """``IGNORED_EVENTS`` is a *deliberate skip*, not a fallback. An event
    cannot be both subscribed and ignored — that would mean someone added
    a handler but forgot to remove the ignore entry, and the next reader
    has to guess which is the source of truth.
    """
    subscribed = set(collector.event_bus.listeners.keys())
    collision = subscribed & TraceCollector.IGNORED_EVENTS
    assert not collision, (
        f"These event names appear in both TraceCollector's subscription "
        f"map and IGNORED_EVENTS — remove from one: {sorted(collision)}"
    )


def test_ignored_event_names_resolve_to_real_classes():
    """A typo'd or stale entry in ``IGNORED_EVENTS`` would silently let a
    really-untracked event slip through — the audit test above would
    still pass because the typo matches no live class.
    """
    all_events = _concrete_status_event_names()
    stale = TraceCollector.IGNORED_EVENTS - all_events
    assert not stale, (
        "These IGNORED_EVENTS entries don't match any live StatusEvent "
        f"subclass — typo or removed class: {sorted(stale)}"
    )
