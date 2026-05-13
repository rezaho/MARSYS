"""The load-bearing regression test: every framework Event class has a disposition.

Walks every event-defining module — across multiple base-class lineages, not
just ``StatusEvent`` subclasses — and asserts each ``*Event`` class is in
``mapping.EVENT_REGISTRY``. Adding a new event class without registering it
fails this test.
"""

from __future__ import annotations

import importlib
import inspect

import pytest

pytest.importorskip("ag_ui")

from marsys.coordination.aggui.mapping import (
    DISPATCH,
    EVENT_REGISTRY,
    INTERNAL_ONLY,
    NOT_YET_EMITTED,
)


# Modules containing framework Event classes the translator must account for.
# The translator subscribes to EventBus, which routes by class name — so any
# class that ends in "Event" and is dispatched via emit() must have a known
# disposition here. New event modules added in future PRs must be added to
# this list.
EVENT_MODULES = (
    "marsys.coordination.status.events",
    "marsys.coordination.tracing.events",
    "marsys.coordination.events",
    "marsys.agents.memory",  # MemoryResetEvent lives here, doesn't inherit StatusEvent
)


def _discover_event_classes() -> set:
    """Return every class named ``*Event`` defined in any EVENT_MODULES module."""
    discovered: set = set()
    for module_name in EVENT_MODULES:
        module = importlib.import_module(module_name)
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Only classes DEFINED in this module (not imported into it)
            if obj.__module__ != module_name:
                continue
            if not name.endswith("Event"):
                continue
            # Skip the base class itself
            if name in {"StatusEvent", "BaseEvent"}:
                continue
            discovered.add(obj)
    return discovered


def test_buckets_are_mutually_exclusive():
    assert DISPATCH.keys().isdisjoint(INTERNAL_ONLY), (
        "Event in both DISPATCH and INTERNAL_ONLY"
    )
    assert DISPATCH.keys().isdisjoint(NOT_YET_EMITTED), (
        "Event in both DISPATCH and NOT_YET_EMITTED"
    )
    assert INTERNAL_ONLY.isdisjoint(NOT_YET_EMITTED), (
        "Event in both INTERNAL_ONLY and NOT_YET_EMITTED"
    )


def test_registry_equals_union_of_buckets():
    assert EVENT_REGISTRY == set(DISPATCH.keys()) | INTERNAL_ONLY | NOT_YET_EMITTED


def test_every_discovered_event_has_a_disposition():
    discovered = _discover_event_classes()
    missing = discovered - EVENT_REGISTRY
    extra = EVENT_REGISTRY - discovered
    assert not missing, (
        f"Event classes without a mapping disposition: "
        f"{sorted(c.__name__ for c in missing)}. "
        f"Add to DISPATCH, INTERNAL_ONLY, or NOT_YET_EMITTED in "
        f"coordination/aggui/mapping.py."
    )
    # extra: events in the registry that no longer exist in source.
    # Permitted only if the framework deliberately removed an event class —
    # require explicit cleanup of the mapping module.
    assert not extra, (
        f"EVENT_REGISTRY references classes not found in source modules: "
        f"{sorted(c.__name__ for c in extra)}. "
        f"Either the class moved (update EVENT_MODULES) or was removed "
        f"(clean up coordination/aggui/mapping.py)."
    )


def test_discovery_finds_memory_reset_event():
    """MemoryResetEvent doesn't inherit from StatusEvent — verify discovery
    handles cross-base-class lineages."""
    discovered = _discover_event_classes()
    names = {c.__name__ for c in discovered}
    assert "MemoryResetEvent" in names
    assert "FinalResponseEvent" in names  # StatusEvent subclass
    assert "ExecutionStartEvent" in names  # tracing/events
    assert "BranchCreatedEvent" in names  # coordination/events
