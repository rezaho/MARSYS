"""Structural + extension-openness tests for the specialized-agent registry.

Covers ADR-009 Decision 2 (single-sourced ``AGENT_KIND_REGISTRY``) — the
agent-axis analog of ``tests/coordination/topology/test_node_kind_registry.py``
for the node axis:

- AC-13  one authoritative ``kind → Agent subclass`` map is the single source
- AC-14  each specialized subclass declares its own ``WIRE_KIND`` class attr
- AC-15  the reverse ``class → kind`` map is *derived* by inversion, not a
         second hand-maintained dict
- AC-16  a build-failing test catches a ``WIRE_KIND``-bearing class missing
         from the registry, or a registry entry without a matching attr
- AC-17  resolution is registry-gated — NOT reflection / ``Agent.__subclasses__``
         and NOT ``marsys.agents.__all__``
"""

from __future__ import annotations

import marsys.agents as agents_pkg
from marsys.agents import (
    BrowserAgent,
    CodeExecutionAgent,
    DataAnalysisAgent,
    FileOperationAgent,
    WebSearchAgent,
)
from marsys.agents.agents import Agent
from marsys.agents.serialize import (
    AGENT_KIND_REGISTRY,
    BASE_AGENT_KIND,
    _CLASS_TO_KIND,
    _KIND_TO_PARAMS_SPEC,
)

# The five in-scope specialized subclasses (ADR-009 scope; LearnableAgent is
# explicitly OUT — it is not an Agent subclass and must not appear here).
_EXPECTED_SPECIALIZED = {
    WebSearchAgent,
    BrowserAgent,
    CodeExecutionAgent,
    DataAnalysisAgent,
    FileOperationAgent,
}


# --- AC-13: one authoritative registry, the single source -------------------


def test_single_authoritative_agent_kind_registry():
    assert AGENT_KIND_REGISTRY == {
        "web_search": WebSearchAgent,
        "browser": BrowserAgent,
        "code_execution": CodeExecutionAgent,
        "data_analysis": DataAnalysisAgent,
        "file_operation": FileOperationAgent,
    }
    # "agent" is the base sentinel and is NOT a specialized registry entry.
    assert BASE_AGENT_KIND == "agent"
    assert BASE_AGENT_KIND not in AGENT_KIND_REGISTRY
    assert set(AGENT_KIND_REGISTRY.values()) == _EXPECTED_SPECIALIZED


# --- AC-14: each subclass declares its own WIRE_KIND ------------------------


def test_each_specialized_subclass_declares_wire_kind():
    for cls in _EXPECTED_SPECIALIZED:
        assert isinstance(getattr(cls, "WIRE_KIND", None), str), cls.__name__
        assert cls.WIRE_KIND, cls.__name__
    # Base Agent must NOT carry a WIRE_KIND (it is the default sentinel).
    assert not hasattr(Agent, "WIRE_KIND")


# --- AC-15: reverse map derived by inversion, not hand-maintained -----------


def test_reverse_class_to_kind_is_derived_by_inversion():
    assert _CLASS_TO_KIND == {
        cls: kind for kind, cls in AGENT_KIND_REGISTRY.items()
    }
    # Every entry is exactly the inverse of a forward entry (no orphan pair).
    for cls, kind in _CLASS_TO_KIND.items():
        assert AGENT_KIND_REGISTRY[kind] is cls


# --- AC-16: bidirectional WIRE_KIND <-> registry consistency (build gate) ---


def test_every_wire_kind_bearing_class_is_registered_and_vice_versa():
    # Forward: every registry kind == that class's own WIRE_KIND attr.
    for kind, cls in AGENT_KIND_REGISTRY.items():
        assert cls.WIRE_KIND == kind, (
            f"{cls.__name__}.WIRE_KIND={cls.WIRE_KIND!r} but registered "
            f"under {kind!r} — a forgotten/typo'd registration fails here, "
            f"not at a user's load."
        )
    # Reverse: every class importable from the package that declares a
    # WIRE_KIND must be in the registry (a new specialized agent added
    # without a registry entry fails the build).
    for name in agents_pkg.__all__:
        obj = getattr(agents_pkg, name)
        if isinstance(obj, type) and "WIRE_KIND" in vars(obj):
            assert obj in _EXPECTED_SPECIALIZED
            assert obj in AGENT_KIND_REGISTRY.values(), (
                f"{obj.__name__} declares WIRE_KIND but is absent from "
                f"AGENT_KIND_REGISTRY."
            )
    # Every registry kind has a matching typed params spec (single-sourced).
    assert set(_KIND_TO_PARAMS_SPEC) == set(AGENT_KIND_REGISTRY)


# --- AC-17: resolution is registry-gated, not reflection / __all__ ----------


def test_resolution_is_not_reflection_or_dunder_all():
    # A specialized-LIKE Agent subclass that is importable but NOT registered
    # must be unresolvable — resolution is registry-membership-gated.
    class SneakyAgent(Agent):
        WIRE_KIND = "sneaky"

    assert SneakyAgent not in AGENT_KIND_REGISTRY.values()
    assert "sneaky" not in AGENT_KIND_REGISTRY
    # Reflection would have found it via Agent.__subclasses__(); the registry
    # does not.
    assert SneakyAgent in Agent.__subclasses__()
    assert SneakyAgent not in _CLASS_TO_KIND

    # __all__ is the package public API, NOT the specialized set: it carries
    # non-specialized symbols that must never be agent-kinds.
    non_specialized = {"BaseAgent", "Agent", "AgentPool", "MemoryManager",
                       "AgentRegistry", "LearnableAgent", "BaseLearnableAgent"}
    for name in non_specialized:
        assert name in agents_pkg.__all__, name
        assert getattr(agents_pkg, name) not in AGENT_KIND_REGISTRY.values()
