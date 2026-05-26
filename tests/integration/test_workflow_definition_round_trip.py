"""Integration test: full WorkflowDefinition round-trip on a non-trivial topology.

Builds a complex pattern-based topology (hub-and-spoke with 5 spokes + parallel
rule), serializes via ``workflow_to_pydantic``, rehydrates via
``pydantic_to_topology``, and asserts the rehydrated topology is structurally
identical (every node, every edge, every rule preserved). Pattern provenance
survives the round-trip and rebuilds an equivalent ``PatternConfig``.

NOTE: This test does NOT call ``Orchestra.run()``. Doing so would require live
LLM calls (cost + flakiness). The structural round-trip is the contract; the
runtime would invoke topology routing identically against either the original
or the rehydrated topology because all of `Topology`, `TopologyGraph`,
and the orchestrator state machine read from the same node/edge surfaces this
test asserts equal.
"""

from __future__ import annotations

import asyncio

import pytest

from marsys.coordination.topology.converters.pattern_converter import (
    PatternConfigConverter,
)
from marsys.coordination.topology.patterns import PatternConfig, PatternType
from marsys.coordination.topology.serialize import (
    pydantic_to_topology,
    topology_equals,
    workflow_to_pydantic,
)


@pytest.fixture(autouse=True)
def _api_key_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")


def test_hub_and_spoke_five_spokes_round_trip_with_parallel_rule():
    config = PatternConfig.hub_and_spoke(
        hub="Coordinator",
        spokes=["Searcher", "Analyzer", "Writer", "Reviewer", "Publisher"],
        parallel_spokes=True,
    )
    original = PatternConfigConverter.convert(config)

    # Sanity: pattern provenance was written by the converter edit.
    assert "original_pattern" in original.metadata
    assert original.metadata["original_pattern"]["pattern"] == "hub_and_spoke"
    # The parallel rule was added by the converter.
    assert len(original.rules) >= 1

    spec = workflow_to_pydantic(None, original)
    rehydrated = asyncio.run(pydantic_to_topology(spec, tool_registry={}))

    assert topology_equals(original, rehydrated)
    assert len(rehydrated.nodes) == len(original.nodes)
    assert {n.name for n in rehydrated.nodes} == {n.name for n in original.nodes}
    assert len(rehydrated.edges) == len(original.edges)

    # Recovered provenance rebuilds an equivalent topology.
    recovered = rehydrated.metadata["original_pattern"]
    rebuilt_config = PatternConfig(
        pattern=PatternType(recovered["pattern"]),
        params=recovered["params"],
        metadata=recovered["metadata"],
    )
    rebuilt = PatternConfigConverter.convert(rebuilt_config)
    assert topology_equals(original, rebuilt)


def test_hierarchical_round_trip_preserves_tree_structure():
    config = PatternConfig.hierarchical(
        tree={
            "Lead": ["Designer", "Engineer"],
            "Designer": ["Illustrator"],
            "Engineer": ["Backend", "Frontend"],
        }
    )
    original = PatternConfigConverter.convert(config)
    spec = workflow_to_pydantic(None, original)
    rehydrated = asyncio.run(pydantic_to_topology(spec, tool_registry={}))
    assert topology_equals(original, rehydrated)
    # Tree edges: Lead→Designer, Lead→Engineer, Designer→Illustrator, Engineer→Backend, Engineer→Frontend, User→Lead
    assert len(rehydrated.edges) == len(original.edges)


def test_mesh_round_trip_preserves_bidirectional_fan_out():
    config = PatternConfig.mesh(agents=["A", "B", "C", "D"])
    original = PatternConfigConverter.convert(config)
    spec = workflow_to_pydantic(None, original)
    rehydrated = asyncio.run(pydantic_to_topology(spec, tool_registry={}))
    assert topology_equals(original, rehydrated)
    # Fully-connected mesh of 4 nodes: 6 unordered pairs, each bidirectional
    # yields 12 directed edges on the runtime side.
    assert len(rehydrated.edges) == 12


def test_serialization_is_idempotent_under_repeated_round_trip():
    config = PatternConfig.hub_and_spoke(hub="Hub", spokes=["S1", "S2", "S3"])
    once = PatternConfigConverter.convert(config)

    twice_spec = workflow_to_pydantic(None, once)
    twice = asyncio.run(pydantic_to_topology(twice_spec, tool_registry={}))

    thrice_spec = workflow_to_pydantic(None, twice)
    thrice = asyncio.run(pydantic_to_topology(thrice_spec, tool_registry={}))

    assert topology_equals(once, twice)
    assert topology_equals(twice, thrice)
    assert len(twice_spec.topology.edges) == len(thrice_spec.topology.edges)
