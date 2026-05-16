"""Property-based round-trip tests for the topology serializer.

Hypothesis strategies (in ``strategies.py``) generate valid topologies and
PatternConfigs; each generated input is round-tripped through
``workflow_to_pydantic`` → ``pydantic_to_topology`` and asserted equal under
``topology_equals``. Configured with ≥100 examples per shape.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings

from marsys.coordination.topology.converters.pattern_converter import (
    PatternConfigConverter,
)
from marsys.coordination.topology.patterns import PatternConfig, PatternType
from marsys.coordination.topology.serialize import (
    pydantic_to_topology,
    topology_equals,
    workflow_to_pydantic,
)

from .strategies import pattern_config_strategy, topology_strategy


@pytest.fixture(autouse=True)
def _api_key_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")


@given(topology=topology_strategy())
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_random_topology_round_trips(topology):
    spec = workflow_to_pydantic(None, topology)
    rehydrated = pydantic_to_topology(spec, tool_registry={})
    assert topology_equals(topology, rehydrated)


@given(pattern_config=pattern_config_strategy())
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_random_pattern_round_trips(pattern_config):
    topology = PatternConfigConverter.convert(pattern_config)
    spec = workflow_to_pydantic(None, topology)
    rehydrated = pydantic_to_topology(spec, tool_registry={})
    assert topology_equals(topology, rehydrated)

    # Recovered provenance rebuilds an equivalent PatternConfig.
    recovered = rehydrated.metadata["original_pattern"]
    assert recovered["pattern"] == pattern_config.pattern.value
    rebuilt_config = PatternConfig(
        pattern=PatternType(recovered["pattern"]),
        params=recovered["params"],
        metadata=recovered["metadata"],
    )
    rebuilt_topology = PatternConfigConverter.convert(rebuilt_config)
    assert topology_equals(topology, rebuilt_topology)
