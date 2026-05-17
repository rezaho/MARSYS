"""Façade identity — the session's core premise (AC-1 / AC-1b / AC-2).

Spren must *re-export* the framework canonical wire types, not re-declare a
look-alike mirror. Every other test would still pass against a hand-rolled
mirror with matching fields/enum values; only object identity proves the
mirror is gone. Frozen module paths per acceptance.md.
"""
from __future__ import annotations

import enum

import pytest

import spren.models as sm


def test_wire_types_are_the_framework_objects():
    """AC-1: each name is the SAME object as the framework symbol."""
    from marsys.agents.serialize import AgentSpec, MemoryRetention
    from marsys.coordination.serialize import (
        ConvergencePolicyConfigSpec,
        ExecutionConfigSpec,
        StatusConfigSpec,
        TracingConfigSpec,
    )
    from marsys.coordination.topology.serialize import (
        EdgeSpec,
        NodeSpec,
        TopologySpec,
        WorkflowDefinition,
    )
    from marsys.models.serialize import ApiProvider, ModelConfigSpec, ModelType

    assert sm.NodeSpec is NodeSpec
    assert sm.EdgeSpec is EdgeSpec
    assert sm.TopologySpec is TopologySpec
    assert sm.WorkflowDefinition is WorkflowDefinition
    assert sm.AgentSpec is AgentSpec
    assert sm.MemoryRetention is MemoryRetention
    assert sm.ModelConfigSpec is ModelConfigSpec
    assert sm.ModelType is ModelType
    assert sm.ApiProvider is ApiProvider
    assert sm.ExecutionConfigSpec is ExecutionConfigSpec
    assert sm.StatusConfigSpec is StatusConfigSpec
    assert sm.ConvergencePolicyConfigSpec is ConvergencePolicyConfigSpec
    assert sm.TracingConfigSpec is TracingConfigSpec


def test_edge_enums_are_the_core_enums_not_serialize_literals():
    """AC-1b: EdgeType/EdgePattern are the real Enums from
    marsys.coordination.topology.core — NOT the serialize string-literal
    aliases."""
    from marsys.coordination.topology import core

    assert issubclass(sm.EdgeType, enum.Enum)
    assert issubclass(sm.EdgePattern, enum.Enum)
    assert sm.EdgeType is core.EdgeType
    assert sm.EdgePattern is core.EdgePattern
    # NodeKind is likewise the real core enum with exactly the 4 members.
    assert issubclass(sm.NodeKind, enum.Enum)
    assert sm.NodeKind is core.NodeKind
    assert {m.value for m in sm.NodeKind} == {"agent", "start", "end", "user"}


@pytest.mark.parametrize("removed", ["NodeType", "NodeCategory", "category_of"])
def test_dropped_mirror_symbols_are_unimportable(removed):
    """AC-2: the Spren-local taxonomy is gone — these no longer import."""
    with pytest.raises(ImportError):
        exec(f"from spren.models import {removed}")
