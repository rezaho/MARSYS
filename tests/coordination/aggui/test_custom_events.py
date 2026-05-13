"""Custom event Pydantic models — strict validation tests."""

from __future__ import annotations

import pytest

pytest.importorskip("ag_ui")

from pydantic import ValidationError

from marsys.coordination.aggui.custom_events import (
    CUSTOM_EVENT_REGISTRY,
    StreamLaggedValue,
    validate_custom_value,
)


def test_registry_lists_expected_custom_events():
    expected = {
        "marsys.aggui.handshake",
        "marsys.stream.lagged",
        "marsys.error",
        "marsys.resource.limit",
        "marsys.generation.metadata",
        "marsys.branch.created",
        "marsys.branch.completed",
        "marsys.parallel.group",
        "marsys.convergence",
        "marsys.user_interaction.pending",
        "marsys.user_interaction.resolved",
        "marsys.user_interaction.timeout",
        "marsys.memory.compaction",
    }
    assert set(CUSTOM_EVENT_REGISTRY.keys()) == expected


def test_strict_validation_returns_model_instance():
    v = validate_custom_value("marsys.stream.lagged", {"count": 3})
    assert isinstance(v, StreamLaggedValue)
    assert v.count == 3


def test_strict_validation_raises_on_missing_field():
    with pytest.raises(ValidationError):
        validate_custom_value("marsys.branch.created", {"branch_id": "b1"})


def test_strict_validation_raises_on_wrong_type():
    with pytest.raises(ValidationError):
        validate_custom_value("marsys.stream.lagged", {"count": "not-an-int-shaped-string"})


def test_unknown_name_raises_key_error():
    with pytest.raises(KeyError) as excinfo:
        validate_custom_value("marsys.something.new", {})
    assert "marsys.something.new" in str(excinfo.value)


def test_branch_created_full_payload_validates():
    v = validate_custom_value(
        "marsys.branch.created",
        {
            "branch_id": "br1",
            "branch_name": "Researcher",
            "source_agent": "Coordinator",
            "target_agents": ["Researcher"],
            "trigger_type": "invoke",
            "parent_branch_id": None,
        },
    )
    assert v.target_agents == ["Researcher"]


def test_parallel_group_status_is_validated_against_literal():
    # Should accept "started"
    v = validate_custom_value(
        "marsys.parallel.group",
        {
            "group_id": "g1",
            "agent_names": ["A", "B"],
            "status": "started",
            "completed_count": 0,
            "total_count": 2,
        },
    )
    assert v.status == "started"
    # Should reject "in_progress" (not in the Literal)
    with pytest.raises(ValidationError):
        validate_custom_value(
            "marsys.parallel.group",
            {
                "group_id": "g1",
                "agent_names": [],
                "status": "in_progress",
                "completed_count": 0,
                "total_count": 0,
            },
        )
