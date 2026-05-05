"""Steering manager retry-tiered _action_error_prompt tests.

Verifies that ACTION_ERROR steering varies deterministically by retry_count so
the LLM is less likely to collapse into repetition when the same error fires
multiple times.
"""
from __future__ import annotations

from marsys.coordination.steering.manager import (
    ErrorContext,
    SteeringContext,
    SteeringManager,
)
from marsys.coordination.validation.types import ValidationErrorCategory


_SENTINEL = object()


def _ctx(retry: int, neighbors=None, available=_SENTINEL) -> SteeringContext:
    if available is _SENTINEL:
        available = ["invoke_agent"]
    return SteeringContext(
        agent_name="Researcher",
        available_actions=available,
        error_context=ErrorContext(
            category=ValidationErrorCategory.ACTION_ERROR,
            error_message="No coordination tool call detected.",
            retry_count=retry,
            failed_action="content_only",
        ),
        topology_neighbors=neighbors or [],
    )


def test_round_one_is_generic():
    mgr = SteeringManager()
    out = mgr._action_error_prompt(_ctx(retry=2))
    assert "Action error" in out
    assert "invoke_agent" in out
    # Round 1 should not name a specific peer
    assert "Coordinator" not in out


def test_round_two_emphasizes_tool_choice():
    mgr = SteeringManager()
    out = mgr._action_error_prompt(_ctx(retry=3))
    assert "must select" in out or "must invoke" in out.lower()
    assert "invoke_agent" in out


def test_round_three_plus_includes_topology_peer():
    mgr = SteeringManager()
    out = mgr._action_error_prompt(_ctx(retry=5, neighbors=["Coordinator"]))
    assert "Coordinator" in out
    assert "invoke_agent" in out


def test_three_rounds_produce_different_messages():
    mgr = SteeringManager()
    r1 = mgr._action_error_prompt(_ctx(retry=2))
    r2 = mgr._action_error_prompt(_ctx(retry=3))
    r3 = mgr._action_error_prompt(_ctx(retry=5, neighbors=["Coordinator"]))
    assert r1 != r2
    assert r2 != r3
    assert r1 != r3


def test_round_three_with_terminate_workflow_only():
    mgr = SteeringManager()
    out = mgr._action_error_prompt(
        _ctx(retry=5, neighbors=[], available=["terminate_workflow"])
    )
    assert "terminate_workflow" in out


def test_round_three_with_ask_user_only():
    mgr = SteeringManager()
    out = mgr._action_error_prompt(
        _ctx(retry=5, neighbors=[], available=["ask_user"])
    )
    assert "ask_user" in out


def test_empty_coord_tools_has_clear_signal():
    mgr = SteeringManager()
    out = mgr._action_error_prompt(_ctx(retry=2, available=[]))
    assert "review topology" in out.lower() or "(none" in out
