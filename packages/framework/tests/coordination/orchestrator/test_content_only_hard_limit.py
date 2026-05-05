"""Hard-limit FAIL path for content-only loops in RealRuntime.

When a branch emits CONTENT_ONLY_HARD_LIMIT consecutive content-only responses
despite steering, RealRuntime.step short-circuits and returns
StepResult(kind="FAIL", error=<diagnostic>). The diagnostic must name the
agent, list available coordination tools, and point at the likely root cause.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from marsys.coordination.execution.orchestrator_types import Branch
from marsys.coordination.execution.real_runtime import (
    CONTENT_ONLY_HARD_LIMIT,
    CONTENT_ONLY_STEERING_THRESHOLD,
    RealRuntime,
)


def _make_branch(consecutive: int) -> Branch:
    return Branch(
        id="br_0001",
        current_agent="Researcher",
        status="RUNNING",
        delivery_target="root",
        input="task",
        memory=[
            {"role": "user", "content": "research the speed of light"},
            {"role": "assistant", "content": "## final_response\n\nHere are findings..." * 5},
        ],
        consecutive_content_only=consecutive,
    )


def _make_runtime(topology_graph=None) -> RealRuntime:
    instance = SimpleNamespace(
        name="Researcher",
        tools_schema=[
            {"function": {"name": "plan_create"}},
            {"function": {"name": "plan_update"}},
        ],
        memory=SimpleNamespace(to_messages=lambda: []),
    )

    registry = MagicMock()
    registry.get_or_acquire = MagicMock(return_value=instance)

    runtime = RealRuntime(
        registry=registry,
        step_executor=MagicMock(),
        validator=MagicMock(),
        topology_graph=topology_graph,
        session_id="test-session",
        execution_config=None,
    )
    runtime._current_instance = instance
    return runtime


@pytest.mark.asyncio
async def test_hard_limit_fires_at_threshold():
    """At CONTENT_ONLY_HARD_LIMIT, branch fails with diagnostic before re-invoking step_executor."""
    runtime = _make_runtime()
    branch = _make_branch(consecutive=CONTENT_ONLY_HARD_LIMIT)

    result = await runtime.step(branch)

    assert result.kind == "FAIL"
    assert "Researcher" in result.error
    assert "Content-only loop" in result.error
    runtime.step_executor.execute_step.assert_not_called()


@pytest.mark.asyncio
async def test_hard_limit_does_not_fire_below_threshold():
    """One below the hard limit, the branch is still given a chance (steering still applies)."""
    runtime = _make_runtime()
    branch = _make_branch(consecutive=CONTENT_ONLY_HARD_LIMIT - 1)

    runtime.step_executor.execute_step = MagicMock()

    async def _stub(*args, **kwargs):
        return SimpleNamespace(
            success=True,
            coordination_action=None,
            coordination_data=None,
            tool_calls=None,
            response="content only",
        )

    runtime.step_executor.execute_step = _stub
    result = await runtime.step(branch)

    assert result.kind != "FAIL" or "Content-only loop" not in (result.error or "")


@pytest.mark.asyncio
async def test_hard_limit_diagnostic_includes_topology_tools():
    """When topology supports terminate_workflow, the diagnostic mentions it."""
    topology = MagicMock()
    topology.get_next_agents = MagicMock(return_value=["Coordinator"])
    topology.has_edge_to_endnode = MagicMock(return_value=False)
    topology.has_edge_to_usernode = MagicMock(return_value=False)

    runtime = _make_runtime(topology_graph=topology)
    branch = _make_branch(consecutive=CONTENT_ONLY_HARD_LIMIT)

    result = await runtime.step(branch)

    assert result.kind == "FAIL"
    assert "invoke_agent" in result.error


@pytest.mark.asyncio
async def test_hard_limit_diagnostic_includes_last_assistant_content():
    """Diagnostic captures a snippet of what the agent kept emitting."""
    runtime = _make_runtime()
    branch = _make_branch(consecutive=CONTENT_ONLY_HARD_LIMIT)
    branch.memory = [
        {"role": "user", "content": "research"},
        {"role": "assistant", "content": "STUCK_RESPONSE_SIGNAL"},
    ]

    result = await runtime.step(branch)

    assert result.kind == "FAIL"
    assert "STUCK_RESPONSE_SIGNAL" in result.error


def test_constants_relationship():
    """Steering threshold must be strictly less than the hard limit."""
    assert CONTENT_ONLY_STEERING_THRESHOLD < CONTENT_ONLY_HARD_LIMIT
    assert CONTENT_ONLY_HARD_LIMIT == 10


def test_execution_config_validates_threshold_below_hard_limit():
    """ExecutionConfig.__post_init__ rejects threshold >= hard_limit."""
    from marsys.coordination.config import ExecutionConfig
    with pytest.raises(ValueError, match="content_only_steering_threshold"):
        ExecutionConfig(
            content_only_steering_threshold=10,
            content_only_hard_limit=10,
        )
    with pytest.raises(ValueError):
        ExecutionConfig(
            content_only_steering_threshold=15,
            content_only_hard_limit=5,
        )


def test_execution_config_default_thresholds_consistent():
    """Default ExecutionConfig honors threshold < hard_limit."""
    from marsys.coordination.config import ExecutionConfig
    cfg = ExecutionConfig()
    assert cfg.content_only_steering_threshold < cfg.content_only_hard_limit


@pytest.mark.asyncio
async def test_hard_limit_reads_from_execution_config():
    """Custom hard-limit value on ExecutionConfig is honored by RealRuntime."""
    from marsys.coordination.config import ExecutionConfig
    cfg = ExecutionConfig(
        content_only_steering_threshold=1,
        content_only_hard_limit=3,
    )

    runtime = _make_runtime()
    runtime.execution_config = cfg

    # At 3 consecutive content-only — exactly the configured hard limit — fail.
    branch = _make_branch(consecutive=3)
    result = await runtime.step(branch)
    assert result.kind == "FAIL"
    assert "Content-only loop" in result.error
    runtime.step_executor.execute_step.assert_not_called()
