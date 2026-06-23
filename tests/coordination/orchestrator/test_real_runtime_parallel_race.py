"""Regression for framework issue #40: RealRuntime must not cross-talk the
per-branch agent identity across concurrent ``step()`` calls.

``RealRuntime`` is constructed once per ``Orchestra.run()`` and the orchestrator
dispatches ``step()`` for every runnable branch as a concurrent asyncio task on
that one shared object. If the per-branch agent instance is round-tripped through
shared mutable state (written before the ``execute_step`` await, read back at
translate time), a sibling branch can clobber it during the await — so one
branch's coordination action is validated against a *different* agent's identity
(and therefore its outgoing topology edges), producing a fabricated
``"Agent X cannot invoke: [...]"`` failure attributed to the wrong agent.

This test forces the racy interleave deterministically (no LLM, no real timing):
``execute_step`` yields control with ``await asyncio.sleep(0)`` between the
instance acquisition and the translate-time validator read, so both branches'
writes land before either's read. Each branch must still be validated against
its own acquired identity.

It is RED on the pre-fix code (shared ``self._current_instance``) and GREEN once
the instance is threaded as a parameter.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from marsys.coordination.execution.orchestrator_types import Branch
from marsys.coordination.execution.real_runtime import RealRuntime
from marsys.coordination.validation.response_validator import ActionType


def _make_branch(branch_id: str, agent: str) -> Branch:
    return Branch(
        id=branch_id,
        current_agent=agent,
        status="RUNNING",
        delivery_target="root",
        input="go",
        memory=[],
        consecutive_content_only=0,
    )


@pytest.mark.asyncio
async def test_parallel_step_does_not_cross_talk_agent_identity():
    """Two concurrent step() calls must each validate against their own agent."""
    # Two distinct agent instances with different identities.
    instances = {
        "AgentA": SimpleNamespace(
            name="AgentA", memory=SimpleNamespace(to_messages=lambda: [])
        ),
        "AgentC": SimpleNamespace(
            name="AgentC", memory=SimpleNamespace(to_messages=lambda: [])
        ),
    }

    registry = MagicMock()
    registry.get_or_acquire = MagicMock(
        side_effect=lambda agent, branch_id: instances[agent]
    )

    # execute_step yields control (so a sibling task can clobber shared state, if
    # the buggy path exists), then returns an invoke_agent action tagged with the
    # acting agent's identity — read from the agent= it was actually called with.
    async def _execute_step(*, agent, request, memory, context):
        await asyncio.sleep(0)  # force the racy interleave
        return SimpleNamespace(
            success=True,
            coordination_action="invoke_agent",
            coordination_data={"acting_agent": agent.name},
            tool_calls=None,
            response=None,
        )

    step_executor = MagicMock()
    step_executor.execute_step = _execute_step

    # The validator records (action's acting agent, identity it was handed) and is
    # valid only when they match — i.e. the identity threaded into translation is
    # the branch's own. On cross-talk they differ -> invalid -> FAIL. Mocking at
    # the validator boundary keeps topology_graph=None out of the edge-check.
    seen: list[tuple[str, str]] = []

    async def _validate(*, action, data, agent, branch, exec_state):
        acting = data["acting_agent"]
        seen.append((acting, agent.name))
        ok = acting == agent.name
        return SimpleNamespace(
            is_valid=ok,
            error_message=(
                None
                if ok
                else f"identity cross-talk: action by {acting!r} "
                f"validated against {agent.name!r}"
            ),
            action_type=ActionType.INVOKE_AGENT,
            next_agent="Orchestrator",
            invocations=[SimpleNamespace(agent_name="Orchestrator", request="x")],
        )

    validator = MagicMock()
    validator.validate_coordination_action = _validate

    runtime = RealRuntime(
        registry=registry,
        step_executor=step_executor,
        validator=validator,
        topology_graph=None,
        session_id="s",
        execution_config=None,
    )

    res_a, res_c = await asyncio.gather(
        runtime.step(_make_branch("br_a", "AgentA")),
        runtime.step(_make_branch("br_c", "AgentC")),
    )

    # Core property: every validation saw the acting branch's own identity.
    assert all(acting == received for acting, received in seen), (
        f"validator was handed a cross-attributed identity: {seen}"
    )
    # Neither branch fails on a fabricated cross-agent validation error.
    assert res_a.kind != "FAIL", res_a.error
    assert res_c.kind != "FAIL", res_c.error
    # Sanity: the happy path actually ran (both translated to SINGLE_INVOKE).
    assert res_a.kind == "SINGLE_INVOKE"
    assert res_c.kind == "SINGLE_INVOKE"
