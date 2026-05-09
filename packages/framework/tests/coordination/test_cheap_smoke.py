"""Cheap-tier smoke tests — runs in CI on every PR when ANTHROPIC_API_KEY is set.

Single canonical end-to-end test exercising the full stack against
Claude Haiku 4.5: ``ModelConfig`` → adapter retry config → ``Agent.run_step`` →
``AgentMessagesPreparedEvent`` → trace span. Catches breakage that mocked
unit tests can't (auth changes, adapter regressions, schema drift between
the framework and the model API).

Cost: ~$0.01-0.05 per run (one short Haiku 4.5 call).
Runtime: ~1-2 seconds when the network behaves.

Skipped when ``ANTHROPIC_API_KEY`` is not configured — see ``conftest.py``.
This works for both local development (no key → skipped, no failure) and
CI runs from forks (no secret access → skipped, no failure).
"""

from __future__ import annotations

import pytest

from marsys.agents import Agent
from marsys.agents.registry import AgentRegistry
from marsys.coordination import Orchestra
from marsys.coordination.config import ExecutionConfig, StatusConfig


@pytest.mark.cheap
@pytest.mark.asyncio
async def test_single_agent_runs_against_claude_haiku(cheap_model):
    """One agent, one prompt, one tool-less turn against Claude Haiku 4.5.

    Validates: ``ModelConfig`` builds a working ``BaseAPIModel``, the
    adapter accepts auth and returns a non-empty response, the orchestra
    drives a single step to ``terminate_workflow``, and the result
    contains a final response. If this passes on a PR, the framework's
    request/response path against Anthropic is sound.
    """
    AgentRegistry.clear()
    try:
        agent = Agent(
            model_config=cheap_model,
            name="SmokeAgent",
            goal="Answer the user's question concisely.",
            instruction=(
                "You are a smoke-test agent. Reply concisely (under 30 words) "
                "and call `terminate_workflow` with your answer."
            ),
            memory_retention="single_run",
        )

        topology = {
            "agents": ["SmokeAgent"],
            "flows": [],
            "entry_point": "SmokeAgent",
            "exit_points": ["SmokeAgent"],
        }

        result = await Orchestra.run(
            task="What is the capital of France?",
            topology=topology,
            agent_registry=AgentRegistry,
            execution_config=ExecutionConfig(
                status=StatusConfig.from_verbosity(0),
                step_timeout=30.0,
            ),
            max_steps=4,
        )

        assert result.success, f"Orchestra reported failure: {result.error}"
        assert result.final_response, "Final response is empty"
        # Light content check — Haiku reliably names Paris when asked this.
        assert "Paris" in str(result.final_response), (
            f"Expected 'Paris' in final response, got: {result.final_response!r}"
        )
    finally:
        AgentRegistry.clear()
