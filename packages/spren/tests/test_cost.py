"""Unit tests for cost calculation."""
from __future__ import annotations

import logging
from datetime import date, timedelta

import pytest

from spren.cost import (
    GenerationCost,
    apply_to_run,
    calculate_cost,
    warn_if_rates_stale,
)
from spren.cost_rates import LAST_UPDATED, OAUTH_PROVIDER_ALIAS, RATES
from spren.storage.db import Database
from spren.storage.migrations.runner import MigrationsRunner
from spren.storage.runs import fetch_run, insert_run
from spren.models import TaskInput


def test_known_rate_anthropic_opus():
    cost = calculate_cost(
        provider="anthropic",
        model="claude-opus-4-7",
        prompt_tokens=1000,
        completion_tokens=500,
    )
    assert cost.rate_known is True
    # 1000 * 15/1M + 500 * 75/1M = 0.015 + 0.0375 = 0.0525
    assert cost.cost_usd == pytest.approx(0.0525)
    assert cost.prompt_tokens == 1000
    assert cost.completion_tokens == 500


def test_known_rate_openai_with_reasoning():
    cost = calculate_cost(
        provider="openai",
        model="o3",
        prompt_tokens=1000,
        completion_tokens=500,
        reasoning_tokens=2000,
    )
    assert cost.rate_known is True
    # 1000 * 2/1M + 500 * 8/1M + 2000 * 2/1M = 0.002 + 0.004 + 0.004 = 0.010
    assert cost.cost_usd == pytest.approx(0.010)


def test_known_rate_no_reasoning_field_treats_as_zero():
    cost = calculate_cost(
        provider="anthropic",
        model="claude-opus-4-7",
        prompt_tokens=1000,
        completion_tokens=500,
        reasoning_tokens=10000,  # rate has no reasoning_per_1m_usd → 0
    )
    # Only prompt+completion contribute
    assert cost.cost_usd == pytest.approx(0.0525)


def test_unknown_rate_emits_zero_with_warning(caplog):
    caplog.set_level(logging.WARNING)
    cost = calculate_cost(
        provider="anthropic",
        model="claude-fictional-7-trillion",
        prompt_tokens=1000,
        completion_tokens=500,
    )
    assert cost.rate_known is False
    assert cost.cost_usd == 0.0
    assert any("no rate for" in r.message for r in caplog.records)
    assert any("claude-fictional-7-trillion" in r.message for r in caplog.records)


def test_unknown_provider_emits_zero(caplog):
    caplog.set_level(logging.WARNING)
    cost = calculate_cost(
        provider="some-new-provider",
        model="any-model",
        prompt_tokens=100,
    )
    assert cost.rate_known is False
    assert cost.cost_usd == 0.0


def test_oauth_provider_aliases_to_underlying_rate():
    """openai-oauth shares rates with openai."""
    cost = calculate_cost(
        provider="openai-oauth",
        model="gpt-5",
        prompt_tokens=1000,
        completion_tokens=500,
    )
    assert cost.rate_known is True
    # 1000 * 2.5/1M + 500 * 10/1M = 0.0025 + 0.005 = 0.0075
    assert cost.cost_usd == pytest.approx(0.0075)


def test_anthropic_oauth_aliases_to_anthropic():
    cost = calculate_cost(
        provider="anthropic-oauth",
        model="claude-sonnet-4-6",
        prompt_tokens=1000,
        completion_tokens=500,
    )
    assert cost.rate_known is True


def test_zero_tokens_zero_cost():
    cost = calculate_cost(
        provider="anthropic",
        model="claude-opus-4-7",
    )
    assert cost.cost_usd == 0.0
    assert cost.prompt_tokens == 0
    assert cost.completion_tokens == 0


def test_apply_to_run_increments_aggregates(data_dir):
    db = Database(data_dir)
    runner = MigrationsRunner(db.connection)
    runner.run()

    # Need a workflow row for the FK
    from spren.models import WorkflowDefinition
    from spren.storage.workflows import insert_workflow
    from spren.models.topology import TopologySpec

    insert_workflow(
        db.connection,
        workflow_id="wf-1",
        name="test",
        description=None,
        definition=WorkflowDefinition(topology=TopologySpec(nodes=[], edges=[]), agents={}),
        provenance="api",
        provenance_metadata=None,
    )

    insert_run(
        db.connection,
        run_id="run-1",
        workflow_id="wf-1",
        task_input=TaskInput(),
    )

    cost1 = calculate_cost(
        provider="anthropic", model="claude-opus-4-7",
        prompt_tokens=1000, completion_tokens=500,
    )
    apply_to_run(db.connection, run_id="run-1", cost=cost1)

    cost2 = calculate_cost(
        provider="anthropic", model="claude-opus-4-7",
        prompt_tokens=2000, completion_tokens=1000,
    )
    apply_to_run(db.connection, run_id="run-1", cost=cost2)

    db.connection.commit()
    run = fetch_run(db.connection, "run-1")
    assert run is not None
    assert run.total_tokens_input == 3000
    assert run.total_tokens_output == 1500
    assert run.total_cost_usd == pytest.approx(0.0525 + 0.105)


def test_warn_if_rates_stale_under_90_days_silent(caplog):
    caplog.set_level(logging.WARNING)
    is_stale = warn_if_rates_stale(today=LAST_UPDATED + timedelta(days=89))
    assert is_stale is False
    assert not any("LAST_UPDATED" in r.message for r in caplog.records)


def test_warn_if_rates_stale_over_90_days_warns(caplog):
    caplog.set_level(logging.WARNING)
    is_stale = warn_if_rates_stale(today=LAST_UPDATED + timedelta(days=91))
    assert is_stale is True
    assert any("LAST_UPDATED" in r.message for r in caplog.records)


def test_rate_table_includes_anthropic_openai_google_xai_openrouter():
    """All five v0.3 providers per architecture/06-observability + plan §3."""
    providers = {p for p, _ in RATES.keys()}
    assert "anthropic" in providers
    assert "openai" in providers
    assert "google" in providers
    assert "xai" in providers
    assert "openrouter" in providers


def test_oauth_aliases_present():
    assert OAUTH_PROVIDER_ALIAS["openai-oauth"] == "openai"
    assert OAUTH_PROVIDER_ALIAS["anthropic-oauth"] == "anthropic"
