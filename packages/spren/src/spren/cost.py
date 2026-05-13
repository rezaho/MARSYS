"""Per-run cost calculation.

Consumes ``Custom("marsys.generation.metadata")`` events from the AG-UI
stream (per Framework 06's mapping for ``GenerationEvent``); the payload
shape is ``{model, provider, prompt_tokens, completion_tokens,
reasoning_tokens, finish_reason}``.

Aggregates per run via ``apply_to_run`` — increments
``total_cost_usd`` / ``total_tokens_input`` / ``total_tokens_output`` on
the ``runs`` row.

Missing rate (unknown provider/model) → log WARN naming the
(provider, model) pair, emit zero. We don't fabricate prices for
unknown models (SP-007).
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta

from spren.cost_rates import LAST_UPDATED, OAUTH_PROVIDER_ALIAS, RATES, CostRate
from spren.storage.runs import apply_cost_delta

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerationCost:
    """Computed cost for one generation. Returned by ``calculate_cost``."""

    cost_usd: float
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    rate_known: bool


def _resolve_rate(provider: str, model: str) -> CostRate | None:
    canonical_provider = OAUTH_PROVIDER_ALIAS.get(provider, provider)
    return RATES.get((canonical_provider, model))


def calculate_cost(
    *,
    provider: str,
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    reasoning_tokens: int = 0,
) -> GenerationCost:
    """Compute USD cost for a single generation. Unknown rate → 0.0 + WARN."""
    rate = _resolve_rate(provider, model)
    if rate is None:
        logger.warning(
            "cost: no rate for provider=%s model=%s; emitting zero cost",
            provider,
            model,
        )
        return GenerationCost(
            cost_usd=0.0,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            rate_known=False,
        )

    reasoning_rate = rate.reasoning_per_1m_usd or 0.0
    cost_usd = (
        prompt_tokens * rate.input_per_1m_usd
        + completion_tokens * rate.output_per_1m_usd
        + reasoning_tokens * reasoning_rate
    ) / 1_000_000

    return GenerationCost(
        cost_usd=cost_usd,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        reasoning_tokens=reasoning_tokens,
        rate_known=True,
    )


def apply_to_run(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    cost: GenerationCost,
) -> None:
    """Increment per-run aggregates from a computed GenerationCost."""
    apply_cost_delta(
        conn,
        run_id=run_id,
        cost_usd=cost.cost_usd,
        tokens_in=cost.prompt_tokens,
        tokens_out=cost.completion_tokens,
    )


def warn_if_rates_stale(today: date | None = None) -> bool:
    """Logs WARN if the rate table is older than 90 days. Returns True if stale.

    Called at daemon start (server.create_app lifespan handler).
    """
    today = today or date.today()
    age = today - LAST_UPDATED
    if age > timedelta(days=90):
        logger.warning(
            "cost: cost_rates.py LAST_UPDATED=%s is %d days old (>90); "
            "model pricing may be stale. Update packages/spren/src/spren/cost_rates.py.",
            LAST_UPDATED.isoformat(),
            age.days,
        )
        return True
    return False
