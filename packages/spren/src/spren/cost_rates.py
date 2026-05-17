"""Provider/model cost rate table.

Rates are USD per 1,000,000 tokens. ``reasoning_per_1m_usd`` is None for
providers that don't bill reasoning tokens separately (treated as zero in
``cost.calculate_cost``).

Lookup key: ``(provider, model_name)`` — both must match. Unknown keys
trigger ``cost.calculate_cost`` to log WARN and emit zero (SP-007 — we
don't fabricate prices for unknown models).

OAuth providers (``openai-oauth``, ``anthropic-oauth``) intentionally
absent — they share rates with their non-OAuth counterparts and
``cost.calculate_cost`` cross-references via ``OAUTH_PROVIDER_ALIAS``.

``LAST_UPDATED`` is checked at daemon start; we log WARN if older than
90 days. Auto-update from provider pricing APIs is deferred to v0.4.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class CostRate:
    input_per_1m_usd: float
    output_per_1m_usd: float
    reasoning_per_1m_usd: float | None = None


LAST_UPDATED: date = date(2026, 5, 13)


# Aliases for OAuth provider variants — treat them as their underlying provider
# for rate lookup. Cost is the same; auth path is different.
OAUTH_PROVIDER_ALIAS: dict[str, str] = {
    "openai-oauth": "openai",
    "anthropic-oauth": "anthropic",
}


# (provider, model_name) → CostRate
RATES: dict[tuple[str, str], CostRate] = {
    # Anthropic — Claude 4 family (per anthropic.com/pricing as of LAST_UPDATED)
    ("anthropic", "claude-opus-4-7"): CostRate(15.00, 75.00),
    ("anthropic", "claude-sonnet-4-6"): CostRate(3.00, 15.00),
    ("anthropic", "claude-haiku-4-5"): CostRate(1.00, 5.00),
    ("anthropic", "claude-haiku-4-5-20251001"): CostRate(1.00, 5.00),
    # OpenAI — GPT-5 family + o-series (per openai.com/pricing as of LAST_UPDATED)
    ("openai", "gpt-5"): CostRate(2.50, 10.00),
    ("openai", "gpt-5-mini"): CostRate(0.25, 1.25),
    ("openai", "o3"): CostRate(2.00, 8.00, reasoning_per_1m_usd=2.00),
    ("openai", "o3-mini"): CostRate(1.10, 4.40, reasoning_per_1m_usd=1.10),
    # Google — Gemini 2.5 family
    ("google", "gemini-2.5-pro"): CostRate(1.25, 10.00),
    ("google", "gemini-2.5-flash"): CostRate(0.10, 0.40),
    # xAI — Grok 4
    ("xai", "grok-4"): CostRate(3.00, 15.00),
    ("xai", "grok-4-mini"): CostRate(0.30, 1.50),
    # OpenRouter passthrough — covers a handful of common gateway-routed models.
    # The OpenRouter API returns the underlying provider in metadata; for keys
    # that aren't in this table we warn-and-emit-zero (per SP-007).
    ("openrouter", "anthropic/claude-opus-4-7"): CostRate(15.00, 75.00),
    ("openrouter", "anthropic/claude-sonnet-4-6"): CostRate(3.00, 15.00),
    ("openrouter", "openai/gpt-5"): CostRate(2.50, 10.00),
    ("openrouter", "google/gemini-2.5-pro"): CostRate(1.25, 10.00),
}
