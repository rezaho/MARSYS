/**
 * Client-side cost estimator for the live Run-button ticker.
 *
 * Mirrors the (provider, model) → rate table from
 * ``packages/spren/src/spren/cost_rates.py``. The server is the source
 * of truth for ``runs.total_cost_usd`` (Spren stores the aggregated
 * value); this client-side mirror exists ONLY to update the live
 * counter on the Run button between server polls.
 *
 * Unknown (provider, model) → 0 (matches server semantics; Spren server
 * also logs WARN on unknowns).
 */
import type { GenerationMetadata } from "./run-sse";

interface ClientRate {
  inputPer1M: number;
  outputPer1M: number;
  reasoningPer1M?: number;
}

const OAUTH_ALIAS: Record<string, string> = {
  "openai-oauth": "openai",
  "anthropic-oauth": "anthropic",
};

const RATES: Record<string, Record<string, ClientRate>> = {
  anthropic: {
    "claude-opus-4-7": { inputPer1M: 15.0, outputPer1M: 75.0 },
    "claude-sonnet-4-6": { inputPer1M: 3.0, outputPer1M: 15.0 },
    "claude-haiku-4-5": { inputPer1M: 1.0, outputPer1M: 5.0 },
    "claude-haiku-4-5-20251001": { inputPer1M: 1.0, outputPer1M: 5.0 },
  },
  openai: {
    "gpt-5": { inputPer1M: 2.5, outputPer1M: 10.0 },
    "gpt-5-mini": { inputPer1M: 0.25, outputPer1M: 1.25 },
    o3: { inputPer1M: 2.0, outputPer1M: 8.0, reasoningPer1M: 2.0 },
    "o3-mini": { inputPer1M: 1.1, outputPer1M: 4.4, reasoningPer1M: 1.1 },
  },
  google: {
    "gemini-2.5-pro": { inputPer1M: 1.25, outputPer1M: 10.0 },
    "gemini-2.5-flash": { inputPer1M: 0.1, outputPer1M: 0.4 },
  },
  xai: {
    "grok-4": { inputPer1M: 3.0, outputPer1M: 15.0 },
    "grok-4-mini": { inputPer1M: 0.3, outputPer1M: 1.5 },
  },
  openrouter: {
    "anthropic/claude-opus-4-7": { inputPer1M: 15.0, outputPer1M: 75.0 },
    "anthropic/claude-sonnet-4-6": { inputPer1M: 3.0, outputPer1M: 15.0 },
    "openai/gpt-5": { inputPer1M: 2.5, outputPer1M: 10.0 },
    "google/gemini-2.5-pro": { inputPer1M: 1.25, outputPer1M: 10.0 },
  },
};

export function calculateCostFromMetadata(metadata: GenerationMetadata): number {
  const canonical = OAUTH_ALIAS[metadata.provider] ?? metadata.provider;
  const rate = RATES[canonical]?.[metadata.model];
  if (!rate) return 0;
  const reasoningRate = rate.reasoningPer1M ?? 0;
  return (
    metadata.prompt_tokens * rate.inputPer1M +
    metadata.completion_tokens * rate.outputPer1M +
    (metadata.reasoning_tokens ?? 0) * reasoningRate
  ) / 1_000_000;
}
