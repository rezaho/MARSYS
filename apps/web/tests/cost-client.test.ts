/**
 * Unit tests for the client-side cost estimator.
 *
 * Mirrors the server-side cost test cases. Server is the source of
 * truth for ``runs.total_cost_usd``; this client mirror only updates
 * the live counter on the Run button between server polls.
 */
import { describe, expect, it } from "vitest";

import { calculateCostFromMetadata } from "../src/lib/cost-client";

describe("calculateCostFromMetadata", () => {
  it("computes anthropic claude-opus-4-7 cost", () => {
    const cost = calculateCostFromMetadata({
      provider: "anthropic",
      model: "claude-opus-4-7",
      prompt_tokens: 1000,
      completion_tokens: 500,
    });
    // 1000 * 15 + 500 * 75 = 15000 + 37500 = 52500 / 1M = 0.0525
    expect(cost).toBeCloseTo(0.0525, 4);
  });

  it("computes openai o3 cost with reasoning tokens", () => {
    const cost = calculateCostFromMetadata({
      provider: "openai",
      model: "o3",
      prompt_tokens: 1000,
      completion_tokens: 500,
      reasoning_tokens: 2000,
    });
    // 1000*2 + 500*8 + 2000*2 = 2000 + 4000 + 4000 = 10000 / 1M = 0.010
    expect(cost).toBeCloseTo(0.01, 4);
  });

  it("returns zero for unknown rate", () => {
    const cost = calculateCostFromMetadata({
      provider: "anthropic",
      model: "claude-fictional",
      prompt_tokens: 1000,
      completion_tokens: 500,
    });
    expect(cost).toBe(0);
  });

  it("aliases openai-oauth to openai", () => {
    const cost = calculateCostFromMetadata({
      provider: "openai-oauth",
      model: "gpt-5",
      prompt_tokens: 1000,
      completion_tokens: 500,
    });
    // 1000 * 2.5 + 500 * 10 = 2500 + 5000 = 7500 / 1M = 0.0075
    expect(cost).toBeCloseTo(0.0075, 4);
  });

  it("aliases anthropic-oauth to anthropic", () => {
    const cost = calculateCostFromMetadata({
      provider: "anthropic-oauth",
      model: "claude-sonnet-4-6",
      prompt_tokens: 1000,
      completion_tokens: 500,
    });
    // 1000*3 + 500*15 = 3000 + 7500 = 10500 / 1M = 0.0105
    expect(cost).toBeCloseTo(0.0105, 4);
  });

  it("ignores reasoning tokens for models without a reasoning rate", () => {
    const cost = calculateCostFromMetadata({
      provider: "anthropic",
      model: "claude-opus-4-7",
      prompt_tokens: 1000,
      completion_tokens: 500,
      reasoning_tokens: 5000,
    });
    expect(cost).toBeCloseTo(0.0525, 4);
  });

  it("returns zero for unknown provider", () => {
    const cost = calculateCostFromMetadata({
      provider: "unknown",
      model: "any",
      prompt_tokens: 100,
      completion_tokens: 100,
    });
    expect(cost).toBe(0);
  });
});
