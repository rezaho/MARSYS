!!! info "Superseded in implementation"
    The decision recorded here remains in force. The `BranchExecutor` mentioned
    in this ADR was replaced by `Orchestrator` + `RealRuntime` in commit `bc19b98`.
    The validator is now invoked from `RealRuntime.step()` via `StepExecutor`.
    See [ADR-005](ADR-005-unified-barrier-algorithm.md) for the algorithm.

# ADR-002: Centralized Response Validation

**Status**: Accepted
**Date**: 2026-01-28

## Context
Agent responses must be parsed into actions and validated against topology permissions. If parsing is duplicated in agents or executors, behavior becomes inconsistent across formats and error handling diverges.

## Decision
Centralize all response parsing and action validation in `ValidationProcessor`. Agents return `Message` objects only, and `BranchExecutor` invokes the validator for routing decisions.

## Consequences
- Positive: Consistent parsing, unified error handling, and format pluggability.
- Negative: Validator becomes a critical dependency and must stay aligned with formats.
- Risks: Validation bugs can halt execution across the system.

## Alternatives Rejected
1. Per-agent parsing: Rejected because it duplicates logic and bypasses topology checks.
2. Parsing inside StepExecutor: Rejected because it couples step execution with routing policy.
