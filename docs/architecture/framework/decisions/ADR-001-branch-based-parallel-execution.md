!!! info "Superseded in implementation"
    The decision recorded here remains in force. The `BranchExecutor` /
    `DynamicBranchSpawner` classes named below were replaced by
    `Orchestrator` + `RealRuntime` in commit `bc19b98` (phase-3 cutover).
    See [ADR-005](ADR-005-unified-barrier-algorithm.md) for the unified-barrier
    algorithm and [ADR-006](ADR-006-deprecation-timeline.md) for the deprecation timeline.

# ADR-001: Branch-Based Parallel Execution

**Status**: Accepted
**Date**: 2026-01-28

## Context
MARSYS must support parallel agent execution, dynamic divergence, and convergence. A simple task queue or single-threaded pipeline cannot represent split/merge workflows or branch-local state needed for multi-agent collaboration.

## Decision
Adopt branch-based execution. Each `ExecutionBranch` is an isolated execution context managed by `BranchExecutor`, while `DynamicBranchSpawner` creates branches at divergence points and coordinates convergence.

## Consequences
- Positive: Parallelism aligns with topology decisions; branch isolation enables deterministic execution and clean aggregation.
- Negative: Increased complexity in branch lifecycle tracking and convergence logic.
- Risks: Branch explosion or resource contention if divergence is unbounded.

## Alternatives Rejected
1. Simple task queue: Rejected because it cannot model branch-local state or convergence semantics.
2. Single sequential pipeline with manual threading: Rejected because it hardcodes concurrency and breaks topology-driven routing.
