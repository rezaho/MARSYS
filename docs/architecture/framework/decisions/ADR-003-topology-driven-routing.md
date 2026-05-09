# ADR-003: Topology-Driven Routing

**Status**: Accepted
**Date**: 2026-01-28

## Context
Routing decisions must remain consistent with the declared topology. Hardcoded routes or agent-specific shortcuts create brittle workflows and undermine the topology as the single source of truth.

## Decision
All routing and transition validation consult `TopologyGraph`. The Router (and related coordination logic) uses graph permissions for every transition and convergence decision.

## Consequences
- Positive: Topology remains authoritative; routing changes are driven by topology updates.
- Negative: Misconfigured topology can block execution until corrected.
- Risks: Overly strict topology validation may require more upfront topology design.

## Alternatives Rejected
1. Hardcoded routing rules: Rejected because they diverge from topology and are hard to maintain.
2. Agent-self-routing without graph validation: Rejected because it bypasses coordination constraints.
