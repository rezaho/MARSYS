# ADR-004: Payload Recovery and Boundary Semantics

**Status**: Accepted  
**Date**: 2026-02-10

## Context

Large multimodal traces (especially repeated screenshots/tool payloads) can exceed provider request-size limits even when token estimates remain below threshold. Without a dedicated recovery path, repeated retries can keep failing and may amplify payload growth.

At the same time, compaction boundaries needed to be consistent across compaction reasons. The previous grace/tail semantics were harder to reason about across processors and payload recovery behavior.

## Decision

1. Introduce explicit API classification `request_too_large` (`REQUEST_TOO_LARGE`):
   - Provider-agnostic `HTTP 413` mapping
   - Message-based override for payload hints (for providers returning `400` with payload-limit text)

2. Route payload-too-large recovery through `Agent.run_step()`:
   - `Agent._run()` re-raises `ModelAPIError` when classification is `request_too_large`
   - `run_step()` catches it, invokes memory payload compaction, re-prepares messages, and retries
   - Retries are hard-bounded to `2` attempts to prevent infinite loops

3. Add payload compaction API in managed memory:
   - `ManagedConversationMemory.compact_for_payload_error(runtime=...) -> bool`
   - Delegation in `MemoryManager.compact_for_payload_error(...)`
   - Payload recovery runs summarization on the compacted prefix (not the full token-trigger processor chain)
   - Success is based on serialized payload-byte reduction
   - Compaction lifecycle events (`started/completed/failed`) are emitted consistently

4. Standardize protected-tail boundary semantics:
   - `_compute_protected_tail_start(messages, grace_recent_messages=n)` now means:
     "protect from the n-th most recent assistant message onward"
   - This assistant-round boundary is reused across summarization/backward packing and payload recovery splits

5. Coordination handling:
   - `REQUEST_TOO_LARGE` is treated as terminal in shared error-message processing to avoid unbounded orchestration-level retry loops.

## Consequences

- Positive:
  - Deterministic, bounded recovery for oversize payload failures
  - Better alignment with DP-001 (`_run` remains pure; recovery orchestration stays in `run_step`)
  - Unified, predictable tail-boundary semantics across processors
  - Improved observability through compaction events and metadata

- Negative:
  - Additional complexity in `run_step` retry path and memory compaction behavior
  - Payload-byte estimation is heuristic, not a provider-native exact size calculation

- Risk:
  - Some payloads may remain too large after compaction and still fail after max retries

## Alternatives Rejected

1. Handle payload recovery only in provider adapters  
   Rejected: adapters lack agent-memory context and cannot perform context-aware compaction.

2. Keep converting all `_run()` exceptions directly into error messages  
   Rejected: prevents `run_step()` from performing targeted payload recovery before validation routing.

3. Use unbounded retries for payload errors  
   Rejected: violates robustness expectations and risks infinite failure loops.
