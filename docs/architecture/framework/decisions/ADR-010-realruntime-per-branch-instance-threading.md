# ADR-010: `RealRuntime` threads the per-branch agent instance — no shared per-step state

**Status**: Accepted
**Date**: 2026-06-23
**Implements**: Framework Session 16 — `docs/implementation/framework/sessions/v0.3.0/16-realruntime-current-instance-race.md`
**Fixes**: Issue #40 (parallel-invoke validation cross-talk)
**Related**: ADR-002 (centralized response validation — the validator this bug fed a wrong identity), ADR-005 (unified-barrier algorithm — defines the concurrent dispatch that surfaced the race); DP-004 (branch isolation)

> **Numbering note:** the decisions directory already contains two `ADR-009-*` files (`-full-payload-llm-tracing`, `-specialized-agent-serialization`) — a pre-existing collision. ADR-010 is the next free integer; renumbering 009 is out of scope for this fix.

## Context

`RealRuntime` is TRUNK-CRITICAL (`docs/architecture/framework/overview.md` §3) and is constructed **once per `Orchestra.run()`**. The orchestrator dispatches `RealRuntime.step(branch)` for every runnable branch as a concurrent `asyncio.Task` on that single shared object (the documented concurrency model — "true parallelism for I/O-bound `runtime.step` calls"). Sync sections between awaits run atomically, but `step()` spans an `await` (the `StepExecutor.execute_step` LLM/tool call).

`step()` stashed the per-branch agent instance on a shared attribute — `self._current_instance = instance` — *before* that await, and `_translate()` read it back *after* the await to feed `ValidationProcessor.validate_coordination_action(agent=...)`. Under `parallel_invoke` fan-out, a sibling branch's `step()` overwrote `self._current_instance` during the await, so a branch's coordination action was validated against a **different** agent's identity — and therefore against that other agent's outgoing topology edges. The validator keys its edge check off `agent.name → topology_graph.get_next_agents(agent.name)`, so the result was a fabricated `"Agent <X> cannot invoke: [...]"` failure attributed to the wrong agent.

The bug was latent when fanned-out workers shared identical outgoing edges (cross-talk passed validation silently) and surfaced non-deterministically when edge sets differed. This is a direct violation of **DP-004 branch isolation** ("Each branch maintains its own memory, trace, metadata, and status; no cross-branch sharing or mutation is allowed") — the shared `_current_instance` *was* cross-branch mutation of per-step state.

## Decision

Thread the per-branch agent instance **explicitly as a parameter** and remove the shared attribute entirely:

- `_translate(self, marsys_result, branch)` → `_translate(self, marsys_result, branch, instance)`.
- The validator call inside `_translate` passes `agent=instance` (the parameter) instead of `agent=getattr(self, "_current_instance", None)`.
- `self._current_instance = instance` is deleted from `step()`.

After this change `RealRuntime` holds **zero per-step mutable state**: its instance attributes are all read-only configuration set at construction; every per-step value (`instance`, `branch`, `context`, `marsys_result`) is a local threaded as a parameter.

This is non-additive (a private method signature changes; a private attribute is removed) and therefore TRUNK-CRITICAL — but the public surface is untouched: the `Runtime` Protocol requires only `step(branch)`, whose signature is unchanged, and `_translate` / `_current_instance` are both private with a single internal call site.

### Forward constraint

**Do not reintroduce per-step caching on the shared `RealRuntime`.** Per-branch state is threaded as a parameter, never stashed on `self`. The runtime is shared across concurrently-dispatched branches; any `self.<x> = <per-step value>` written before an `await` and read after it is a DP-004 violation and a latent cross-talk race. Run-lifetime configuration on `self` is fine; per-tick values are not.

## Rationale

The fix conforms the lone outlier to the file's own existing convention. `step()` already passes `instance` explicitly to `execute_step` (`agent=instance`) and to `_build_content_only_diagnostic(branch, instance)` — `_current_instance` was the single place a per-step value round-tripped through shared mutable state. Parameter-passing is the day-one shape: had the file been written with concurrent dispatch in mind, `_translate` would have taken `instance` like its siblings. No new mechanism is introduced; this is extend-and-unify, not add-parallel.

## Alternatives considered

### `contextvars.ContextVar` for the current instance
Rejected. A `ContextVar` is task-local and would also be race-free, but it stands up a *parallel* propagation mechanism beside the explicit parameter-passing the file already uses everywhere. It is less explicit, heavier, and a worse fit — a future reader would ask why one per-step value travels by context while all others travel by parameter.

### Construct a fresh `RealRuntime` per branch
Rejected. Defeats the documented "one runtime per `Orchestra.run()`" design and is a far larger TRUNK-CRITICAL blast radius (touches the orchestrator's dispatch construction) for no benefit over parameter-passing.

## Consequences

- **No backward-compatibility concern.** The public `Runtime` Protocol and `step(branch)` signature are unchanged; no consumer depends on the private `_translate` signature or the `_current_instance` attribute.
- **Correctness.** `parallel_invoke` fan-out to agents with heterogeneous outgoing edges no longer cross-talks; each branch is validated against its own identity regardless of interleaving. DP-004 compliance is restored for the step/translate path.
- **Regression guard.** A deterministic, no-LLM test (`tests/coordination/orchestrator/test_real_runtime_parallel_race.py`) forces the racy interleave via `asyncio.gather` + an `execute_step` that yields control; it is RED on the pre-fix code and GREEN after. The issue-#40 live `parallel_invoke` reproducer remains manual verification only (provider key, timing-dependent) — not a committed CI test.

## Approval

This ADR requires framework-team approval before code is written (TRUNK-CRITICAL policy).

- [x] Framework lead approval: rezaho (Reza Hosseini), 2026-06-23 (approved the TRUNK-CRITICAL touch in the session that produced this ADR)
