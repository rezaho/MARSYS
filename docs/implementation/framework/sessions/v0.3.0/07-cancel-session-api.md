# Framework Session 07: `Orchestra.cancel_session` API

**Status: scoped (stub). Full brief written during the framework's own architect-session flow when this session is prioritized. Drafted 2026-05-13 alongside Spren Session 04 architect work.**

Required by Spren v0.3 Session 04 (cancel endpoint backing the canvas Run button's Cancel UX). The Spren-side handler at `POST /v1/runs/{id}/cancel` calls `Orchestra.cancel_session(session_id, force_after=5.0)` and awaits its return; the user-visible 5-second countdown on the Run button mirrors the framework's graceful-drain phase. Other consumers (MARSYS Cloud's hosted control plane, MARSYS Studio's run inspector, third-party Python users invoking `Orchestra.run()` in their own scripts) consume the same API for the same reason — programmatic mid-run termination of a long-running orchestration.

---

## Multi-consumer justification (mandatory)

Every consumer that runs `Orchestra.run()` in a long-lived process eventually needs to cancel a run mid-flight. Today, the only termination paths are (a) waiting for the orchestrator's natural timeouts (`convergence_timeout` / `branch_timeout` / `step_timeout`), or (b) `asyncio.Task.cancel()` on the run task — which is cooperative-cancel-only with no graceful-drain phase. Both are insufficient for user-facing products: (a) is too slow, (b) can leave the framework's `_active_orchestrators` registry inconsistent if cancellation interrupts mid-tick.

This session ships a single primitive — `Orchestra.cancel_session(session_id, force_after=5.0) -> None` — that mirrors the shape of `Orchestra.pause_session()` (Session 03): looks up the live orchestrator in `_active_orchestrators`, runs `quiesce()` for graceful drain (capped by `force_after`), then `task.cancel()` for forced termination if the drain doesn't complete. The result is a clean termination contract every consumer can rely on.

**Forbidden:** any code path special-cased for Spren, or any cancel semantics that only one consumer needs. The framework knows about the cancellation contract; it knows nothing about specific consumers.

---

## The big picture — what we're building and why

### What this PR ships

A new method on `marsys.coordination.orchestra.Orchestra`:

```python
async def cancel_session(
    self,
    session_id: str,
    force_after: float = 5.0,
) -> None:
    """Cleanly halt the run for ``session_id``, force-cancel if not drained
    within ``force_after`` seconds.

    Two-phase semantics:
    1. **Graceful drain** (0 → force_after seconds): looks up the live
       ``Orchestrator`` via ``self._active_orchestrators[session_id]``;
       calls ``await orchestrator.quiesce()`` (Session 03 primitive) inside
       ``asyncio.wait_for(timeout=force_after)``; if drain completes,
       returns and the run task transitions to terminal naturally.
    2. **Forced cancel** (force_after → force_after + 2 seconds): on
       quiesce timeout, calls ``run_task.cancel()`` on the asyncio Task
       running ``Orchestra.run()``; awaits up to 2s more for terminal
       (CancelledError propagates through the orchestrator's finalize
       block; the run task transitions to terminal). Logs WARN if the
       task still isn't terminal after both phases.

    Idempotent: calling on a session that has already terminated is a
    no-op log. Raises ``SessionNotFoundError`` if ``session_id`` is
    unknown and no terminal record exists.

    Distinct from ``pause_session``: cancel is one-way, no resume; no
    state snapshot is written. Distinct from ``discard_paused_session``:
    that targets *paused* runs (deletes the snapshot); ``cancel_session``
    targets *running* runs.
    """
```

That is the entire surface. Roughly 50 LOC additive to `coordination/orchestra.py`. No TRUNK-CRITICAL signature changes. No new dependencies.

### Why this is framework, not a consumer concern

1. **`_active_orchestrators` is framework-internal.** Consumers don't have access to the live `Orchestrator` instance — they hold the `Orchestra` object and the `asyncio.Task`. A consumer-side `task.cancel()` skips the graceful-drain phase entirely and risks leaving `_active_orchestrators` in an inconsistent state.
2. **`quiesce()` is framework-internal** (Session 03). The graceful-drain phase requires calling it; only the framework can.
3. **Symmetry with `pause_session` / `discard_paused_session`.** Those are framework-internal methods on `Orchestra`. Cancel completes the lifecycle-management triad. Pushing cancel out to consumers fragments the contract.
4. **Multi-consumer pattern.** Spren, Cloud, Studio, third-party users all need it. Centralizing in framework follows the same precedent as `TelemetrySink` (Session 02), AG-UI translator (Session 06), `pause_session` (Session 03).

### Why now (in v0.3, not deferred)

Spren v0.3 Session 04 ships a user-facing Cancel button with a 5-second drain UX (per Spren Session 04 §8.7). Without `cancel_session`, that UX is unbacked — the Spren handler can only `task.cancel()` the run task with no drain, and the user-visible "5-second countdown" becomes fictional. Either Framework 07 ships in v0.3 alongside Spren 04, or Spren 04's cancel UX softens to "Stop run (best effort)" with no countdown (a clear product regression).

### Out of scope (deliberate)

| Excluded | Reason | Where it lands |
|---|---|---|
| Resumable cancel | Cancel is one-way by design. Resume = pause. | `pause_session` already covers this. |
| Per-branch cancel | Branches can complete naturally during drain; consumers cancel the whole run. | Future session if multi-branch consumer demand surfaces. |
| Cancel reason tracking | The `runs` row's `error` field is the consumer's concern. | Consumer-side. |
| Cancel callbacks | The `EventBus` already emits `BranchCompletedEvent` + `FinalResponseEvent` on cancel; consumers subscribe. | Existing seam. |

---

## Open questions for the framework architect

These need explicit answers before implementation. Surface in the full brief during architect-session.

1. **`force_after` default.** The stub picks `5.0` to match Spren's UX countdown. Is that the right framework-side default? Larger (10s) gives more drain headroom for tool-heavy runs; smaller (3s) is harsher. Recommendation: keep at `5.0` — Spren's user-visible UX needs it, and other consumers can override per call.

2. **Behavior when `session_id` is unknown.** Raise `SessionNotFoundError`? Return silently? The `pause_session` precedent raises after checking both `_active_orchestrators` and the storage backend. Cancel doesn't have a storage backend equivalent. Recommendation: raise `SessionNotFoundError` for an unknown session_id (no `_active_orchestrators` entry and no terminal record); idempotent no-op log when the session is already terminal.

3. **Behavior on a paused run.** `pause_session` already paused it; should `cancel_session` on a paused run reuse `discard_paused_session` semantics (delete the snapshot, mark cancelled)? Or raise? Recommendation: route through `discard_paused_session` — the paused snapshot becomes garbage on cancel anyway.

4. **Watchdog timeout for the forced phase.** The stub picks 2s after `force_after` for `task.cancel()` to drive terminal. Is 2s realistic for the orchestrator's finalize block to run? Verify against Session 01's tracing-finalize timeout (which is bounded; can be parameterized).

5. **Logging granularity.** WARN if forced phase needed? WARN if forced phase doesn't deliver terminal? INFO on successful graceful drain? Recommendation: INFO on success, WARN on entering forced phase, ERROR on watchdog-exceeded (caller's row stays in `cancelling` until terminal — caller decides escalation).

6. **`Custom("marsys.cancel.timeout")` event.** When the watchdog fires (forced phase didn't deliver terminal), should the framework emit a Custom event on the EventBus so AG-UI / TelemetrySink consumers see it? Recommendation: yes — emit `BranchCompletedEvent(success=False, error="cancel_timeout")` to match the existing `BranchCompletedEvent` shape; AG-UI Session 06's mapping already turns that into `Custom("marsys.branch.completed")` + final `RunError`. No new event type needed.

7. **Concurrency: two `cancel_session` calls on the same session.** Idempotent? Lock? Recommendation: lock per session_id (one cancel-in-flight at a time); subsequent calls return after the first completes.

8. **Test surface.** Real 3-agent run with a long-running tool fixture; assert (a) graceful drain on a fast-tool fixture completes within `force_after`, (b) forced cancel on a hung-tool fixture completes within `force_after + 2`, (c) idempotent on terminal session, (d) `SessionNotFoundError` on unknown session_id, (e) framework regression suite green.

---

## High-level scope

In scope:
- `Orchestra.cancel_session(session_id, force_after=5.0) -> None` async method on `marsys.coordination.orchestra.Orchestra`
- Per-session lock to prevent concurrent cancel calls
- Reuse of `Orchestrator.quiesce()` (Session 03 primitive) for graceful drain
- `task.cancel()` with watchdog for forced phase
- `BranchCompletedEvent(success=False, error="cancel_timeout")` emission on watchdog timeout (consumer observability)
- Integration with `discard_paused_session` for paused-run cancel
- Tests: unit (state machine), integration (real run with fast/slow/hung tool fixtures), regression (existing suite green)
- CHANGELOG entry under `[Unreleased]`

Out of scope:
- New SDK / new namespace (the method lives on existing `Orchestra`)
- Cancel callbacks (consumers subscribe to existing `EventBus` events)
- Per-branch cancel (whole-run only)
- Cancel-then-restart / cancel-then-resume primitives

---

## File map

### Files to modify

- `packages/framework/src/marsys/coordination/orchestra.py` — add `cancel_session` method (~50 LOC additive after `pause_session`); add per-session cancel lock to the existing per-session lock dict
- `packages/framework/CHANGELOG.md` — release entry under `[Unreleased]`
- `packages/framework/tests/coordination/test_orchestra_cancel.py` — NEW test module
- `docs/implementation/framework/v0.3-spren-support.md` — update Session 07 row

### Files to create

- `packages/framework/tests/coordination/test_orchestra_cancel.py` — unit + integration tests

### Files NOT to touch

- TRUNK-CRITICAL: `coordination/execution/orchestrator.py` (`quiesce()` already exists from Session 03), `coordination/execution/real_runtime.py`, `coordination/validation/response_validator.py`, `coordination/topology/graph.py`. `cancel_session` reuses existing primitives.
- `coordination/event_bus.py` — no EventBus changes; cancel reuses existing `BranchCompletedEvent`.

---

## Acceptance

- [ ] `Orchestra.cancel_session(session_id, force_after=5.0) -> None` implemented per the docstring shape above.
- [ ] Graceful drain phase: real 3-agent run with fast-completing tools cancels within `force_after` seconds; final `BranchCompletedEvent(success=False)` fires; row transitions to terminal.
- [ ] Forced phase: real 3-agent run with a deliberately-hung tool fixture cancels within `force_after + 2` seconds via `task.cancel()`; `CancelledError` propagates through the orchestrator's finalize block; row transitions to terminal.
- [ ] Watchdog: if forced phase doesn't deliver terminal within 2s, `BranchCompletedEvent(success=False, error="cancel_timeout")` emitted on the EventBus; method returns and logs WARN.
- [ ] Idempotency: calling `cancel_session` on a session that has already terminated logs INFO and returns; does not raise.
- [ ] Unknown session_id: raises `SessionNotFoundError` (matches `pause_session` precedent for unknown sessions).
- [ ] Paused-run cancel: routes through `discard_paused_session`, deletes the snapshot, marks the run terminal.
- [ ] Concurrency: two concurrent `cancel_session` calls on the same session_id serialize through a per-session lock; the second returns after the first completes (idempotent).
- [ ] Framework regression suite green (zero new failures vs. baseline).
- [ ] No Spren type imported in this PR. No `if running under Spren` code paths.
- [ ] CHANGELOG entry under `[Unreleased]`.
- [ ] `docs/implementation/framework/v0.3-spren-support.md` updated — Session 07 row marked ✅ shipped.

---

## Cross-references

- **Consumer (Spren v0.3 Session 04)**: [`../../../spren/v0.3.0/02-run-execution-and-inspection/sessions/04-run-execution.md`](../../../spren/v0.3.0/02-run-execution-and-inspection/sessions/04-run-execution.md) §3 (cancel endpoint), §6 J-2 (cancel mid-run journey), §8.7 (cancel UX).
- **Related framework primitives**: `Orchestra.pause_session()` (Session 03 — symmetric shape), `Orchestrator.quiesce()` (Session 03 — reused for graceful drain), `Orchestra.discard_paused_session()` (paused-run cancel path).
- **Spren architecture**: [`../../../../architecture/spren/03-api-design.md`](../../../../architecture/spren/03-api-design.md) §Cancellation (extends to add `cancelling` interim status).
- **Spren design principles**: [`../../../../architecture/spren/08-design-principles.md`](../../../../architecture/spren/08-design-principles.md) — SP-018 (framework purity; this is a framework primitive, not a Spren-specific code path).
- **Framework spren-support summary**: [`../../v0.3-spren-support.md`](../../v0.3-spren-support.md) — Session 07 row currently absent; this session adds it.

---

## Sign-off

On completion:

1. Update **What was actually built** below with delta from plan
2. Update [`../../v0.3-spren-support.md`](../../v0.3-spren-support.md) — add Session 07 row to the table
3. Note the framework release version that ships this feature
4. Add **Lessons / Surprises** below

### What was actually built (filled by implementer)

> _Implementer fills this in._

### Lessons / Surprises (filled by implementer)

> _Implementer fills this in._
