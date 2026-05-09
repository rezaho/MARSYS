# ADR-007: Pause/Resume Snapshot — Quiesce Primitive, Snapshot Surface, and Storage Protocol

**Status**: Accepted
**Date**: 2026-05-09 (drafted), 2026-05-10 (approved)
**Implements**: Framework Session 03 — `docs/implementation/framework/sessions/v0.3.0/03-pause-resume-completion.md`
**Related**: ADR-005 (unified-barrier algorithm — defines the state this ADR snapshots)

## Context

Spren v0.4-29 (run-inspector pause/resume UI + REST endpoints + meta-agent tools) and MARSYS Cloud's managed long-runs both need to pause an in-flight workflow, persist its state, and later resume from that state — possibly in a different process. The current `Orchestra.pause_session` / `resume_session` methods are stubs (`pause_session` writes a snapshot in the dead `ExecutionBranch` shape; `resume_session` returns a placeholder `OrchestraResult` with a `# TODO: Implement proper state restoration` marker).

The legacy `coordination/state/state_manager.py` (`StateSnapshot` dataclass, `_serialize_branches` / `_deserialize_branches` helpers, `StateManager.save_session` / `load_session` / `pause_execution` / `resume_execution`) targets the dead `ExecutionBranch` shape from `coordination/branches/types.py` — built before the unified-barrier orchestrator landed (ADR-005). It cannot be patched in place because the orchestrator's mutable state is now a different shape (`branches: dict[str, Branch]` + `barriers: dict[str, Barrier]` + `runnable: deque[str]` + `_fire_queue: list[str]` + `convergence_barriers: dict[str, str]` + `_completed_emitted: set[str]` + `_user_interactions: deque` + `_user_interaction_inflight: bool`).

This PR is CRITICAL-tier per the framework's `CLAUDE.md` because it adds public methods to TRUNK-CRITICAL files (`Orchestra`, `Orchestrator`) and rewrites two TRUNK-CRITICAL public method bodies (`Orchestra.pause_session`, `Orchestra.resume_session`). The `CLAUDE.md` policy mandates an ADR before any code is written.

### Why the existing `Orchestrator` cannot be paused without new primitives

`Orchestrator.run()` (orchestrator.py:162-228) dispatches every `RUNNING` branch as an `asyncio.Task`, awaits `FIRST_COMPLETED`, and re-dispatches newly-runnable branches in a tight loop. There is no `_pause_requested` signal, no quiesce point, no way for an external caller to halt the loop cleanly without setting `_workflow_error` (which would cause a failure cascade). Pausing without a clean signal would either (a) snapshot mid-tick state (undefined: a tick mutates branches/barriers atomically inside an `await runtime.step(branch)` — interrupting it leaves arbitrary partial state), or (b) cancel in-flight tasks (loses paid-for LLM work and has the same partial-state problem).

Adding a `quiesce()` primitive is therefore not optional — it's the load-bearing primitive that makes pause's semantics defined.

### Why `snapshot()` cannot live outside `Orchestrator`

Reading the orchestrator's mutable state from outside the class would require exposing (`branches`, `barriers`, `convergence_barriers`, `runnable`, `_fire_queue`, etc.) as a public surface, which makes every external caller a potential mutator and breaks the orchestrator's encapsulation guarantees. The class already owns this state; the snapshot is a deep-copy read-only view of fields it knows are coherent at quiesce time. An adapter would be a wrapper with no boundary value (alternative considered + rejected below).

### Why `Orchestra` must hold a reference to the running orchestrator

`Orchestra.execute()` constructs `Orchestrator` as a local variable (orchestra.py:856). External callers (Spren's REST endpoint handler, Cloud's pause-on-autoscale event handler) call `Orchestra.pause_session(session_id)` from a different async task than the one awaiting `execute()`. Without a `dict[session_id, Orchestrator]` lookup on `Orchestra`, `pause_session` has no way to reach the live orchestrator.

## Decision

This PR adds the following to TRUNK-CRITICAL files. All changes are additive except the rewrites of `Orchestra.pause_session` / `resume_session` bodies (already approved as non-additive by the brief's escalation gate — those methods are stubs today, so the rewrite produces no behavior change for any currently-working caller).

### 1. New methods on `Orchestrator` (`coordination/execution/orchestrator.py`)

```python
class Orchestrator:
    # Existing __init__ adds:
    self._pause_requested: Optional[asyncio.Event] = None  # lazy-init in quiesce()

    async def quiesce(self) -> None:
        """Set the pause flag and await all in-flight branch ticks (the
        loop's tracked asyncio.Tasks) to complete. Idempotent. Cooperative:
        the run-loop body checks _pause_requested between dispatch
        iterations and exits without setting _workflow_error.
        """

    def snapshot(self) -> OrchestratorState:
        """Return a deep-copy of mutable state. Caller MUST have called
        quiesce() (or be operating outside an active run); calling snapshot()
        while branches are dispatching is undefined.
        """

    def restore_from(self, state: OrchestratorState) -> None:
        """Replace mutable state with `state`. Must be called on a freshly-
        constructed Orchestrator (asserts self.branches == {} and
        self.root_barrier_id is None).
        """

    async def resume(self) -> WorkflowResult:
        """Run-loop entry point used after restore_from(). Does NOT call
        init_workflow (state is already populated). Returns a WorkflowResult
        matching the shape of run().
        """
```

The internal refactor: `Orchestrator.run()`'s loop body (lines 179-227) is extracted into a private `_dispatch_loop()` shared by `run()` and `resume()`. The public `run()` signature is preserved.

### 2. New methods on `Orchestra` (`coordination/orchestra.py`)

```python
class Orchestra:
    # Existing __init__ adds:
    self._active_orchestrators: dict[str, Orchestrator] = {}

    # Existing __init__ kwargs add:
    storage_backend: StorageBackend | None = None
    snapshot_retention: timedelta = timedelta(days=30)

    # Existing __init__ kwargs remove:
    state_manager: StateManager | None  # StateManager class deleted in this PR

    async def pause_session(self, session_id: str) -> None: ...
    async def resume_session(self, session_id: str) -> OrchestraResult: ...
    async def list_paused_sessions(self) -> list[PausedSessionMetadata]: ...
    async def discard_paused_session(self, session_id: str) -> None: ...
    def _wire_event_bus(self) -> None: ...

    # Existing methods removed:
    async def create_checkpoint(...): ...   # CheckpointManager removed
    async def restore_checkpoint(...): ...  # CheckpointManager removed
```

`Orchestra.execute()` is modified to populate `self._active_orchestrators[session_id] = orchestrator` after construction (orchestra.py:856) and pop it in the `finally` block.

### 3. New on-disk `StateSnapshot` (Pydantic v2)

Lives at `coordination/state/snapshot.py`. The model is the on-disk wire shape; the in-memory `OrchestratorState` is a sibling dataclass. The Orchestra layer maps between them — set→list serialization for `Barrier.candidates/upstream/downstream`, `Message.model_dump()` round-trip for `Branch.memory` items that are Pydantic models, plain dicts for items that aren't. Non-JSON-serializable values in `Barrier.arrived` raise a clear error at pause time (no silent corruption).

`framework_version: str` is matched exact-string on resume; mismatch raises `IncompatibleSnapshotError`. v0.3 ships no migration tooling.

### 4. New `StorageBackend` Protocol (`coordination/state/storage.py`)

```python
class StorageBackend(Protocol):
    async def read(self, key: str) -> bytes: ...
    async def write(self, key: str, data: bytes) -> None: ...  # MUST be atomic
    async def delete(self, key: str) -> None: ...
    async def list_with_metadata(self) -> list[StorageEntry]: ...
    async def expire_older_than(self, age: timedelta) -> int: ...
```

`FileStorageBackend(root: Path)` ships in this PR — the only concrete implementation. Atomic-write semantics: temp-file-then-`os.replace` + `fsync(parent_dir_fd)` (POSIX; no-op on Windows). MARSYS Cloud's `S3StorageBackend` and CI integrations' artifact-store backend are out-of-scope follow-ups.

### 5. Removed surface (CHANGELOG-documented)

- `StateManager` class — removed wholesale.
- `CheckpointManager` class — removed (it was wired through `StateManager.load_session` / `save_session` / etc., all of which the new shape doesn't have).
- `Orchestra.create_checkpoint` / `Orchestra.restore_checkpoint` — removed (delegated to the deleted `StateManager`).
- `Orchestra.__init__(state_manager=...)` keyword — removed.
- `Orchestra.run(state_manager=...)` keyword — removed.
- `tests/coordination/test_state_manager_integration.py` — deleted.

## Rationale

### Why `snapshot()` / `restore_from()` / `quiesce()` / `resume()` live on `Orchestrator` directly (not on a sibling `OrchestratorStateAdapter`)

The snapshot reads state the class already owns. An adapter would be a wrapper that needs attribute access to `branches`, `barriers`, etc. — meaning either those fields become public (eliminating the encapsulation argument for an adapter) or the adapter accesses private fields (defeating the slim-orchestrator argument). The existing `DetNodeContext` Protocol pattern is the right precedent for *behavior* boundaries (where the orchestrator exposes a narrow API for non-LLM nodes); state is not behavior. Splitting would add ceremony without reducing complexity.

### Why pause is at-tick-boundary, not at-instruction-granularity

`runtime.step(branch)` performs an `await` (LLM call, tool call) and returns a `StepResult`. Mid-`await` interruption produces undefined partial state in `branch.memory` and any `tool_calls` already emitted. Tick-boundary pause is the only sane semantic — it captures a state where every branch is either fully `WAITING`/`TERMINATED`/etc. (settled) or hasn't started its next tick yet. The at-least-once tool-call contract (resumed runs may re-execute tool calls in-flight at pause time) is the trade-off.

### Why JSON, not pickle / MessagePack

Pickle is version-fragile and a security risk on load (untrusted snapshot files). MessagePack is faster but less debuggable; for snapshots that fire at user pause, the 2x speed advantage doesn't pay for the cross-language inspectability loss. JSON is debuggable, version-portable, and inspectable from any language.

### Why exact-string `framework_version` match (no semver-minor compatibility)

Patch releases that change orchestrator state shape (e.g., adding a field on `Branch`) would silently corrupt paused snapshots if compatibility were minor-version-loose. Conservative exact-match catches this before corruption. A future framework PR can add migration tooling and relax the comparison once the migration story is real.

### Why no auto-checkpoint in this PR

Auto-checkpoint while running (continuous snapshotting at orchestrator natural quiesce points so daemon crashes recover in-flight runs) is a follow-up framework session. It uses the primitives this PR ships — `quiesce()`, `snapshot()`, `restore_from()` — without further TRUNK-CRITICAL touches. The seam is the explicit "design hooks for future auto-checkpoint" scope decision recorded in Phase A: these primitives are public, idempotent, and reentrant for exactly this reason.

### Why `_active_orchestrators: dict[str, Orchestrator]` (not a registry class, not a session-handle redesign)

`Orchestra` is the existing facade. Adding a registry class (`OrchestratorRegistry`) would create one more indirection for one method's lookup. Refactoring to a session-handle pattern (`Orchestra.start_session() -> handle`) would change the public API for every existing caller — out of scope for a "complete pause/resume" PR. A dict on `Orchestra` mirrors the existing `self._sessions` (orchestra.py:152) and is the smallest delta.

### Why `StorageBackend` is a Protocol (not an ABC)

Cloud / CI integrations' backends will live outside the framework. Forcing them to inherit from a framework class creates a coupling (every cloud-side change to the backend would require coordinating with the framework's class hierarchy). Protocol satisfies the boundary purely structurally. The trade-off — `FileStorageBackend` doesn't formally announce its protocol satisfaction — is acceptable since `FileStorageBackend` ships in the same module as the Protocol and a unit test verifies satisfaction via a `runtime_checkable` Protocol or a duck-type assertion.

(The improver flagged a concern that the existing `StorageBackend` ABC was being broken by the Protocol switch; that concern was resolved by removing `CheckpointManager` and `StateManager` entirely — the only consumers of the legacy ABC's methods. With those gone, the new Protocol has no incompatibility.)

## Alternatives considered

### Pickle-based snapshot
Rejected: version-fragile (Python pickle protocol changes across minor releases), security risk on load (pickle.load is RCE on untrusted files), opaque on disk (no way to debug a corrupt snapshot without Python).

### MessagePack-based snapshot
Rejected: ~2x faster than JSON but loses debuggability and cross-language inspectability. Snapshots fire at user pause action — sub-100ms latency is the floor either way, so the speed advantage doesn't matter for the dominant case.

### Cancel-and-rollback to last barrier on pause
Rejected: loses paid-for LLM work (an agent that just made a $0.40 LLM call gets rolled back), complicates barrier semantics (would require synthetic "cancelled at pause" branch states), and produces worse UX for the dominant case (user pauses to inspect, doesn't expect rollback).

### Snapshot via `OrchestratorStateAdapter` sibling class
Rejected: ceremony with no boundary value. The adapter would either expose state to a public surface (defeats encapsulation) or access private fields (defeats slim-orchestrator). The existing `DetNodeContext` Protocol pattern is for behavior, not state.

### Continuous auto-checkpoint in this PR
Rejected: significantly larger scope (frequency triggers, mutex semantics, performance impact, ADR section on serialization-cost-vs-staleness tradeoff). Decoupled into a follow-up that uses the primitives shipped here. Phase A scope decision: ship pause/resume primitive + design hooks (~10 LoC of seams in the form of public, idempotent, reentrant methods).

### Bumping `__version__` from `0.2.1` to `0.3.0` in this PR
Deferred: belongs to a separate release-prep PR alongside CHANGELOG release-note items documenting the surface removals. The version-mismatch test in this PR is portable across actual `__version__` values — it writes a known-bad string (`"0.0.0-mismatch-test"`) and asserts the resume path raises.

### Keeping `CheckpointManager` as a thin adapter over the new `StorageBackend`
Rejected: adds scope without consumer demand. The only external callers of `CheckpointManager` are `tests/coordination/test_state_manager_integration.py` (deleted in this PR) and `Orchestra.create_checkpoint` / `restore_checkpoint` (also removed). If a future use case surfaces, a separate session can restore it.

### `is_idempotent: bool` flag on the tool registry + warning at workflow definition time
Deferred: the field doesn't exist anywhere in the framework. Adding it touches the tool registry surface and the docs/help system — a separate small follow-up PR with proper plumbing. Tool idempotency is documented as the user's responsibility in v0.3.

## Consequences

### Backward compatibility

- **Breaking**: `StateManager`, `CheckpointManager`, `Orchestra.create_checkpoint` / `Orchestra.restore_checkpoint`, `Orchestra(state_manager=...)`, `Orchestra.run(state_manager=...)`. All removed. Documented in CHANGELOG. No external in-tree callers (verified by grep at PR time); external consumers must migrate to the new pause/resume API.
- **Snapshot format**: legacy `StateSnapshot` (dataclass shape from `state_manager.py`) cannot be read by this version. Any in-flight v0.2 paused runs are unrecoverable post-merge; users with ongoing v0.2 work should resume those runs before upgrading.
- **Public-surface additions**: `Orchestra.pause_session`, `resume_session`, `list_paused_sessions`, `discard_paused_session`; `Orchestrator.snapshot`, `restore_from`, `quiesce`, `resume`; `StorageBackend`, `FileStorageBackend`, `StateSnapshot`, `IncompatibleSnapshotError`. All new — no backward-compat concern.

### Operational

- **Snapshot retention**: 30-day default sweeper runs once on `Orchestra.__init__`. Long-running daemons that never reconstruct `Orchestra` won't sweep automatically; either re-construct periodically or call `expire_older_than` directly. This is acceptable for v0.3 since the dominant use case (Spren daemon) re-constructs `Orchestra` on each run.
- **Atomic-write durability**: `os.replace` + `fsync(parent_dir_fd)` is correct on POSIX. On Windows, parent-dir fsync is a no-op, but `os.replace` is atomic per `MoveFileEx` semantics. Tested via simulated mid-write crash.
- **At-least-once tool calls**: paused-then-resumed runs may re-execute tool calls in-flight at pause time. Documented as user responsibility. Tools that mutate external state non-idempotently (e.g., charge a credit card, send an email) need user-side mitigation. A future small PR can add `is_idempotent: bool` on tool registry; out of scope here.
- **Skipped scheduled events on resume**: barriers with `policy.timeout` deadlines crossed during the paused interval are skipped on resume and logged at WARNING. Users who want re-anchored deadlines must re-set them explicitly post-resume.

### Performance

- Snapshot serialization: O(branches + barriers) deep-copy plus JSON encode. On typical Spren workflows (<100 branches, <50 barriers, <1MB total memory), pause should complete in <50ms. Documented in test assertions for the multi-consumer test.
- Quiesce wait: bounded by the longest in-flight tick (typically one LLM call: 1-30s). Pause response time is therefore dominated by LLM latency, not framework overhead. Acceptable for the user-pause use case.

### Future-proofing (the "design seams")

The four primitives `quiesce()`, `snapshot()`, `restore_from()`, `resume()` are public, idempotent, and reentrant by deliberate design. A follow-up framework session can add continuous auto-checkpoint by:

1. Defining a `SnapshotScheduler` Protocol with one method (e.g., `async def should_snapshot(orchestrator) -> bool`).
2. The orchestrator's `_dispatch_loop` calls `scheduler.should_snapshot()` after `_drain_fires()` between dispatch iterations.
3. If true: `quiesce()` → `snapshot()` → write asynchronously (without blocking the loop) → continue dispatch.

No further TRUNK-CRITICAL changes needed. This is the explicit Phase A scope decision: ship pause primitive + design hooks for future auto-checkpoint, recorded as item D in the brief's "Items open for future framework sessions" section.

## Approval

This ADR requires framework-team approval before any code is written. Approval is recorded by the framework lead amending this section with their name + date, OR by an explicit approval message in the PR thread.

- [x] Framework lead approval: rezaho (Reza Hosseini), 2026-05-10

---

**Sign-off note**: The implementer Phase A work that produced this ADR included a session-plan-validator pass (8 failed claims surfaced and resolved), a session-plan-improver pass (10 improvements applied — all merged into the brief), 5 user-confirmed scope decisions captured above, and a frozen `acceptance.md` with 63 criteria at `docs/implementation/framework/sessions/v0.3.0/03-pause-resume-completion/acceptance.md`. Implementation begins after this ADR is approved.
