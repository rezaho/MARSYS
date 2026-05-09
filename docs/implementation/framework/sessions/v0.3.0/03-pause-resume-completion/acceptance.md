# Acceptance criteria — Framework Session 03: Pause/Resume Completion

Frozen at 2026-05-09T00:00:00Z. The test auditor reads ONLY this file plus the test files. Implementation source is intentionally hidden.

## Functional

### Pydantic snapshot model — `StateSnapshot`

- AC-1: `StateSnapshot` is a Pydantic v2 model importable from `marsys.coordination.state` (re-exported from the public `coordination/state/__init__.py`).
- AC-2: `StateSnapshot` round-trips losslessly through `model_dump_json()` followed by `StateSnapshot.model_validate_json(...)` — the resulting model equals the original by field-by-field comparison.
- AC-3: `StateSnapshot` exposes its JSON Schema via `StateSnapshot.model_json_schema()` (Pydantic v2 default behavior); the schema is JSON-Schema-draft-2020-12 compatible.
- AC-4: A golden-schema test asserts the schema shape is stable (i.e., breaking schema changes are caught before merge).
- AC-5: `StateSnapshot` carries the following top-level fields with the documented types: `framework_version: str`, `session_id: str`, `workflow_id: str | None`, `topology_digest: str`, `created_at: datetime`, `paused_at: datetime`, `branches: dict[str, BranchState]`, `barriers: dict[str, BarrierState]`, `convergence_barriers: dict[str, str]`, `runnable: list[str]`, `fire_queue: list[str]`, `root_barrier_id: str | None`, `workflow_error: str | None`, `completed_emitted: list[str]`, `user_interactions: list[UserInteractionState]`, `user_interaction_inflight: bool`.
- AC-6: `BranchState` mirrors every field of the live `Branch` (including `id`, `current_agent`, `status`, `delivery_target`, `input`, `memory`, `waiting_on`, `candidate_of`, `parent_spawn`, `step_count`, `created_at`, `last_invoked_agent`, `consecutive_content_only`); `BranchState.status` accepts only the documented enum values `"RUNNING" | "WAITING" | "TERMINATED" | "FAILED" | "ABANDONED"`.
- AC-7: `BarrierState` mirrors every field of the live `Barrier` (including `id`, `policy`, `status`, `resolver_branch`, `resolver_agent`, `rendezvous_node`, `candidates`, `arrived`, `failed`, `upstream`, `downstream`, `created_at`, `metadata`); `BarrierState.status` accepts only `"OPEN" | "FIRED" | "CANCELLED"`. [added 2026-05-09]
- AC-8: `BarrierState.candidates`, `upstream`, and `downstream` round-trip as JSON arrays of strings (the live orchestrator stores them as `set[str]`); the wire shape is `list[str]`. [added 2026-05-09]
- AC-9: `BarrierState.arrived` accepts JSON-safe values; if the live orchestrator's `Barrier.arrived` contains a non-JSON-serializable value at pause time, `pause_session` raises a clear error (e.g., a `SnapshotSerializationError` or equivalent) — silent corruption is not allowed. [added 2026-05-09]
- AC-10: `UserInteractionState` carries `suspended_branch_id: str`, `prompt: Any`, `resume_agent: str` (one queued user interaction).
- AC-11: `PausedSessionMetadata` carries `session_id`, `workflow_id: str | None`, `paused_at: datetime`, `framework_version: str`, `snapshot_size_bytes: int` and is the return-element type of `Orchestra.list_paused_sessions()`.

### `Orchestrator` design seams (TRUNK-CRITICAL, public)

- AC-12: `Orchestrator.snapshot() -> OrchestratorState` is a public method that returns a deep-copy of all mutable orchestrator state (`branches`, `barriers`, `convergence_barriers`, `runnable`, `_fire_queue`, `root_barrier_id`, `_workflow_error`, `_completed_emitted`, `_user_interactions`, `_user_interaction_inflight`).
- AC-13: `Orchestrator.snapshot()` returned `Branch` and `Barrier` instances are NOT shared with the live orchestrator: mutating the live orchestrator after `snapshot()` does not mutate the snapshot, and vice versa.
- AC-14: `Orchestrator.restore_from(state: OrchestratorState) -> None` is a public method that replaces all mutable state on a freshly-constructed `Orchestrator` such that `branches`, `barriers`, `convergence_barriers`, `runnable`, `_fire_queue`, `root_barrier_id`, `_workflow_error`, `_completed_emitted`, `_user_interactions`, and `_user_interaction_inflight` match `state` after the call.
- AC-15: `Orchestrator.restore_from(...)` rebuilds `_resume_user_responses` (which is `asyncio.Queue`) fresh; pending user interactions are carried via the snapshot's `user_interactions` field, not via the queue.
- AC-16: `Orchestrator.quiesce() -> None` is an `async` public method that sets a pause flag, awaits all in-flight branch ticks (the dispatch loop's tracked `asyncio.Task`s) to complete, then returns; after `await quiesce()`, calling `snapshot()` is safe.
- AC-17: `Orchestrator.quiesce()` is idempotent: calling `await quiesce()` again on an already-quiesced orchestrator returns without error and produces no additional side effects (no-op).
- AC-18: `Orchestrator.quiesce()` exits the run-loop without setting `_workflow_error` (a quiesced orchestrator is not an errored orchestrator).
- AC-19: `Orchestrator.resume() -> WorkflowResult` is an `async` public method that runs the dispatch loop without calling `init_workflow` (state is already restored); it returns a `WorkflowResult` matching the shape returned by `Orchestrator.run()`.
- AC-20: `Orchestrator.resume()` exits on the same conditions as `Orchestrator.run()` (no `runnable`, no in-flight tasks, no pending user-interactions). [added 2026-05-09]
- AC-21: `Orchestrator.snapshot`, `Orchestrator.restore_from`, `Orchestrator.quiesce`, and `Orchestrator.resume` are reentrant — they may be called repeatedly across the orchestrator's lifecycle, supporting the future auto-checkpoint use case (this is the "design seam" guarantee). [added 2026-05-09]
- AC-22: `Orchestrator.run()`'s public signature is preserved (the loop-body extraction into a shared `_dispatch_loop()` helper is internal and not observable).

### `Orchestra` public API — pause/resume

- AC-23: `Orchestra.pause_session(session_id: str) -> None` is async; on success it awaits the live orchestrator's `quiesce()`, takes a snapshot, and writes it atomically via the configured `StorageBackend`.
- AC-24: `Orchestra.pause_session(...)` raises `SessionNotFoundError` when `session_id` is not currently in `Orchestra._active_orchestrators` (i.e., no live run with that id).
- AC-25: `Orchestra.pause_session(...)` is idempotent: calling it twice on the same session returns without error and writes the snapshot at most once (or writes identical content the second time); the second call logs a no-op message.
- AC-26: After `pause_session`, the awaited `Orchestra.execute()` call returns an `OrchestraResult` flagged as paused: `metadata["paused"] is True`, `success is False`, `error is None`.
- AC-27: `Orchestra.resume_session(session_id: str) -> OrchestraResult` is async; it reads the snapshot for `session_id`, verifies `framework_version`, reconstructs a fresh `Orchestrator` (and supporting components), replays state via `Orchestrator.restore_from`, rebuilds the standard listener set, resumes dispatch, and returns the final `OrchestraResult` (matching the shape of `Orchestra.run`).
- AC-28: `Orchestra.resume_session(...)` raises `IncompatibleSnapshotError` when the snapshot's `framework_version` differs from the running framework version (exact-string match in v0.3); the error message contains both versions and a clear instruction (e.g., re-run from scratch or downgrade).
- AC-29: `Orchestra.resume_session(...)` does NOT return events via the return value — events flow via the existing `EventBus` → SSE pathway. The return value is solely the final `OrchestraResult`.
- AC-30: `Orchestra.list_paused_sessions() -> list[PausedSessionMetadata]` returns metadata for every paused snapshot without loading the full snapshot bodies eagerly (verified by spying on the storage backend's read calls — no full-body reads triggered by enumeration).
- AC-31: `Orchestra.discard_paused_session(session_id: str) -> None` deletes one snapshot identified by `session_id`; subsequent `list_paused_sessions()` calls do not include it.

### `Orchestra` constructor & lifecycle wiring

- AC-32: `Orchestra.__init__` accepts `storage_backend: StorageBackend | None = None` (default constructs a `FileStorageBackend` rooted at the framework's standard data directory).
- AC-33: `Orchestra.__init__` accepts `snapshot_retention: timedelta = timedelta(days=30)`.
- AC-34: `Orchestra.__init__` triggers `storage_backend.expire_older_than(snapshot_retention)` once during construction (the periodic snapshot sweeper). [added 2026-05-09]
- AC-35: `Orchestra._active_orchestrators` is a `dict[str, Orchestrator]` populated when `Orchestra.execute()` constructs a new orchestrator and popped in the `execute()` `finally` block — so `pause_session(session_id)` can find a live orchestrator while `execute()` is awaiting, and the entry is cleared on exit. [added 2026-05-09]
- AC-36: `Orchestra._wire_event_bus()` is a method called both from `Orchestra.__init__` and from `resume_session` to attach the standard listener set (`StatusManager`, `TraceCollector`, registered `TelemetrySink` instances) onto the `EventBus`. [added 2026-05-09]

### Removed public surface (observable)

- AC-37: `StateManager` is no longer importable from `marsys.coordination` or `marsys.coordination.state` (the import site that previously worked now raises `ImportError` / `AttributeError`).
- AC-38: `CheckpointManager` is no longer importable from `marsys.coordination` or `marsys.coordination.state`.
- AC-39: `Orchestra.create_checkpoint` and `Orchestra.restore_checkpoint` no longer exist as public methods — calling them on an `Orchestra` instance raises `AttributeError`.
- AC-40: `Orchestra.__init__` no longer accepts a `state_manager` keyword argument (callers passing it get a `TypeError`).
- AC-41: `Orchestra.run` classmethod no longer accepts a `state_manager` keyword argument.

### `StorageBackend` Protocol & `FileStorageBackend`

- AC-42: `StorageBackend` is a Python `Protocol` (not an ABC) importable from `marsys.coordination.state` with async methods `read(key) -> bytes`, `write(key, data) -> None`, `delete(key) -> None`, `list_with_metadata() -> list[StorageEntry]`, `expire_older_than(age: timedelta) -> int` (returns count deleted).
- AC-43: `FileStorageBackend(root: Path)` ships in the framework, satisfies the `StorageBackend` Protocol, and is importable from `marsys.coordination.state`.
- AC-44: `FileStorageBackend.write(key, data)` is atomic: writes go to a temp path, then `fsync` the file fd, then `os.replace` to the target, then `fsync(parent_dir_fd)` (POSIX; no-op on Windows).
- AC-45: A simulated mid-write crash (write fd closed between `fsync` and `rename`, then process killed) leaves the prior snapshot intact — no torn or partially-written file is observable at the target key.
- AC-46: `FileStorageBackend.expire_older_than(age)` deletes entries older than `age`, leaves newer entries, and returns the count of deleted entries.
- AC-47: The default snapshot path under `FileStorageBackend` is `<storage_backend_root>/<session_id>/snapshot.json`.
- AC-48: `StorageEntry` carries `key: str`, `size_bytes: int`, `modified_at: datetime` and is the element type of `StorageBackend.list_with_metadata()`.

### Errors

- AC-49: `IncompatibleSnapshotError` is importable from `marsys.coordination.state` and is the type raised by `resume_session` on `framework_version` mismatch.
- AC-50: `SessionNotFoundError` (or equivalent named exception) is the type raised by `pause_session` when `session_id` is not in `_active_orchestrators`. [needs clarification — the plan names this error `SessionNotFoundError` in AC text but does not pin its module path; auditor should accept whatever module the implementer chose, but the type name is `SessionNotFoundError`]
- AC-51: `IncompatibleSnapshotError`'s message contains both `framework_version` values from the snapshot and the running framework, formatted as documented (e.g., the message containing both versions and an instruction).

### Pause/resume correctness

- AC-52: A pause-then-resume run produces the same `OrchestraResult.success` as the non-paused baseline running the same workload (semantic-equivalence; not byte-equivalence).
- AC-53: A pause-then-resume run produces the same `final_response` shape as the non-paused baseline — same set of top-level keys (when dict) and same per-key value types; or the same scalar shape when not a dict.
- AC-54: A pause-then-resume run produces the same per-barrier arrival counts as the non-paused baseline.
- AC-55: Pause/resume works across two separate Python processes: process A starts a run, pauses, and exits; process B starts a fresh `Orchestra`, calls `resume_session`, and the run completes successfully.
- AC-56: Listener restoration works: after `resume_session` returns (or once dispatch starts), the standard event types have non-zero listener counts (`event_bus.get_listener_count(...) > 0`) for `StatusManager`, `TraceCollector`, and any registered `TelemetrySink` listeners.
- AC-57: Custom listeners attached by callers via `EventBus.subscribe` BEFORE the original run are NOT restored on resume — the contract is documented in `resume_session`'s docstring; callers must re-attach custom listeners before invoking `resume_session`. [added 2026-05-09]

### Non-functional / contract guarantees

- AC-58: `framework_version` comparison in v0.3 is exact-string match — patch-level differences (e.g., `0.3.1` vs `0.3.2`) reject with `IncompatibleSnapshotError`.
- AC-59: At-least-once tool-call contract on resume: a resumed workflow may re-execute tool calls that were in-flight at pause time (mid-tick state is not captured); this is the documented contract.
- AC-60: Scheduled events (e.g., `Barrier.policy.timeout` deadlines) whose `fire_at` was crossed during the paused interval are skipped on resume and logged at `WARNING` level — they do not fire on resume.
- AC-61: No imports from `spren` exist in the framework: `grep -rn 'from spren\|import spren' packages/framework/` returns zero matches (SP-018 invariant).

### Regression suite delta

- AC-62: `tests/coordination/test_state_manager_integration.py` is deleted; the regression-suite count drops by approximately the number of tests in that file (~30) and gains the tests added by `tests/coordination/state/test_snapshot.py`, `tests/coordination/state/test_storage.py`, and `tests/integration/test_pause_resume.py`. [added 2026-05-09]
- AC-63: The framework regression suite passes at the new total (post-delete + post-additions); no new skips silently introduced. [added 2026-05-09]

## Out of scope

- Continuous auto-checkpoint while running (in-flight crash recovery for non-paused runs) — follow-up framework session uses the seams shipped here.
- Snapshot migration tooling for cross-version resume (beyond rejecting with `IncompatibleSnapshotError`).
- An `is_idempotent: bool` field on the framework's tool registry — separate small follow-up PR.
- A `S3StorageBackend` or `GCSStorageBackend` — ships later in MARSYS Cloud's PR.
- Removing the legacy `coordination/branches/types.py` module — it is still imported by 7 other modules and stays out of this PR's scope.
- Bumping `marsys.__version__` from `0.2.1` to `0.3.0` — belongs to a separate release-prep PR; the version-mismatch test is portable across actual `__version__` values.
- `MockRuntime` / the simulator at `research/orchestration/simulator/` — tests must NOT use it; the live `DeterministicRuntime` at `coordination/execution/deterministic_runtime.py` is the test driver.
- Continuous (background) snapshot retention sweeper beyond the single sweep on `Orchestra.__init__`.
- Format-pluggable snapshot serializers (JSON only in v0.3).
- A `Restore` path that translates legacy (pre-v0.3) `StateSnapshot` shapes — legacy snapshots are rejected via `IncompatibleSnapshotError`.

## Open / needs clarification

- AC-50: The plan names `SessionNotFoundError` in the acceptance text but does not specify its module path; the auditor should verify by name only and accept whichever module the implementer chose.
