# Framework Session 03: Pause/Resume Completion in `state_manager`

> **Tier**: CRITICAL. The implementation reaches into TRUNK-CRITICAL public surface (`Orchestra.pause_session` / `resume_session`) and reads/restores TRUNK-CRITICAL internal state (`Orchestrator.branches`, `Orchestrator.barriers`, `Orchestrator.runnable`, `Orchestrator._fire_queue`, `Orchestrator._user_interactions`). An ADR is mandatory before any code is written — see the **Pre-flight escalation gate** below.

Required by Spren v0.4 — specifically Spren session **v0.4-29 (workflow pause / resume Spren-side)**, which adds `POST /v1/runs/{id}/pause` and `POST /v1/runs/{id}/resume` endpoints, run inspector UI controls, and meta-agent `pause_run` / `resume_run` tools. This framework PR provides the underlying primitive; v0.4-29 wires it into the Spren product.

---

## Working rules — how we collaborate (READ FIRST)

You are a peer on this project. You are NOT an order-taker. You share equal voice and equal responsibility for the success of the marsys framework.

### Be a peer with equal voice

- **Push back when you disagree.** If this brief is wrong, or if a "best practice" cited here is outdated, or if a structural choice will cause us pain later, say so. Defend with evidence.
- **Stay engaged.** Comment in this session file as you go; flag concerns before they become problems.
- **Be proactive.** If you see something this session is missing, raise it. If a Spren-side assumption embedded here doesn't hold from the framework's perspective, push back loudly.

### Take responsibility

- **Ownership is shared.** If something fails, it's our shared failure.
- **You own correctness.** Manually verify acceptance criteria, not just unit tests.
- **You own follow-through.** Update this file's "What was actually built". Update the corresponding `v0.4-spren-support.md` checkbox. Add "Lessons / Surprises" if anything surprised you.

### Double-check before any decision

- **Read the framework code before changing it.** Don't assume; verify.
- **Verify file paths and symbols still exist** before referencing them.
- **Run framework tests after every meaningful change**, not just at end.
- **Use git commits as checkpoints.**

### Critically assess the plan itself

This brief was written from the Spren-side perspective. **It might miss framework-internal constraints.**

- **Read the actual framework code** for any module the brief references.
- **Spawn an independent verification sub-agent** when material doubt exists.
- **Run online research** for any "best practice" claim (atomic-write semantics on Linux/macOS/Windows, JSON Schema portability, Pydantic round-trip patterns).
- **Cite sources when you challenge the plan.**

### Ask the framework team when blocked on intent

Strategic, opinionated, or architectural decisions belong to the framework team. Use `AskUserQuestion` for:

- **Architectural:** "The brief assumes we add `Orchestrator.snapshot()` / `Orchestrator.restore()`. Should this live on `Orchestrator` itself, or on a new `OrchestratorStateAdapter` to keep the orchestrator class slim?"
- **Naming / API surface:** "The brief proposes `StateSnapshot` for the Pydantic model. Is there a framework convention I should follow?"
- **Scope expansion:** "Implementing this requires also touching `RealRuntime` to expose its session_id getter. Should I expand or split?"

Do NOT ask for purely technical implementation choices you can decide.

### Multi-consumer justification (every feature, no exceptions)

Before writing code, you must be able to name the OTHER plausible consumers of this primitive beyond Spren. This brief lists four — verify they hold from the framework's perspective.

### Build the smallest thing that works first

- Acceptance criteria first. Iterate within scope only after they're green.
- Don't add migration tooling for snapshot-version mismatch in this PR. Future framework PR can add it when needed.

### Don't expand scope silently

- Frame's PR review will catch silent scope expansion; don't waste reviewer cycles.

---

## Pre-flight escalation gate (MANDATORY for this brief)

This PR modifies TRUNK-CRITICAL public methods on `Orchestra` (`pause_session`, `resume_session`) and reads/writes TRUNK-CRITICAL internal state on `Orchestrator`. Per the framework's `CLAUDE.md`, **non-additive changes to TRUNK-CRITICAL components require an ADR before any code is written.**

The gate is unconditional for this brief:

1. **Read end-to-end** before drafting the ADR:
   - `packages/framework/src/marsys/coordination/orchestra.py` (whole file — note the current shape of `Orchestra.__init__`, the `_initialize_components` wiring of `StatusManager` + `TraceCollector`, the existing `pause_session` body at lines 999–1032, and the existing `resume_session` TODO stub at lines 1034–1079)
   - `packages/framework/src/marsys/coordination/execution/orchestrator.py` (whole file — note `Orchestrator.__init__` field assignments at lines 102–121: `branches`, `barriers`, `convergence_barriers`, `runnable`, `_fire_queue`, `root_barrier_id`, `_workflow_error`, `_completed_emitted`, `_user_interactions`, `_user_interaction_inflight`, `_resume_user_responses`)
   - `packages/framework/src/marsys/coordination/execution/orchestrator_types.py` (whole file — note `Branch` at lines 95–112 and `Barrier` at lines 130–151, including `Barrier.arrived: dict[str, Any]` at line 146)
   - `packages/framework/src/marsys/coordination/state/state_manager.py` (whole file — the reusable storage primitives plus the legacy `StateSnapshot` shape this PR replaces)
   - `packages/framework/src/marsys/coordination/event_bus.py` (whole file — listener model is `Dict[str, List[Callable]]`; not picklable)

2. **Draft the ADR** (file under `docs/architecture/framework/adr/NNN-pause-resume-snapshot.md`, numbered against the next free ADR number) covering:
   - The minimum touch surface in TRUNK-CRITICAL files (a precise list of which methods you will add and which existing methods you will rewrite, line by line).
   - Why each change is necessary — i.e., what couldn't be done from outside the class.
   - The snapshot shape (the Pydantic models in this brief's "Load-bearing shapes" section), and the contract for `Orchestrator.snapshot()` / `Orchestrator.restore_from(snapshot)`.
   - The semantic-equivalence definition for the resume-correctness test (NOT byte-equivalence — see "Determinism guarantees" below).
   - The at-least-once contract for in-flight tool calls at pause boundaries (see "In-flight tool-call retry semantics").
   - Rejected alternatives: pickle-based snapshot, MessagePack-based snapshot, cancel-and-rollback semantics.

3. **Get framework-team approval on the ADR**, in writing, in the ADR document itself, before opening a PR with code changes.

If you skip the ADR and start coding, the PR will be sent back unread.

---

## The big picture — what this feature is and why

### Why the framework has features that come from Spren-side planning

The marsys framework serves multiple consumers: Spren (this OSS umbrella's meta-agent), MARSYS Cloud (proprietary hosted control plane), MARSYS Studio (proprietary hosted UI), third-party Python users, and observability backends. When one consumer (Spren) needs a framework capability that doesn't yet exist, the resulting framework feature must be **multi-consumer-justifiable** — usable by other consumers, not Spren-specific.

### Multi-consumer (mandatory)

- **Spren** uses pause/resume for daemon-restart resilience (in-flight runs survive a Spren daemon crash + restart) and for user-facing pause/resume in v0.4-29. Spren's REST endpoints, run inspector UI, and meta-agent tools all call into `Orchestra.pause_session` / `resume_session`.
- **MARSYS Cloud** uses pause/resume for managed long-running workflows that span node restarts or autoscaling events. Cloud will plug an `S3StorageBackend` into the same `StorageBackend` Protocol this PR ships; that backend ships in a Cloud-side PR, not here.
- **CI integrations** that run long workflows across multiple jobs use pause/resume for state handoff: process A pauses to an artifact store, process B downloads and resumes.
- **Framework users running locally** gain crash resilience for hours-long workflows via the file backend.

### What MARSYS Spren is (consumer context)

MARSYS Spren is the open-source umbrella product on top of the marsys framework. It contains a continuously-active personal AI assistant (the meta-agent), a visual workflow builder, a Tauri desktop shell, a Textual TUI, and a Python adapter SDK. It consumes the framework via three doors (per Spren's SP-018):

1. **`Orchestra.run(topology, task) → OrchestraResult`** — finite workflow execution
2. **`EventBus.subscribe(event_type, listener)`** — in-process workflow lifecycle events
3. **`TelemetrySink` protocol** — generic observability hook (added by Session 02)

This session adds two methods to door (1): `Orchestra.pause_session()` and `Orchestra.resume_session()` are full implementations replacing the placeholder bodies. Plus two new helpers — `Orchestra.list_paused_sessions()` and `Orchestra.discard_paused_session()` — for daemon-startup discovery and explicit cleanup.

### Your role as a framework implementer

1. Honor the framework's architecture
2. Honor the framework's design principles (DP-001..DP-007)
3. Honor multi-consumer justification (no Spren-only paths; no `from spren` imports — SP-018)
4. Ship a working PR — never half-finished
5. Write all required tests
6. Push back when something is wrong

### Where to read deeper if you need it

- Framework's own architecture docs (in the framework worktree): `docs/architecture/framework/`
- Framework's own `CLAUDE.md`
- Spren's view of this feature: [`../../v0.4-spren-support.md`](../../v0.4-spren-support.md), [`../../../spren/v0.4-extensions.md`](../../../spren/v0.4-extensions.md) Phase Z (v0.4-29)
- API design (consumer expectations): [`../../../../architecture/spren/03-api-design.md`](../../../../architecture/spren/03-api-design.md) §"Pause / resume semantics (v0.4)" lines 174–185
- Snapshot path (consumer): [`../../../../architecture/spren/02-data-model.md`](../../../../architecture/spren/02-data-model.md) line 20 (Spren maps framework's `session_id` to `run_id` in `<data-dir>/data/runs/{run_id}/snapshot.json`)
- Affected framework modules: see "Pre-flight escalation gate" above

---

## What came before this session

**Previous framework PRs from this dir:** Sessions 01 + 02 (NDJSON streaming tracing writer; TelemetrySink Protocol). Independent of this PR's surface — listener wiring touches `TelemetrySink`-style listeners but does not require them to land first.

**State at start of this session:**

- `Orchestra` (in `coordination/orchestra.py`) has a `pause_session(session_id)` method (lines 999–1032) that writes a snapshot in the legacy `ExecutionBranch` shape, and a `resume_session(session_id)` method (lines 1034–1079) that returns a placeholder `OrchestraResult` with `final_response="Session resumed (implementation pending)"` and a `# TODO: Implement proper state restoration and continuation` marker at line 1067.
- `Orchestrator` (in `coordination/execution/orchestrator.py`, lines 79–121) is the live unified-barrier execution loop. Its mutable state — what a snapshot must capture — lives at lines 102–121: `branches: dict[str, Branch]`, `barriers: dict[str, Barrier]`, `convergence_barriers: dict[str, str]`, `runnable: collections.deque[str]`, `_fire_queue: list[str]`, `root_barrier_id: Optional[str]`, `_workflow_error: Optional[str]`, `_completed_emitted: set[str]`, `_user_interactions: collections.deque`, `_user_interaction_inflight: bool`, `_resume_user_responses: Optional[asyncio.Queue]`.
- `Branch` and `Barrier` (in `coordination/execution/orchestrator_types.py`, lines 95–112 and 130–151) are the live data classes. `Barrier.arrived: dict[str, Any]` at line 146 carries arbitrary Python values delivered by branches — this is the load-bearing serialization complexity of this feature.
- `coordination/state/state_manager.py` provides reusable storage primitives that this PR keeps and extends: the `StorageBackend` ABC (lines 76–102) and `FileStorageBackend` (lines 105–191). The legacy `StateSnapshot` dataclass (lines 41–73) and the `StateManager._serialize_branches` / `_deserialize_branches` helpers (lines 465–542) target the dead `ExecutionBranch` shape from `coordination/branches/types.py` and are replaced wholesale by this PR.
- `EventBus` (in `coordination/event_bus.py` line 27): `listeners: Dict[str, List[Callable]]`. Listeners are typically bound methods of `StatusManager`, `TraceCollector`, and any registered `TelemetrySink` instances. Not serializable; resume rebuilds the standard listener set by re-running the same wiring `Orchestra.__init__` runs.

**Verify state with:**
```bash
cd /home/rezaho/research_projects/marsys-spren-work/packages/framework/
source ../../.venv/bin/activate
pytest tests/ -x --tb=short                                            # baseline counts
git log --oneline -20 src/marsys/coordination/state/
git log --oneline -20 src/marsys/coordination/orchestra.py
git log --oneline -20 src/marsys/coordination/execution/orchestrator.py
grep -n 'pause_session\|resume_session\|TODO.*pause\|TODO.*resume' src/marsys/coordination/orchestra.py
grep -n 'class StateSnapshot\|class StateManager\|class StorageBackend\|class FileStorageBackend' src/marsys/coordination/state/state_manager.py
grep -rn 'from .branches\|from \.\.branches\|coordination.branches' src/marsys/  # current legacy footprint
```

The legacy `coordination/branches/types.py` is **still imported** by `coordination/routing/router.py`, `coordination/validation/response_validator.py`, `coordination/execution/step_executor.py`, `coordination/execution/real_runtime.py`, `coordination/rules/rules_engine.py`, `coordination/communication/user_node_handler.py`, and `coordination/__init__.py`. Removing it is **out of scope for this PR**. Scope here is to drop the legacy `StateSnapshot` shape from `coordination/state/state_manager.py` and replace it with the new shape; the legacy types module is left in place for the modules that still import it.

---

## What this session ships

After merge:

- A canonical `StateSnapshot` Pydantic model capturing every field of `Orchestrator.__init__` mutable state (per orchestrator.py:102–121), plus `framework_version`, `session_id`, `topology_digest`, `created_at`, `branches: dict[str, BranchState]`, `barriers: dict[str, BarrierState]`, `runnable: list[str]`, `fire_queue: list[str]`, `convergence_barriers`, `root_barrier_id`, `workflow_error`, `completed_emitted`, `user_interactions: list[UserInteractionState]`, `user_interaction_inflight: bool`.
- A `StorageBackend` Protocol with methods `read(key) -> bytes`, `write(key, bytes)`, `delete(key)`, `list_with_metadata() -> list[StorageEntry]`, `expire_older_than(timedelta)`. The existing ABC is rewritten as a Protocol; the existing `FileStorageBackend` is rewritten against the new Protocol with atomic write semantics (write-temp + fsync(fd) + os.replace + fsync(parent_dir_fd)).
- `Orchestrator.snapshot() -> OrchestratorState` and `Orchestrator.restore_from(state: OrchestratorState) -> None` — new TRUNK-CRITICAL public methods. ADR-gated.
- `Orchestra.pause_session(session_id) -> None` rewritten: cleanly halts the run after the in-flight branch tick completes; serializes `Orchestrator.snapshot()` to a `StateSnapshot`; writes the snapshot atomically via the configured `StorageBackend`. Idempotent.
- `Orchestra.resume_session(session_id) -> AsyncIterator[Event]` rewritten: reads the snapshot; verifies `framework_version`; reconstructs the `Orchestrator` afresh; replays state via `Orchestrator.restore_from`; rebuilds the standard listener set by calling the extracted `Orchestra._wire_event_bus()`; resumes dispatch.
- `Orchestra.list_paused_sessions() -> list[PausedSessionMetadata]` and `Orchestra.discard_paused_session(session_id) -> None` — discovery and explicit-cleanup APIs.
- The `Orchestra` constructor accepts `storage_backend: StorageBackend | None = None`. Default: `FileStorageBackend(<framework_data_dir>)` where the framework data directory follows the existing `state_manager` convention.
- A periodic snapshot sweeper runs once on `Orchestra.__init__` and calls `storage_backend.expire_older_than(timedelta(days=30))`. The 30-day default is configurable via `Orchestra(snapshot_retention=...)` parameter.
- The legacy `StateSnapshot` dataclass + `_serialize_branches` / `_deserialize_branches` helpers in `state_manager.py` are deleted.
- Framework regression suite green at the same counts as baseline.
- Tests cover: round-trip JSON schema, atomic-write under simulated mid-write crash, semantic-equivalence pause-then-resume vs baseline, cross-process pause/resume via subprocess fixture, framework-version mismatch error.

### Acceptance criteria

- [ ] ADR for TRUNK-CRITICAL touch points filed and approved (linked in the PR description)
- [ ] `StateSnapshot` Pydantic model defined; round-trips via JSON Schema 2020-12
- [ ] `Orchestrator.snapshot() -> OrchestratorState` returns a value sufficient to fully reconstruct `Orchestrator` mutable state
- [ ] `Orchestrator.restore_from(state)` reconstructs all of `branches`, `barriers`, `convergence_barriers`, `runnable`, `_fire_queue`, `root_barrier_id`, `_workflow_error`, `_completed_emitted`, `_user_interactions`, `_user_interaction_inflight`. (`_resume_user_responses: asyncio.Queue` is rebuilt fresh on resume; pending user interactions ride in `_user_interactions`.)
- [ ] `Orchestra.pause_session(session_id) -> None` writes the snapshot atomically; idempotent (calling twice has no extra effect; second call is a no-op log line)
- [ ] `Orchestra.resume_session(session_id) -> AsyncIterator[Event]` reconstructs orchestrator state + listener wiring + continues dispatch
- [ ] `Orchestra.list_paused_sessions() -> list[PausedSessionMetadata]` returns metadata for all paused snapshots without loading the full snapshot bodies
- [ ] `Orchestra.discard_paused_session(session_id) -> None` deletes one snapshot
- [ ] Atomic-write failure tested: simulated crash mid-write leaves the prior snapshot intact (no torn writes); `os.replace` + `fsync(parent_dir)` are exercised
- [ ] Pause-then-resume produces semantically-equivalent final state to a non-paused baseline (deterministic test workload via the simulator at `packages/framework/research/orchestration/simulator/`)
- [ ] Pause + resume across two separate Python processes (subprocess fixture in tests)
- [ ] Snapshot version mismatch (`framework_version` differs) raises `IncompatibleSnapshotError` with a clear message
- [ ] Snapshot retention sweeper exists and is invoked from `Orchestra.__init__`
- [ ] Framework regression suite green (same test counts as baseline; no new skips)
- [ ] **Multi-consumer justification documented in PR description**: explicit list of consumers (Spren v0.4-29; MARSYS Cloud's future `S3StorageBackend`; CI integrations; framework local users)
- [ ] **No Spren type imported into framework** — `grep -rn 'from spren\|import spren' packages/framework/` returns zero matches (SP-018)
- [ ] Spren-side coordination: confirm [`../../../spren/v0.4-extensions.md`](../../../spren/v0.4-extensions.md) v0.4-29 row matches this PR's actual API surface (`list_paused_sessions`, `discard_paused_session`, the `framework_version` mismatch error path); amend the v0.4-29 row if any drift exists
- [ ] Framework architecture docs updated where applicable (the new ADR is the primary doc; if a `coordination/state/` overview exists, update it; if not, do not create one)
- [ ] CHANGELOG entry added
- [ ] PR description references this session brief
- [ ] PR merges to framework's main; tagged framework release noted in this session file's "What was actually built"

---

## Background reading (do this before writing code)

Don't skim. Read these end-to-end.

1. The framework's `CLAUDE.md` — TRUNK-CRITICAL list, design principles
2. [`../../v0.4-spren-support.md`](../../v0.4-spren-support.md) — the row for Session 03
3. [`../../../spren/v0.4-extensions.md`](../../../spren/v0.4-extensions.md) Phase Z (v0.4-29) — Spren-side consumer
4. [`../../../../architecture/spren/03-api-design.md`](../../../../architecture/spren/03-api-design.md) §"Pause / resume semantics (v0.4)" lines 174–185 — REST contract Spren expects
5. [`../../../../architecture/spren/02-data-model.md`](../../../../architecture/spren/02-data-model.md) line 20 — Spren snapshot path
6. [`../../../../architecture/spren/08-design-principles.md`](../../../../architecture/spren/08-design-principles.md) — SP-018 (framework purity), SP-019 (API as truth)
7. The framework files listed in the **Pre-flight escalation gate** — every line, not just the section being touched
8. Atomic-write semantics: `man fsync(2)`, `man rename(2)`, Python `os.replace` docs (POSIX `rename` semantics on Linux/macOS; Windows `MoveFileEx` semantics for `os.replace`). Cite the source of any "best practice" claim you make in the ADR.

**Verify before proceeding:**
- Capture baseline test counts BEFORE any change
- `git log --oneline -20` for each affected path; confirm no in-flight commits land between the read and the start of code
- Read referenced framework files end-to-end for any module you're integrating with
- Confirm `coordination/branches/types.py` is still imported by the seven other modules listed under "What came before this session" — if any of those import paths have moved, **update the brief and re-confirm scope** before writing code

---

## Detailed plan

### Files to create

- `packages/framework/src/marsys/coordination/state/snapshot.py` — `StateSnapshot`, `BranchState`, `BarrierState`, `UserInteractionState`, `PausedSessionMetadata` Pydantic models. The single source of truth for the on-disk shape.
- `packages/framework/src/marsys/coordination/state/storage.py` — the new `StorageBackend` Protocol, `FileStorageBackend` concrete implementation, `StorageEntry` dataclass for `list_with_metadata`. Atomic-write helper lives here.
- `packages/framework/src/marsys/coordination/state/errors.py` — `IncompatibleSnapshotError`, `SnapshotCorruptionError`, `SnapshotNotFoundError`. (May be folded into `agents/exceptions.py` if the framework prefers a single exceptions file — flag in the ADR.)
- `docs/architecture/framework/adr/NNN-pause-resume-snapshot.md` — the ADR. Filed before code.
- `packages/framework/tests/coordination/state/test_snapshot.py` — round-trip + version-mismatch unit tests
- `packages/framework/tests/coordination/state/test_storage.py` — atomic-write + retention-sweeper unit tests
- `packages/framework/tests/integration/test_pause_resume.py` — semantic-equivalence + cross-process pause/resume (subprocess fixture). Uses the simulator at `packages/framework/research/orchestration/simulator/`.

### Files to modify

- `packages/framework/src/marsys/coordination/state/state_manager.py` — delete the legacy `StateSnapshot` dataclass (lines 41–73), `_serialize_branches` / `_deserialize_branches` (lines 465–542), `_serialize_results` / `_deserialize_results` (lines 568–622), and the `StateManager.save_session` / `load_session` / `pause_execution` / `resume_execution` methods that target the legacy shape. Keep the file as the home of `CheckpointManager` (if used elsewhere) and any other still-live utilities — verify with grep before deletion. The new snapshot path bypasses `StateManager` entirely; `Orchestra` calls the `StorageBackend` directly.
- `packages/framework/src/marsys/coordination/execution/orchestrator.py` — add `Orchestrator.snapshot() -> OrchestratorState` and `Orchestrator.restore_from(state: OrchestratorState) -> None`. ADR-gated. These are the only TRUNK-CRITICAL non-additive touches in this file. (`OrchestratorState` is a dataclass internal to the orchestrator — distinct from the on-disk `StateSnapshot` Pydantic model. The Orchestra layer maps between them.)
- `packages/framework/src/marsys/coordination/orchestra.py`:
  - Extract listener-wiring from `_initialize_components` into a new `_wire_event_bus()` method, callable from both `__init__` and `resume_session`. (Listener wiring spans `StatusManager` setup at lines ~178–203 and `TraceCollector` setup at lines ~206–217; the extraction is mechanical.)
  - Add `storage_backend: StorageBackend | None = None` and `snapshot_retention: timedelta = timedelta(days=30)` parameters to `Orchestra.__init__`. Defaults preserve existing behavior (no breaking change for callers that don't pass them).
  - Trigger `storage_backend.expire_older_than(snapshot_retention)` once during `__init__`.
  - Rewrite `pause_session(session_id)` (lines 999–1032) to call `Orchestrator.snapshot()`, build a `StateSnapshot`, write atomically via the configured `StorageBackend`. Idempotent.
  - Rewrite `resume_session(session_id)` (lines 1034–1079) per the AsyncIterator contract above. Replace the placeholder `OrchestraResult` at lines 1069–1079.
  - Add `list_paused_sessions()` and `discard_paused_session(session_id)` methods.
- `packages/framework/CHANGELOG.md` — entry under the v0.3.x or v0.4.0 release this PR targets.

### Files NOT to touch

- TRUNK-CRITICAL beyond the additive surface: `coordination/execution/real_runtime.py`, `coordination/topology/graph.py`, `coordination/validation/response_validator.py` — must remain untouched. If you find that you cannot snapshot without reading something only `RealRuntime` exposes, **stop and escalate** — do not patch `RealRuntime`.
- Spren-side code under `packages/spren/`. `grep -rn 'from spren\|import spren' packages/framework/` must return zero matches at PR time.
- Anything outside `coordination/state/`, the additive surface on `coordination/orchestra.py` (only `__init__`, `pause_session`, `resume_session`, `list_paused_sessions`, `discard_paused_session`, the extracted `_wire_event_bus`), the additive surface on `coordination/execution/orchestrator.py` (only `snapshot` and `restore_from`), the test directories, the new `state/snapshot.py` / `state/storage.py` / `state/errors.py` files, and the new ADR.

### Load-bearing shapes

`StateSnapshot` and the `StorageBackend` Protocol are the contract.

```python
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol, AsyncIterator
from pydantic import BaseModel, Field


class BranchState(BaseModel):
    """Mirror of orchestrator_types.Branch (lines 95-112), JSON-safe."""
    id: str
    current_agent: str
    status: str  # one of "RUNNING" | "WAITING" | "TERMINATED" | "FAILED" | "ABANDONED"
    delivery_target: str
    input: Any = None
    memory: list[dict[str, Any]] = Field(default_factory=list)
    waiting_on: str | None = None
    candidate_of: list[str] = Field(default_factory=list)  # set serializes as list
    parent_spawn: str | None = None
    step_count: int = 0
    created_at: float
    last_invoked_agent: str | None = None
    consecutive_content_only: int = 0


class ConvergencePolicyState(BaseModel):
    """Mirror of ConvergencePolicy (orchestrator_types.py:115-127)."""
    min_ratio: float = 1.0
    on_insufficient: str = "fail"  # "fail" | "proceed" | "user"
    terminate_orphans: bool = True
    timeout: float | None = None


class BarrierState(BaseModel):
    """Mirror of orchestrator_types.Barrier (lines 130-151), JSON-safe."""
    id: str
    policy: ConvergencePolicyState
    status: str  # "OPEN" | "FIRED" | "CANCELLED"
    resolver_branch: str | None = None
    resolver_agent: str | None = None
    rendezvous_node: str | None = None
    candidates: list[str] = Field(default_factory=list)
    arrived: dict[str, Any] = Field(default_factory=dict)  # see "Barrier value contract" below
    failed: dict[str, str] = Field(default_factory=dict)
    upstream: list[str] = Field(default_factory=list)
    downstream: list[str] = Field(default_factory=list)
    created_at: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class UserInteractionState(BaseModel):
    """One queued user interaction. Orchestrator._user_interactions is a deque
    of (suspended_branch_id, prompt, resume_agent) tuples; this is the JSON-safe
    form."""
    suspended_branch_id: str
    prompt: Any
    resume_agent: str


class StateSnapshot(BaseModel):
    """The on-disk pause/resume snapshot. JSON-encoded.

    Versioned by `framework_version`. A snapshot whose `framework_version`
    differs from the running framework version is rejected on restore with
    `IncompatibleSnapshotError`. v0.4 ships no migration tooling; future
    framework PRs may add one when the shape changes.
    """
    framework_version: str
    session_id: str
    workflow_id: str | None = None
    topology_digest: str  # opaque hash of the frozen topology, for sanity checks
    created_at: datetime
    paused_at: datetime
    branches: dict[str, BranchState]
    barriers: dict[str, BarrierState]
    convergence_barriers: dict[str, str]
    runnable: list[str]
    fire_queue: list[str]
    root_barrier_id: str | None = None
    workflow_error: str | None = None
    completed_emitted: list[str]
    user_interactions: list[UserInteractionState]
    user_interaction_inflight: bool


class PausedSessionMetadata(BaseModel):
    """Lightweight metadata returned by Orchestra.list_paused_sessions().
    Does NOT contain the full snapshot body — only what's needed to render
    a list of paused runs."""
    session_id: str
    workflow_id: str | None
    paused_at: datetime
    framework_version: str
    snapshot_size_bytes: int


class StorageEntry(BaseModel):
    key: str
    size_bytes: int
    modified_at: datetime


class StorageBackend(Protocol):
    """Generic storage abstraction for snapshots. The framework ships
    FileStorageBackend; MARSYS Cloud and CI integrations supply their own
    (S3, GCS, GitHub Actions artifact store)."""

    async def read(self, key: str) -> bytes: ...
    async def write(self, key: str, data: bytes) -> None: ...  # MUST be atomic
    async def delete(self, key: str) -> None: ...
    async def list_with_metadata(self) -> list[StorageEntry]: ...
    async def expire_older_than(self, age: timedelta) -> int:
        """Delete entries older than `age`. Returns count deleted."""
        ...


class FileStorageBackend:
    """File-backed storage. Atomic writes via:
       1. open(target.tmp, 'wb')
       2. write data
       3. fsync(file_fd)
       4. close
       5. os.replace(target.tmp, target)        # atomic on POSIX + Windows
       6. fsync(parent_dir_fd)                  # POSIX-only; no-op on Windows
       (Source: man rename(2), man fsync(2), Python os.replace docs.)
    """
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
    # ... implements StorageBackend
```

`Orchestra` integration:

```python
class Orchestra:
    def __init__(
        self,
        agent_registry: AgentRegistry,
        rule_factory_config: RuleFactoryConfig | None = None,
        state_manager: StateManager | None = None,
        communication_manager: "CommunicationManager | None" = None,
        execution_config: "ExecutionConfig | None" = None,
        storage_backend: StorageBackend | None = None,        # new
        snapshot_retention: timedelta = timedelta(days=30),    # new
    ) -> None: ...

    async def pause_session(self, session_id: str) -> None:
        """Cleanly halt the run; write a snapshot atomically. Idempotent on a
        session that is already paused."""

    async def resume_session(self, session_id: str) -> AsyncIterator[Event]:
        """Read the snapshot for `session_id`; verify framework_version;
        rebuild Orchestrator + listener wiring; continue dispatch. Yields
        events as the resumed run emits them.

        NOTE: only the standard listener set is restored on resume
        (StatusManager, TraceCollector, registered TelemetrySink instances).
        Custom listeners attached via EventBus.subscribe by the caller are NOT
        restored; the caller must re-attach them before iterating the yielded
        events."""

    async def list_paused_sessions(self) -> list[PausedSessionMetadata]: ...

    async def discard_paused_session(self, session_id: str) -> None: ...

    def _wire_event_bus(self) -> None:
        """Wire StatusManager, TraceCollector, and any execution_config-driven
        listeners onto self.event_bus. Called from __init__ and from
        resume_session (which constructs a fresh EventBus)."""
```

`Orchestrator` snapshot/restore (TRUNK-CRITICAL, ADR-gated):

```python
@dataclass
class OrchestratorState:
    """In-memory shape of Orchestrator's mutable state. Distinct from the
    on-disk StateSnapshot — Orchestra maps between them."""
    branches: dict[str, Branch]
    barriers: dict[str, Barrier]
    convergence_barriers: dict[str, str]
    runnable: list[str]   # deque content as a list
    fire_queue: list[str]
    root_barrier_id: str | None
    workflow_error: str | None
    completed_emitted: set[str]
    user_interactions: list[tuple[str, Any, str]]   # deque content
    user_interaction_inflight: bool


class Orchestrator:
    def snapshot(self) -> OrchestratorState:
        """Return a deep copy of mutable state. Caller is responsible for
        ensuring no in-flight tick is mutating state; Orchestra.pause_session
        awaits the in-flight tick before calling this."""

    def restore_from(self, state: OrchestratorState) -> None:
        """Replace mutable state with `state`. Must be called on a freshly
        constructed Orchestrator that has not yet been started."""
```

### Snapshot path

The framework's `FileStorageBackend` writes to `<storage_backend_root>/<session_id>/snapshot.json`. Spren maps its `run_id` to the framework's `session_id` (per `02-data-model.md` line 87) and configures `FileStorageBackend(root=<data-dir>/data/runs)` so the on-disk path matches Spren's data model: `<data-dir>/data/runs/{run_id}/snapshot.json`. Other consumers (Cloud, CI) configure their own backends.

### Determinism guarantees

Resume correctness is **semantic-equivalence**, not byte-equivalence. LLM calls are not byte-deterministic in production (provider-side nondeterminism even at temperature 0; retry semantics; tokenizer quirks). The brief states this explicitly so reviewers don't expect byte-equivalence on traces or memory.

The resume-correctness test uses the **deterministic simulator** at `packages/framework/research/orchestration/simulator/`. The simulator replays scripted `StepResult` sequences via `DeterministicRuntime`; LLM nondeterminism is taken out of the picture. The test:

1. Run scripted workload W to completion → record `OrchestraResult.final_response` shape and per-barrier arrival counts.
2. Run scripted workload W with a pause-resume midway → record same.
3. Assert: same `final_response.success`, same `final_response.shape` (or, where final_response is a dict, same set of top-level keys and same per-key value types), same per-barrier arrival counts within the workflow.

The test does **not** assert byte-equivalence on intermediate spans, trace events, or memory contents. Spren-side observability already tolerates this (each run has its own trace; resumed runs append).

### In-flight tool-call retry semantics

**Contract: at-least-once.** Resuming a workflow may re-execute tool calls that were in-flight at pause time. Mid-tick state is not captured: pause awaits the in-flight branch tick before snapshotting, but if the workflow's next tick (post-resume) is the same agent re-issuing the same tool call (because the snapshot reflects the pre-tick state), the tool will run again.

Tools that are not idempotent surface a warning at workflow definition time (the framework's tool registry already has a `is_idempotent: bool` field — verify; if not, this is an additive change worth flagging in the ADR). The warning text references this brief.

Rejected alternative — cancel-and-rollback to the last barrier on pause: loses work the user paid for (LLM tokens already spent), complicates barrier semantics (would require synthetic "cancelled at pause" branch states), and produces worse user experience for the dominant case (user pauses to inspect, doesn't expect rollback). Documented in the ADR.

### Scheduled-event handling across the paused interval

**Contract: skip-and-log.** On resume, scheduled events whose `fire_at` has passed (e.g., `Barrier.policy.timeout` deadlines crossed during the paused interval) are logged via the standard `WARNING` channel and **skipped** — they do not fire on resume. The framework does not try to "catch up" missed schedules.

Rationale: catching up would either fire a cascade of timeout-failures the moment resume runs (likely surprising the user who paused for hours and didn't expect every barrier to fail) or require the framework to re-anchor deadlines to resume-time (semantically dishonest — a "5-minute timeout" then meant something different). Skip-and-log keeps semantics honest and lets users re-set deadlines explicitly if they need to.

### Snapshot retention / garbage collection

- **Default retention: 30 days.** Configurable per-Orchestra via `Orchestra(snapshot_retention=...)`.
- **Periodic sweep**: runs once on `Orchestra.__init__`. Calls `storage_backend.expire_older_than(snapshot_retention)`. Single sweep per Orchestra construction is sufficient for the typical Spren / Cloud / local-script lifecycle; a continuous sweeper is out of scope (a long-running daemon can re-construct an Orchestra periodically if needed, or call `expire_older_than` directly).
- **Explicit cleanup**: `Orchestra.discard_paused_session(session_id) -> None` deletes a single snapshot. Spren v0.4-29 calls this on `cancel` of a paused run.

### Discovery of paused runs at daemon startup

`Orchestra.list_paused_sessions() -> list[PausedSessionMetadata]` enumerates paused snapshots without loading their bodies. Implementation: `storage_backend.list_with_metadata()` returns `StorageEntry` records; the method reads the JSON header (just enough to extract `session_id`, `workflow_id`, `paused_at`, `framework_version`, file size) without parsing the rest. For `FileStorageBackend`, this is one `os.stat` + a small JSON streaming read per snapshot file.

Spren v0.4-29 calls this on daemon startup to populate the run-inspector UI's "paused" tab. Cloud calls it on node-spinup to discover which runs need to migrate.

### Multi-consumer storage backend abstraction

`StorageBackend` is a Protocol (not an ABC) so consumer-supplied backends don't have to inherit from a framework class — they just satisfy the interface. The framework ships `FileStorageBackend(root: Path)`. `S3StorageBackend` and `GCSStorageBackend` ship later in MARSYS Cloud's PR. CI integrations may provide their own (e.g., a backend that writes to GitHub Actions artifact upload/download).

The `Orchestra` constructor accepts `storage_backend: StorageBackend | None = None`. When `None`, `Orchestra` constructs a `FileStorageBackend` rooted at the framework's standard data directory. This mirrors how other framework optionals work (`state_manager`, `communication_manager`).

### EventBus listener restoration

The standard listener set — `StatusManager`, `TraceCollector`, registered `TelemetrySink` instances (Session 02 surface) — is restored by extracting the existing listener wiring from `Orchestra._initialize_components` into a new `_wire_event_bus` method that runs from both `__init__` and `resume_session`. The extraction is mechanical: `StatusManager` setup at orchestra.py ~178–203 and `TraceCollector` setup at ~206–217 move into `_wire_event_bus`.

**Custom listeners attached via `EventBus.subscribe` by the caller (third-party Python users; tests) are NOT restored.** The `resume_session` docstring states this explicitly. The contract is: callers that attach custom listeners must re-attach them after `resume_session` returns the AsyncIterator and before they begin iterating. This is the same contract third-party users live with after any process restart and is the only sane resolution given that listeners are arbitrary `Callable`s (typically bound methods of objects holding sockets or LLM clients) and cannot be serialized.

### Resume across framework versions

`StateSnapshot.framework_version` is read on every restore. Mismatch raises `IncompatibleSnapshotError(snapshot_version, current_version)` with a clear message: `"Snapshot was created on framework v{snapshot_version}; running framework v{current_version}. Automatic migration is not supported in v0.4. Re-run the workflow from scratch, or downgrade the framework to {snapshot_version} to resume."` v0.4 ships no migration tooling; future framework PRs may add one when the shape changes.

The version comparison is **exact-string match** in v0.4. Patch-level mismatch (e.g., 0.3.1 vs 0.3.2) rejects too. This is conservative on purpose — patch releases that touch orchestrator state shape will get caught instead of corrupting paused snapshots — and is documented in the ADR with a follow-up issue tag for "consider semver-minor compatibility once migration tooling lands".

---

## Hard rules

### Multi-consumer justification (mandatory)

- [ ] List at least one consumer beyond Spren: Cloud (`S3StorageBackend` future PR), CI integrations, framework local users — verified above
- [ ] No Spren type imported in this PR: `grep -rn 'from spren\|import spren' packages/framework/` returns zero matches
- [ ] No "if running under Spren" code paths

### Framework design principles

Per the framework's `CLAUDE.md`:
- DP-001 (pure agent logic) — pause/resume does not touch agent execution semantics
- DP-002 (centralized validation) — snapshot/restore does not parse model output; ValidationProcessor is untouched
- DP-003 (unified-barrier orchestration) — snapshot captures the unified-barrier state shape exactly
- DP-004 (branch isolation) — each `Branch` serializes independently; restore re-establishes isolation
- DP-005 (topology-driven routing) — topology is frozen at run start (snapshot stores `topology_digest` for sanity-check); routing is not re-derived from the snapshot
- DP-006 (adapter pattern) — `StorageBackend` is the adapter for snapshot persistence
- DP-007 (format pluggability) — n/a (snapshot format is JSON only in v0.4; format-pluggable serializers are out of scope)

If this feature would force a violation of any of these, **escalate** before writing code.

### TRUNK-CRITICAL touch points (ADR-mandatory)

- `Orchestra.pause_session` (rewrite the body — non-additive)
- `Orchestra.resume_session` (rewrite the body — non-additive)
- `Orchestra.__init__` (additive: two new keyword-only parameters with safe defaults)
- `Orchestra._wire_event_bus` (additive: refactor extracting existing wiring; existing call sites updated)
- `Orchestra.list_paused_sessions`, `Orchestra.discard_paused_session` (additive)
- `Orchestrator.snapshot`, `Orchestrator.restore_from` (additive: two new public methods)

The ADR enumerates each of these, justifies why each is necessary, and is approved before any code lands. No "verify and proceed" — the gate is unconditional.

### Clean code rules

- Smallest implementation that passes acceptance criteria
- Delete the legacy `StateSnapshot` shape and its serialization helpers (do not leave dead-code paths)
- No descriptive comments for self-naming code — only WHY when not obvious
- One concern per file when reasonable

---

## Tests (required for "done")

### Unit tests (in `packages/framework/tests/coordination/state/`)

- `test_snapshot.py` — `StateSnapshot` round-trips through JSON; `BranchState` and `BarrierState` capture every field of `Branch` and `Barrier` (covers `Barrier.arrived` with JSON-safe values; explicitly fails with a clear error when `Barrier.arrived` contains a non-JSON-serializable value — the brief takes the position that JSON-safe values are the contract; consumers passing non-JSON values get a clear error at pause time, not a silent corruption).
- `test_storage.py` — atomic-write succeeds; simulated mid-write crash (write fd closed between fsync and rename, then process killed) leaves prior snapshot intact; `expire_older_than` deletes entries older than the threshold and leaves newer entries.

### Integration tests (in `packages/framework/tests/integration/`)

- `test_pause_resume.py`:
  - **Semantic-equivalence**: scripted workload via the simulator at `packages/framework/research/orchestration/simulator/`; pause-then-resume produces same `final_response.success` + same per-barrier arrival counts as the non-paused baseline.
  - **Cross-process**: subprocess fixture — process A starts the run, pauses, exits; process B starts a fresh `Orchestra`, calls `resume_session`, the run completes successfully.
  - **Version mismatch**: write a snapshot with `framework_version="0.0.0"`; `resume_session` raises `IncompatibleSnapshotError`.
  - **Listener restoration**: `StatusManager` and `TraceCollector` listeners are re-attached on resume (assert `event_bus.get_listener_count(...) > 0` for the standard event types after `resume_session`).
  - **Discovery**: pause two runs; `list_paused_sessions()` returns two entries with the right `session_id` values; the snapshot bodies are not loaded eagerly (assert by inspecting `FileStorageBackend` call counts via a spy).
  - **Discard**: `discard_paused_session(session_id)` removes the snapshot; subsequent `list_paused_sessions()` does not include it.
  - **Idempotent pause**: calling `pause_session` twice on the same session returns without error and writes the snapshot at most once (or writes identical content the second time).

### Framework regression test

- Entire framework test suite passes with the SAME counts as baseline (no new failures, no new skips silently introduced)
- Document baseline + post-change counts in "What was actually built"

### Multi-consumer test

- A test that exercises `pause_session` / `resume_session` via the same surface a third-party Python user would use: construct `Orchestra(..., storage_backend=FileStorageBackend(tmp_path))`, run a workload, pause, restore, complete. Demonstrates the feature is genuinely usable from outside the framework's own internals.

---

## Open questions for the framework team

> The brief commits to nine design choices below; they appear here so the ADR can cite the rationale and so consumer briefs (Spren v0.4-29; the future Cloud PR) can match the framework's contract. Three additional items stay open for the ADR to settle.

### Items the design above commits to

1. **Snapshot format**: JSON. Pickle is version-fragile + insecure on load; MessagePack is faster but less debuggable. JSON is debuggable, version-portable, and cross-language-inspectable. Atomic writes via write-temp + fsync(file_fd) + os.replace + fsync(parent_dir_fd).

2. **Snapshot path**: filesystem default = `<storage_backend_root>/<session_id>/snapshot.json`. Spren maps its `run_id` to the framework's `session_id` and configures `FileStorageBackend(root=<data-dir>/data/runs)`, matching `02-data-model.md` line 20. Other consumers configure their own roots.

3. **In-flight tool-call handling at pause**: at-least-once. Pause awaits the in-flight branch tick; resume re-issues any tool call that was about to start when pause arrived. Tools whose `is_idempotent` flag is False surface a warning at workflow definition time. The alternative — cancel-and-rollback to the last barrier — was rejected because it loses paid-for LLM work and complicates barrier semantics.

4. **Scheduled events crossed by the paused interval**: skip-and-log. Re-anchoring to resume time is semantically dishonest (a "5-minute timeout" then meant something different); firing a cascade of timeout-failures on resume is surprising. Skip-and-log keeps semantics honest and lets users re-set deadlines explicitly.

5. **Snapshot retention / garbage collection**: 30-day default; sweeper runs on `Orchestra.__init__`; explicit `Orchestra.discard_paused_session(session_id)` for early cleanup.

6. **Discovery of paused runs at daemon startup**: `Orchestra.list_paused_sessions() -> list[PausedSessionMetadata]` returns metadata only (no eager body load). Spren v0.4-29 calls this on daemon startup; Cloud calls it on node-spinup.

7. **Multi-consumer storage backend**: `StorageBackend` is a Protocol (not an ABC). `FileStorageBackend` ships in this PR; `S3StorageBackend` / `GCSStorageBackend` ship later in MARSYS Cloud's PR. CI integrations supply their own.

8. **Resume across framework versions**: exact-string match on `framework_version`; mismatch raises `IncompatibleSnapshotError`. No migration tooling in v0.4; future framework PRs may add one.

9. **EventBus listener restoration**: standard listener set rebuilt via `_wire_event_bus` (extracted from `Orchestra._initialize_components`). Custom listeners attached by the caller via `EventBus.subscribe` are NOT restored; the `resume_session` docstring states this explicitly.

### Items open for the ADR

A. **In-tick atomicity vs. tick-boundary atomicity**: pause is at-tick-boundary, not at-instruction-granularity. The implementer must surface to the framework team whether the orchestrator currently exposes a clean "no in-flight tick" signal. If it does not, adding one is part of the ADR.

B. **Where `snapshot()` / `restore_from()` live**: directly on `Orchestrator`, or on a sibling `OrchestratorStateAdapter` to keep the orchestrator class slim? Architectural preference of the framework team. The ADR must pick one.

C. **Idempotent-tool flag**: does the framework's tool registry already carry an `is_idempotent: bool` field? If not, adding it here is an additive change worth flagging in the ADR. Verify with grep before drafting.

---

## Sign-off

On completion:

1. Update **What was actually built** below with the delta from the plan, if any
2. Update [`../../v0.4-spren-support.md`](../../v0.4-spren-support.md) checkbox for this feature
3. Note the framework release version that ships this feature (e.g., "shipped in framework v0.3.0")
4. Add a **Lessons / Surprises** entry below
5. Confirm the Spren-side brief at [`../../../spren/v0.4-extensions.md`](../../../spren/v0.4-extensions.md) v0.4-29 reflects the actual API surface that landed (`list_paused_sessions`, `discard_paused_session`, `framework_version` mismatch error)

### What was actually built (filled by implementer)

> _Implementer fills this in._
>
> Include: baseline test counts (before change), post-change test counts (must match for regression suite + new tests added), framework PR number + URL, framework release version that includes this feature, anything done differently from the plan with reasons, ADR number + link.

### Lessons / Surprises (filled by implementer)

> _Implementer fills this in._
