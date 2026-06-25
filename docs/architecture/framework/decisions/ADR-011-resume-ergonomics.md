# ADR-011: Resume Ergonomics for External Cross-Process Consumers

**Status**: Proposed
**Date**: 2026-06-24
**Implements**: Framework Session 17 — `docs/implementation/spren/v0.3.0/13-unified-browser-and-pause-resume/framework/17-resume-ergonomics.md` (co-located under the Spren bundle per founder convention)
**Related**: ADR-007 (pause/resume snapshot — the substrate this makes externally usable)

## Context

ADR-007 shipped durable on-demand pause/resume (`Orchestra.pause_session` / `resume_session`, cross-process, restart-surviving). But `resume_session` is only cleanly usable by a consumer that `execute()`d the run in the *same* process. A consumer that reconstructs a paused run from disk in a *fresh* process (Spren Session 61 re-materializes the frozen workflow → builds a new `Orchestra` → resumes) hits two gaps:

1. **Topology must be pre-bound, but it can only be built inside `execute()`.** `resume_session` raises `StateError(RESUME_NO_TOPOLOGY)` unless `self.topology_graph` is already set (`orchestra.py:1439-1445`). The snapshot deliberately does not carry the topology (`:1434-1438`). The only code that builds an *equivalent* `topology_graph` — `analyze` + `_apply_legacy_topology_shim` + `validate`/`validate_workflow` — lives inline in `execute()` (`:1024-1042`) and is `None` until then (`:430`). A consumer *can* poke `orch.topology_graph` / `orch.canonical_topology` directly (the existing tests do, `test_pause_resume.py:563`), but that skips the analyze/shim/validate pipeline → a graph **not equivalent** to execute's, which then mismatches the snapshot's `topology_digest`.

2. **The resume bus rebuild silently drops the consumer's custom listeners.** `resume_session` rebuilds the EventBus (`self.event_bus = EventBus(); self._wire_event_bus()`, `:1427-1428`), restoring only the standard listener set (StatusManager, TraceCollector, AGGUITranslator). A consumer's own `bus.subscribe(...)` — e.g. a per-run cost/telemetry adapter — is lost, and the rebuild + dispatch happen *inside* `resume_session`, so there is no external seam to re-attach after the new bus exists.

This is CRITICAL-tier per the framework `CLAUDE.md`: it extracts a shared helper from a TRUNK-CRITICAL method (`Orchestra.execute`) and adds parameters to a TRUNK-CRITICAL method (`Orchestra.resume_session`). The changes are additive (two optional, keyword-only params) + a behavior-preserving refactor, but the policy mandates an ADR before code.

These are **general** resume-ergonomics gaps, not Spren-specific: any cross-process resume consumer (a hosted control plane resuming on a fresh node after migration/restart; a CI integration where process B resumes what process A snapshotted) needs topology re-binding and custom-listener preservation.

## Decision

### 1. Extract `_build_topology_graph` (the single topology-build path)

Extract `execute()`'s topology-build block (`orchestra.py:1024-1042`) into a private helper, mirroring the framework's existing execute/resume-sharing idiom (`_initialize_per_topology` `:286`, `_wire_event_bus` `:318` — both `self`-mutating private helpers called by both paths).

```python
def _build_topology_graph(self, canonical, execution_config) -> None:
    """The single topology-build path (extracted from execute()): analyze +
    legacy shim + validate. Sets BOTH self.canonical_topology and
    self.topology_graph. Called by execute() and resume_session()."""
    self.canonical_topology = canonical
    canonical.metadata = canonical.metadata or {}
    canonical.metadata.setdefault("auto_inject_user", False)
    self.topology_graph = self.topology_analyzer.analyze(canonical)
    self.topology_graph.metadata["execution_config"] = execution_config
    self._apply_legacy_topology_shim(self.topology_graph, canonical)
    self.topology_graph.validate()
    self.topology_graph.validate_workflow()
```

- **Sets both** `self.canonical_topology` and `self.topology_graph` — the resume digest check (`_compute_topology_digest`, `:1617`) and `_initialize_per_topology` (`:310`) both read `self.canonical_topology`, so setting only `topology_graph` would pass the guard but then mis-digest or build rules against a stale canonical.
- **`execute()` keeps `:1013-1023`** — `_ensure_topology(topology)`, execution-config resolution + `context["execution_config"]`, and the `auto_inject_user` read from execute's `context` — that prep is caller-local and has no `context` on the resume path. `execute()` keeps its explicit `canonical.metadata["auto_inject_user"] = context.get(...)` set *before* calling the helper (so its behavior is unchanged); the helper's `setdefault(False)` is a no-op for execute and the safe default for resume. The trace-update block (`:1044-1053`, reads execute-only `session_id`/`trace_collector`) also stays in `execute()`.

### 2. Two additive, optional, keyword-only params on `resume_session`

```python
async def resume_session(
    self, session_id: str, *,
    canonical_topology=None,
    on_bus_rebuilt: "Callable[[EventBus], None] | None" = None,
) -> OrchestraResult: ...
```

- **`canonical_topology`** — when supplied, bind via `_build_topology_graph(canonical_topology, execution_config)` *before* the `RESUME_NO_TOPOLOGY` guard (`:1439`) and the digest check (`:1448`). A cross-process consumer need not pre-set `topology_graph`/`canonical_topology`.
- **`on_bus_rebuilt`** — when supplied, invoked once as `on_bus_rebuilt(self.event_bus)` **after** the guard and the digest check pass (after `:1453`, before `_initialize_per_topology` `:1459` / dispatch `:1509`), so the consumer re-attaches its own subscribers to the rebuilt bus.

The bus rebuild stays at `:1427-1428` (the listener-rebuild test, `test_pause_resume.py:567-577`, depends on rebuild-before-digest ordering). Both params default to `None` → today's behavior for every existing caller.

### 3. `on_bus_rebuilt` fires only on a resume that proceeds

The callback fires **after** the precondition checks (topology bound + digest match). On a `RESUME_NO_TOPOLOGY` or `IncompatibleSnapshotError` abort it does **not** fire — so a consumer's subscribers never attach to a bus a failing resume discards, and a raising callback cannot preempt the real precondition error. (Coordinate the keyword-only signature with the future ADR-012 `user_response` param — distinct additive concerns.)

## Rationale

- **Why extract a helper, not inline the build into resume.** AC-2 (resume produces the graph *equivalent* to execute's) is only honest with a single source. Duplicating analyze+shim+validate into resume is the add-parallel failure mode and would drift the moment the (v0.4-bound) shim changes. Extraction matches the established `_initialize_per_topology`/`_wire_event_bus` idiom — a third `_build_*`/`_initialize_*` helper a reader expects.
- **Why two kwargs, not a `bind_topology()` method or an options object.** The two concerns are orthogonal (topology bind vs listener restore). The listener-restore concern has no home *outside* `resume_session` (the rebuild lives there), so a separate `bind_topology()` would split one resume-ergonomics concern across two surfaces and still need a kwarg on `resume_session`. An options dataclass is premature abstraction for two optional args.
- **Why `on_bus_rebuilt` fires post-precondition.** Validate before side effects: running consumer code on a resume that's about to raise `IncompatibleSnapshotError` is the wrong order, and it lets a callback error mask the real precondition error. On a failed resume there are no run events and the bus is discarded, so the callback firing there buys nothing.
- **Why the helper defaults `auto_inject_user`.** It is an *input* to `analyze()` (`analyzer.py`) that changes the analyzed graph's node/edge structure (a legacy User-node injection) but is **not** part of the digest. A resume whose supplied `canonical_topology` carried a different `auto_inject_user` than the original run would silently build a divergent graph. The helper defaults it `False` (the new-style case); legacy auto-inject topologies are documented NOT cross-process-resumable in v0.3, and the digest-ignores-metadata gap is RISK-LOGGED for a future snapshot-format PR.

## Alternatives considered

- **Public `bind_topology(canonical)` method** — rejected: splits one resume-ergonomics concern across two API surfaces (you still need a kwarg on `resume_session` for `on_bus_rebuilt`), and is heavier than the private-helper idiom the codebase already uses.
- **Standing persistent-subscriber registry on `Orchestra`** — rejected: a heavier, stateful mechanism for what a one-shot post-rebuild callback solves.
- **Fire `on_bus_rebuilt` at the bus rebuild (`:1428`), symmetric with `_wire_event_bus`** — rejected: it runs consumer code on doomed (digest-mismatch / no-topology) resumes and lets a raising callback preempt the precondition error; firing post-precondition matches the callback's purpose (prepare subscribers for the upcoming dispatch).
- **Serialize the topology into the snapshot** (so no rebind needed) — deferred: a snapshot-format change beyond this slice's additive scope (ADR-007 already flags it as a future PR).

## Consequences

### Backward compatibility
- Fully back-compat. Both params are optional + keyword-only and default to today's behavior; `resume_session(session_id)` is unchanged, including the existing tests that set `orch.topology_graph` directly. The `_build_topology_graph` extraction is behavior-preserving for `execute()` (verified by the existing execute/topology suite staying green before the resume call is added).

### Multi-consumer
- **Spren (S61)** — reconstruct + resume a paused run from disk with per-run cost tracking intact (the immediate driver).
- **MARSYS Cloud** — resume on a fresh node after migration/restart, with telemetry subscribers preserved.
- **CI / local consumers** — process B resuming a run process A snapshotted.
- No `from spren` import; no "if Spren" path (SP-018).

### Known limitations
- Legacy `auto_inject_user` topologies are not cross-process-resumable in v0.3 (the digest does not cover the flag). RISK-LOGGED for a future snapshot-format PR.
- A consumer supplying a `canonical_topology` whose digest differs from the snapshot's gets a clean `IncompatibleSnapshotError` after the bind (not a silent wrong-topology run).

## Completion — the bus rebuild must re-point reused emitters (2026-06-25)

`on_bus_rebuilt`'s motivating use case — re-attaching a per-run cost adapter so post-resume LLM spend is billed — did not actually work as first shipped. Spren's S61 **live** test (real OAuth model, real pause→resume) found it: `resume_session` rebuilds `self.event_bus` and hands consumers the new bus via `on_bus_rebuilt`, and `_wire_event_bus` re-creates the listener set (TraceCollector / StatusManager / AGGUITranslator) on it — **but the REUSED `step_executor`** (which emits `LLMCallEvent`, the event cost is computed from) **and `_user_node_handler` still held the prior bus.** So the resumed dispatch's LLM events published on the stale bus, and a consumer re-attached via `on_bus_rebuilt` (subscribed to the new bus) received nothing. FW17's own resume test missed this because it used a **stub agent that makes no LLM call** — `on_bus_rebuilt` was only ever exercised against orchestrator-level events (`BranchCompletedEvent`), never a real `LLMCallEvent`.

Fix: `_wire_event_bus` now also re-points the reused publishers (`step_executor.event_bus`, `_user_node_handler.event_bus`) to `self.event_bus`, guarded (they are created *after* this call in `__init__`, so the guards no-op there and bind the fresh bus at construction; on resume they exist and get re-pointed). With this, a resumed run's `LLMCallEvent`s reach the rebuilt bus and `on_bus_rebuilt` delivers its full contract. Regression: the framework pause/resume suite (34 tests) stays green; Spren's `test_pause_resume_live.py` asserts `total_cost_usd` accrues across a real resume.

## Approval

This ADR requires framework-team approval before merge. Approval is recorded here by the framework lead, OR by an explicit approval message in the PR thread.

- [ ] Framework lead approval: _pending_
