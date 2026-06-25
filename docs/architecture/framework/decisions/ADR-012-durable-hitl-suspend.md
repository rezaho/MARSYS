# ADR-012: Durable Human-in-the-Loop Suspend/Resume

**Status**: Proposed
**Date**: 2026-06-26
**Implements**: Framework Session 16 — `docs/implementation/spren/v0.3.0/13-unified-browser-and-pause-resume/framework/16-durable-hitl-suspend.md` (co-located under the Spren bundle per founder convention)
**Related**: ADR-007 (pause/resume snapshot — the substrate this extends), ADR-011 (resume ergonomics — `resume_session`'s keyword-only params this rebases onto), ADR-013 (the `escalate_to_user` directive — the immediate v0.3 consumer, lands after this)

## Context

ADR-007 shipped durable **on-demand** pause/resume (`Orchestra.pause_session` / `resume_session`, cross-process, restart-surviving). It never composed pause with the human-in-the-loop `ask_user` wait. Today a `UserNode`/`ask_user` interaction is **synchronous and in-memory**: `Orchestrator.enqueue_user_interaction` spawns an `asyncio.create_task(_drive())` (orchestrator.py:422-445) that awaits `UserNodeHandler.handle_user_node`, which blocks on an `asyncio.Future` with a default 300s timeout. Three gaps follow:

1. **The in-flight interaction is not durably captured.** `enqueue_user_interaction` appends to `self._user_interactions` only when an interaction is *already* in-flight (orchestrator.py:409-411) — i.e. only the queued siblings. The in-flight interaction's `(prompt, resume_agent, delivery_target)` lives only in the `_drive` closure + the WAITING branch. `snapshot()` serializes the deque (orchestrator.py:1340) → the in-flight prompt/resume_agent is lost; `restore_from()` rebuilds `_resume_user_responses = None` (orchestrator.py:1377) and never re-drives it.
2. **The dispatch loop blocks on the in-memory wait.** With no runnable branches and a pending interaction, the loop blocks on `await self._resume_user_responses.get()` (orchestrator.py:273-282). While blocked it cannot re-check the pause flag, so `pause_session` on a `UserNode`-parked run hangs until the human answers or the 300s timeout fires (`quiesce()` awaits `_loop_exited_event`, orchestrator.py:328-333).
3. **No resume-with-a-response surface.** `resume_session` (ADR-007/011) continues a paused run but has no way to inject a human's answer into the suspended branch.

The need — "a workflow pauses pending an out-of-band human action that may exceed 300s and survive a restart" — is exactly "make `ask_user` durable." This is a focused EXTENSION of ADR-007, not a new mechanism: the suspended branch is already an ordinary `WAITING` branch that `snapshot()` deep-copies (orchestrator.py:1332), and `resume_branch_with_user_response` (orchestrator.py:462-494) is a **synchronous** response-injection seam with no dependency on the `_drive` closure or the `asyncio.Future` — so the in-flight wait need not stay an in-memory coroutine.

This is CRITICAL/TRUNK-CRITICAL: it makes a non-additive behavior change to `Orchestrator._dispatch_loop_inner` (the user-wait park) and `Orchestrator.enqueue_user_interaction` (a durable arm), widens the `DetNodeContext` protocol, and adds an additive parameter to `Orchestra.resume_session`. Per the framework `CLAUDE.md`, an ADR is mandatory before code.

These are **general** durable-HITL gaps, not Spren-specific (multi-consumer below).

## Decision

### 1. A `durable` arm on `enqueue_user_interaction` + a single scalar capture

`Orchestrator.enqueue_user_interaction` gains a keyword-only `durable: bool = False`. The non-durable arm (the in-memory `_drive`/Future/300s path) is **unchanged**. The durable arm: marks the branch `WAITING`, records the interaction in a single scalar `self.pending_user_interaction`, sets `self._user_interaction_inflight = True` (so a second interaction still queues as a sibling), and **spawns no `_drive`** (no Future, no timeout).

The in-flight interaction is captured as **one scalar**, NOT folded into the `_user_interactions` deque. Folding is unsafe: `resume_branch_with_user_response` pops every deque entry as a *queued sibling* and re-dispatches it (orchestrator.py:487-494), so a folded in-flight item would be mis-dispatched on resume — which is exactly why the in-flight interaction is already excluded from the deque (orchestrator.py:409-411). The scalar follows the existing live/wire split:

- live `Orchestrator.pending_user_interaction: Optional[tuple]` — the `(suspended_branch_id, prompt, resume_agent, delivery_target)` shape, matching the deque's items;
- `OrchestratorState.pending_user_interaction: Optional[tuple] = None`;
- `StateSnapshot.pending_user_interaction: Optional[UserInteractionState] = None` — the typed wire shape (reuses the existing `UserInteractionState`, snapshot.py:84-97). `Optional` + default `None` keeps pre-ADR-012 snapshots valid under `extra='forbid'`.

Mapping reuses the existing `_user_interaction_to_state` (orchestra.py:1804) forward + a one-line reverse, exactly as the deque maps — so `execution/orchestrator.py` gains no import of the snapshot wire model. The name `pending_user_interaction` is deliberately distinct from the existing bool `user_interaction_inflight`.

`resume_agent` is persisted (not re-derived on resume): `UserNode._resume_agent_for` needs the live topology successors + `branch.last_invoked_agent` (det_nodes.py:145-152), and `last_invoked_agent` is written only on the agent-continuation hop (real_runtime.py:140), never on the det-node hop. Persisting it is deterministic and means a topology edit during a human-timescale wait cannot silently re-route the resume (consistent with the version/digest lock).

### 2. Dispatch-loop snapshot-and-exit at the durable boundary

In `_dispatch_loop_inner`, when there are no runnable branches and no in-flight ticks, a **durable** pending interaction returns a new `_build_awaiting_user_result()` instead of blocking on `_resume_user_responses.get()`:

```python
if not in_flight:
    if self.pending_user_interaction is not None:
        return self._build_awaiting_user_result()   # durable: snapshot-and-exit
    if self._user_interaction_inflight or self._user_interactions:
        if self._resume_user_responses is not None:  # SYNC path: unchanged
            ...await self._resume_user_responses.get()...
    break
```

The durable-exit keys off the **scalar**, so `restore_from` re-hydrating the scalar is what makes the resumed loop behave correctly. `_build_awaiting_user_result()` (beside `_build_paused_result`, orchestrator.py:1297) returns `WorkflowResult(error="awaiting_user")` — a second pause sentinel parallel to `"paused"`.

### 3. `Orchestra.execute()` / `resume_session()` self-pause + injection

- **`_snapshot_and_write(session_id, orchestrator)`** is extracted from `pause_session`'s existing write block (orchestra.py:1409-1411 — `_build_state_snapshot` → `model_dump_json` → `storage_backend.write`) and shared by both call sites.
- **`execute()`** (and **`resume_session()`**, since a resumed run can itself re-pause awaiting-user) treat `workflow.error in ("paused", "awaiting_user")` as paused; on `"awaiting_user"` they `await self._snapshot_and_write(...)` **inside the try, before the `finally` pops `_active_orchestrators`**, and set the public `metadata["awaiting_user"] = True` (mirroring how `metadata["paused"]` is derived). `FinalResponseEvent` stays suppressed for both pause flavors.
- **`resume_session`** gains `user_response`, additive to ADR-011's signature → the combined **keyword-only-optional** surface:

```python
async def resume_session(
    self, session_id: str, *,
    canonical_topology=None, on_bus_rebuilt=None, user_response=None,
) -> OrchestraResult: ...
```

  After loading the snapshot it validates (AC-5): `user_response` provided but `snapshot.pending_user_interaction is None` → `ValueError`; a pending durable interaction but `user_response is None` → `ValueError` (you cannot resume a human-wait without the answer). When both are present it injects **before** `resume()` via the existing public seam `orchestrator.resume_branch_with_user_response(suspended_branch_id, user_response, resume_agent)`. That seam's tail (its existing in-flight-consumption logic, orchestrator.py:487-494) also clears `self.pending_user_interaction = None` — a no-op on the SYNC path. No new injection method; no orchestra.py poking of orchestrator internals beyond the existing tightly-coupled lifecycle surface.

### 4. The durable trigger — `UserNode(durable=)` flag, expressible in code AND in a workflow definition

The durable signal is a `durable: bool` flag (extend-and-unify), NOT a `CommunicationMode` value (the durable path bypasses `_drive`/`handle_user_node` entirely, so it never enters the `CommunicationMode`-consuming legacy handler) and NOT a new node kind (a sibling `DurableUserNode` would duplicate `UserNode`'s `_resume_agent_for`/`on_*` for one bool):

- `DetNodeContext.enqueue_user_interaction` (orchestrator_types.py:264) and its impl gain `*, durable: bool = False` (additive protocol widening);
- `UserNode.__init__` gains `durable: bool = False`; `on_single_invoke`/`on_dispatch` pass `durable=self.durable`;
- **workflow-definition expressibility** (founder decision 2026-06-26): a USER node carrying `durable` in the topology spec materializes a `UserNode(durable=True)`. In v0.3 the USER node rides the legacy shim (orchestra.py:638-645); the shim reads `durable` from the canonical source node (it receives the canonical topology, which retains node metadata) and constructs `UserNode(durable=...)`. Written forward-compatibly so the v0.4 generic-materialization path (analyzer.py:211-213) inherits it when the USER carve-out is removed.

### 5. `communication/` is NOT touched

The durable arm spawns no `_drive`, so `handle_user_node` / `CommunicationMode` are never on the durable path. The SYNC `ask_user` path keeps them unchanged (back-compat). Routing "durable" through `CommunicationMode` would add a mode to a subsystem the durable path provably never enters.

## Rationale

- **Why extend, not a parallel suspend mechanism.** The frame's disconfirmer — a constraint forcing the in-flight wait to stay an in-memory coroutine — is absent: `resume_branch_with_user_response` is synchronous and reconstructs the resume branch from `(branch_id, response, resume_agent)` alone (orchestrator.py:462-494). The suspended branch is already first-class durable state. So a durable interaction reuses snapshot/restore + the injection seam verbatim; a parallel mechanism would be the add-parallel failure mode.
- **Why a scalar, not the deque.** The deque's FIFO pop treats entries as queued siblings (orchestrator.py:487-494); the in-flight interaction is the *current* one, deliberately excluded from the deque (:409-411). A scalar slot is the honest shape.
- **Why a second `error` sentinel, not a new `WorkflowResult` field.** Pause already travels as a sentinel string on `.error`, and both translation sites are string checks (orchestra.py:1163, :1570). Reusing the channel adds one branch per site; a new field would thread a parallel signal through two layers.
- **Why a `durable` flag on `UserNode`, expressible in the spec.** Extend-and-unify on the existing user-interaction node. Making it spec-expressible (vs constructor-only) is the founder's call (full declarative capability now); the v0.3 cost is the shim read, written to survive the v0.4 carve-out removal.

## Alternatives considered

- **Fold the in-flight interaction into `_user_interactions`** — rejected: the FIFO pop (orchestrator.py:487-494) would re-dispatch it as a sibling on resume.
- **Hack `pause_session` to self-pause from inside the loop** — rejected: `pause_session` calls `quiesce()` + the already-terminal race guard, both wrong for a loop that exited *itself*. Extract `_snapshot_and_write` and call it directly.
- **Keep the in-memory Future + a longer timeout** — rejected: does not survive a restart and still blocks the loop; the whole point is durability across process death.
- **A `CommunicationMode.ASYNC_*` durable mode** — rejected: consumed only by the legacy `handle_user_node` the durable path never enters.
- **A sibling `DurableUserNode` kind** — rejected: duplicates `UserNode` for one bool (anti-pattern #2).
- **Constructor-only durable trigger (no spec expressibility)** — considered; overridden by founder decision to ship the full declarative capability.

## Consequences

### Backward compatibility
Fully back-compat. `durable` defaults `False` (SYNC `ask_user` unchanged — AC-6); `resume_session`'s new param is keyword-only optional and defaults to today's behavior (ADR-007 on-demand resume unchanged — AC-5); the new snapshot field is `Optional` + default `None` (old snapshots still validate under `extra='forbid'`). `_build_topology_graph` and the FW17 resume machinery are reused unchanged.

### Multi-consumer
- **Spren (the immediate v0.3 consumer)**: FW18's `escalate_to_user` directive (ADR-013) routes to `enqueue_user_interaction(durable=True)` with no topology User node — the S62 browser re-auth flow. The durable `UserNode` (code + spec) is the general pre-wired-durable-user-step capability, not the S62 path.
- **MARSYS Cloud**: operator-approval gates on managed long-runs that survive node restarts.
- **CI integrations**: a human gate spanning multiple CI jobs (process A suspends; process B resumes).
- **Framework local users**: an overnight approval that survives a reboot.
- No `from spren` import; no "if Spren" path (SP-018).

### Known limitations
- **Version-lock during a human-timescale wait.** The snapshot is locked to the exact `framework_version` (ADR-007). A framework upgrade *during* a long re-auth wait makes the run unresumable (`IncompatibleSnapshotError`) — more likely than for on-demand pause. Surfaced as an honest failure, not silent corruption.
- **Single-pending durable (v0.3).** One `resume_session(user_response)` answers the one in-flight durable interaction; durable siblings serialize FIFO. Sufficient for the real consumer (one re-auth per run). Multi-pending-durable ergonomics are deferred.
- **At-least-once re-run (ADR-007).** The suspended branch is `WAITING` (not mid-tool-call) at the durable boundary, so the resume's re-run surface is the resume_agent's first tick — no worse than ADR-007.
- **`auto_inject_user` / metadata not in the digest** (inherited from ADR-011) — a durable USER node's `durable` flag rides the workflow definition (re-analyzed on resume), and the in-flight datum is authoritative from the snapshot scalar, so resume does not re-derive it; a topology edit changing `durable` during a wait is governed by the same digest limitation ADR-011 already documents.

## Approval

This ADR requires framework-team approval before merge. Approval is recorded here by the framework lead, OR by an explicit approval message in the PR thread.

- [x] Framework lead (founder) approved the design + full scope to proceed on 2026-06-26 (the implementer's Phase-A synthesis gate; founder chose option 2 — the workflow-definition durable trigger included). Formal merge sign-off occurs at central submodule-PR integration.
