# ADR-005: Unified-Barrier Orchestration Algorithm

**Status**: Accepted
**Date**: 2026-04-21 (initial), 2026-05-01 (phase 3.5 amendments)
**Supersedes**: ADR-001 in implementation only — the decision in ADR-001 (branch-based parallel execution) remains in force.
**Unaffected by [ADR-008](ADR-008-unified-node-kind-model.md)**: Session 08 left the orchestrator dispatch body, the `TopologyLike` Protocol, and the `DeterministicNode` behaviour classes/hooks unchanged. Only how a det-node is *materialized* changed (from `kind` at the `analyzer._add_nodes` seam, not an `isinstance`/string instance in `Topology.nodes`). The barrier algorithm described here is unchanged.

## Context

Phase 3 cutover (commit `bc19b98`) replaced the legacy `BranchSpawner` / `DynamicBranchSpawner` (creation) + `BranchExecutor` (execution) split with a single `Orchestrator` event loop and a per-branch `RealRuntime` driver. This ADR formalizes the algorithm so future contributors can reason about correctness without reverse-engineering the implementation.

The previous split caused three classes of bugs:

1. **Drift between creation and execution.** The spawner could enqueue a branch whose execution conditions had since changed; the executor had no symmetric view of pending sources.
2. **Double-counted contributions.** A branch could be registered as a child of one barrier and a candidate of another; failure paths could record the same source twice.
3. **Inconsistent failure cascades.** Different barrier types (FORK / CONVERGENCE / ROOT) had different failure routing, so an upstream failure could leave one barrier orphaned.

A unified-barrier model with one `Barrier` shape (no enum), a fixed-order set of fire gates, and a narrow `DetNodeContext` Protocol for non-LLM nodes addresses all three.

## Decision

A single `Orchestrator` (`src/marsys/coordination/execution/orchestrator.py`, 1151 LoC) owns the entire workflow execution graph. It maintains:

- `branches: dict[str, Branch]` — every branch ever spawned, keyed by id.
- `barriers: dict[str, Barrier]` — every barrier ever created.
- `convergence_barriers: dict[str, str]` — currently-OPEN rendezvous barrier per convergence node (lazy).
- `runnable: deque[str]` — branch ids ready to dispatch.
- `_fire_queue: list[str]` — barriers queued for fire/cancel processing between dispatch ticks.
- `root_barrier_id: Optional[str]` — the unique workflow sink.

Every workflow has exactly one ROOT barrier. Every spawn (parallel-fork or rendezvous) produces a single `Barrier` shape with a `resolver_branch` (None for ROOT only). Branches transition through `RUNNING → WAITING → RUNNING → … → TERMINATED | FAILED | ABANDONED`.

Concurrency model: every `RUNNING` branch in the runnable queue is dispatched as an `asyncio.Task` calling `Runtime.step(branch)`. The orchestrator awaits `FIRST_COMPLETED`, applies side effects from the returned `StepResult` inline, drains the fire queue, and re-dispatches newly-runnable branches. Cooperative scheduling means the orchestrator's algorithm body is single-threaded between awaits while I/O-bound `step` calls (e.g., concurrent LLM requests) run truly in parallel.

## Data model

### `Branch` (`orchestrator_types.py:96`)

```python
@dataclass
class Branch:
    id: str
    current_agent: str
    status: BranchStatus            # "RUNNING" | "WAITING" | "TERMINATED" | "FAILED" | "ABANDONED"
    delivery_target: str            # exactly one barrier id (invariant I1)
    input: Any = None
    memory: list[dict[str, Any]] = field(default_factory=list)
    waiting_on: Optional[str] = None
    candidate_of: set[str] = field(default_factory=set)
    parent_spawn: Optional[str] = None
    step_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_invoked_agent: Optional[str] = None
    consecutive_content_only: int = 0
```

### `Barrier` (`orchestrator_types.py:131`)

```python
@dataclass
class Barrier:
    id: str
    policy: ConvergencePolicy
    status: BarrierStatus = "OPEN"  # "OPEN" | "FIRED" | "CANCELLED"
    resolver_branch: Optional[str] = None       # None only for ROOT
    resolver_agent: Optional[str] = None        # mirrors resolver.current_agent
    rendezvous_node: Optional[str] = None       # set for rendezvous; None for fork
    candidates: set[str] = field(default_factory=set)
    arrived: dict[str, Any] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)
    upstream: set[str] = field(default_factory=set)
    downstream: set[str] = field(default_factory=set)
```

`Barrier.is_root` is computed (`resolver_branch is None and rendezvous_node is None`). The `Barrier.kind` property (`"ROOT"` / `"CONVERGENCE"` / `"FORK"`) is informational only — the orchestrator's algorithm does not branch on it.

### `ConvergencePolicy` (`orchestrator_types.py:115`)

```python
@dataclass
class ConvergencePolicy:
    min_ratio: float = 1.0                                          # default: all candidates required
    on_insufficient: Literal["fail", "proceed", "user"] = "fail"
    terminate_orphans: bool = True                                  # abandon pending on fire
    timeout: Optional[float] = None                                 # not yet enforced
```

## Invariants (preserved across all paths)

From `orchestrator_types.py:17-24`:

1. **I1.** Every `Branch` has exactly one `delivery_target` (a barrier id, never None).
2. **I2.** Every `Barrier` is `OPEN` exactly once and `FIRED` or `CANCELLED` exactly once.
3. **I3.** Every non-ROOT `Barrier` has `resolver_branch` set at creation.
4. **I4.** A `Branch` settles in exactly one of `{TERMINATED, FAILED, ABANDONED}`.
5. **I5.** When a branch settles, every barrier with it as candidate is notified.
6. **I6.** `fire(barrier)` is idempotent (`status` check is the first fire gate).
7. **I7.** `arrived ∩ failed = ∅` at any barrier (a source contributes once).

## Two paths to a barrier (one shape)

### Parallel-fork path

An agent emits `invoke_agent` with multiple invocations in a single tool call. The orchestrator:

1. Creates a fork `Barrier` with `resolver_branch` = the invoking branch (which transitions `RUNNING → WAITING`), `rendezvous_node = None`.
2. Spawns N child branches, each with `delivery_target` = the new fork barrier.
3. Awaits delivery. When all children deliver (passing the six fire gates), the fork fires: `Orchestrator._aggregate(barrier)` returns an `AgentInput.aggregate(...)` (typed-text-blocks per source), the resolver branch transitions back to `RUNNING` with that aggregated input, and re-enters the runnable queue.

### Rendezvous path (lazy `ensure_barrier`)

An agent in branch B invokes target Y via `invoke_agent("Y", request)`. If `topology.is_convergence(Y)` (multiple incoming edges), the orchestrator's reach-based registration (`_register` at `orchestrator.py:774`) ensures a rendezvous barrier exists at Y:

1. `_ensure_barrier(Y)` returns the currently-OPEN rendezvous barrier at Y, or creates one with `resolver_branch` = a freshly-spawned `WAITING` branch at Y, `rendezvous_node = Y`.
2. B's value flows to that rendezvous barrier; B transitions to `TERMINATED`.
3. When all expected branches from the predecessor convergence set have arrived (or failed), the rendezvous fires; `_aggregate` packages the arrivals; the rendezvous resolver transitions to `RUNNING` and runs at Y.

Both paths produce identical `Barrier` shapes — there is no `BarrierType` enum.

## The six fire gates (in evaluation order)

`Orchestrator._maybe_fire(bar)` (`orchestrator.py:957`) evaluates the gates in order. A barrier fires only when **ALL** gates pass.

| # | Gate | Logic | Source |
|---|---|---|---|
| 1 | **status** | `bar.status == "OPEN"` (else return — guarantees idempotency, invariant I6) | line 958 |
| 2 | **ROOT-defer** | If `bar.is_root`: while any branch is `RUNNING` or `WAITING`, return. Re-triggered via `_notify_branch_settled` when branches settle. (Fix 1: prevents premature ROOT-fire on initial dispatch.) | lines 963-969 |
| 3 | **upstream** | For each barrier in `bar.upstream`: if `OPEN` → return; if its resolver is `FAILED` → cascade via `_fire_with_failure`; track if upstream resolvers are still active. | lines 971-989 |
| 4 | **pending** | If `bar.pending()` is non-empty (some candidate hasn't arrived or failed yet) → return. | line 992 |
| 5 | **vestigial-cancel** | If barrier has zero arrivals AND zero failures AND is not ROOT, AND no upstream resolver is active → cancel (rather than fire) via `_cancel(bar)`. | lines 996-1000 |
| 6 | **ratio** | `len(arrived) / (len(arrived) + len(failed)) < bar.policy.min_ratio` → run `on_insufficient` action: `"fail"` → `_fire_with_failure`, `"user"` → `_fire_with_failure` (user error), `"proceed"` → fall through. | lines 1003-1011 |

When all gates pass, `_fire(bar)` runs:

1. `bar.status = "FIRED"`.
2. If rendezvous: remove from `convergence_barriers`.
3. If `policy.terminate_orphans`: abandon every branch still in `bar.pending()`.
4. `aggregated = self._aggregate(bar)` — packages arrivals (typed-text-blocks per source for non-ROOT).
5. Emit `ConvergenceEvent` and (for non-ROOT forks with ≥ 2 children) `ParallelGroupEvent(status="completed")`.
6. If barrier has a resolver: transition `resolver.status = "RUNNING"`, set `resolver.input = aggregated`, append to `runnable`. (For rendezvous resolvers spawned `WAITING` internally, emit `BranchCreatedEvent` now so trace spans exist for convergence-link attachment.)
7. Cascade: append every barrier in `bar.downstream` to `_fire_queue`.

## Failure cascade

When a branch fails, `_fail_to(br, error)` (`orchestrator.py:740`):

1. Look up `bar = barriers[br.delivery_target]`. If absent or not `OPEN`: mark `br` `FAILED`, unregister from all `candidate_of` barriers, emit completion events, return.
2. Otherwise: record `bar.failed[br.id] = error`, ensure `br.id` is in `bar.candidates`, mark `br.status = "FAILED"`, emit `BranchCompletedEvent(success=False)`.
3. For every barrier `other` in `br.candidate_of - {target_id}`: discard `br.id` from `other.candidates` (so other barriers don't wait on a failed source); enqueue `other` for fire/cancel.
4. Enqueue `bar` for fire/cancel.
5. Notify resolver-settled and branch-settled.

If the failure ratio exceeds `policy.min_ratio`, the barrier fires normally (with the failed source in `arrived` proportion). If insufficient, `policy.on_insufficient` decides: `"fail"` cascades the failure to the next resolver via `_fire_with_failure`; `"proceed"` fires anyway with whatever arrived; `"user"` cascades with a user-marked error.

ROOT failure sets `_workflow_error` and ends the run; `Orchestrator.run` returns a `WorkflowResult` with the error string.

## Reach-based registration

To make the rendezvous path work without static planning, the orchestrator uses **reach-based registration** (`_register` at line 774, `_refresh_reachable` at line 789):

- When a branch is spawned at agent X, it's registered as a candidate of every rendezvous barrier reachable from X (via `topology.reachable_convergence_points(X)`).
- When a branch transitions to a new agent Y (via `SINGLE_INVOKE`), `_refresh_reachable(branch)` drops it from rendezvous barriers no longer reachable and registers with newly-reachable ones.
- Lazy `_ensure_barrier(cnode)` creates the rendezvous barrier the first time a branch reaches a convergence point.

This eliminates the need for the orchestrator to plan all forks at workflow start. Convergence is discovered as the workflow runs.

## Resolver branches and `_aggregate`

A resolver branch is the branch that resumes when a barrier fires:

- **Fork barrier** — resolver is the invoking agent's branch, transitioned to `WAITING` at fork creation.
- **Rendezvous barrier** — resolver is a freshly-spawned `WAITING` branch at the rendezvous node.
- **ROOT barrier** — no resolver (`resolver_branch = None`).

`_aggregate(bar)` (`orchestrator.py:1114`) packages arrivals into the resolver's input:

- For non-ROOT barriers: returns `AgentInput.aggregate({source_agent: value})` — a single user `Message` with typed-text-blocks per source. Branch-id → agent-name resolution makes the per-source markers human-readable.
- For ROOT: preserves raw arrival values for `final_response` exposure (no `AgentInput` wrapping; the public `WorkflowResult` exposes them directly).

## Consequences

- **Positive:**
  - Single source of truth for branch / barrier state — no creation / execution drift.
  - Idempotent fire (gate 1) makes correctness trivial to reason about.
  - Reach-based registration avoids static planning; convergence emerges from runtime decisions.
  - Narrow `DetNodeContext` Protocol keeps non-LLM nodes pluggable without exposing internal state.
  - One `Barrier` shape simplifies failure cascade (no per-type code paths).

- **Negative:**
  - The fixed-order fire gates are subtle; correctness depends on evaluating them in the documented order. Adding a new gate requires considering interaction with all existing gates.
  - Reach-based registration is `O(branches × convergence_points)` per spawn; for very wide topologies this could become a hot spot (not yet measured).

- **Risk:**
  - A bug in `_register` / `_refresh_reachable` (over- or under-registration) could leak orphan candidates or starve barriers. Mitigation: the `_register` / `_refresh_reachable` logic is exercised heavily by `tests/coordination/orchestrator/`.

## Alternatives Rejected

1. **Two `Barrier` types (`FORK` and `CONVERGENCE`) with separate fire logic.**
   Rejected because failure cascade became inconsistent across types and added cyclomatic complexity in every gate. The unified shape with `is_root` / `rendezvous_node` properties carries the same information without the typed-state explosion.

2. **Pre-plan all forks at workflow start (static planning).**
   Rejected because workflows are dynamic — agents decide at runtime whether to fan out. Static planning would require reserving capacity for the worst-case fan-out, wasting branches when the agent decides to invoke a single peer.

3. **Make `_maybe_fire` a coroutine and `await` between gates.**
   Rejected because the gates are synchronous reads of in-memory state. Adding awaits would introduce interleaving bugs (a branch could settle between gates 3 and 4) without buying anything.

## References

- Source: `src/marsys/coordination/execution/orchestrator.py` (1151 LoC).
- Companion: `src/marsys/coordination/execution/orchestrator_types.py` (data model + Protocols).
- Det-node interaction: `src/marsys/coordination/execution/det_nodes.py`.
- Architecture overview: [overview.md §5 Unified barrier](../overview.md#5-unified-barrier).
- Design principle: [DP-003: Unified-barrier orchestration](../design-principles.md#dp-003-unified-barrier-orchestration).
- Deprecation timeline: [ADR-006](ADR-006-deprecation-timeline.md).
- Original branch-based decision: [ADR-001](ADR-001-branch-based-parallel-execution.md) (preserved as historical record).
- Centralized validation: [ADR-002](ADR-002-centralized-response-validation.md) (validator now invoked from `RealRuntime.step`).
