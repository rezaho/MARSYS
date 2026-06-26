# ADR-013: Agent Control Directive — `escalate_to_user`

**Status**: Proposed
**Date**: 2026-06-26
**Implements**: Framework Session 18 — `docs/implementation/spren/v0.3.0/13-unified-browser-and-pause-resume/framework/18-agent-control-directive.md` (co-located under the Spren bundle per founder convention)
**Related**: ADR-003 (topology-driven routing — the model this bounded-departs from), ADR-007 (pause/resume snapshot substrate), ADR-011 (resume ergonomics — `resume_session`'s keyword-only surface), ADR-012 (durable HITL — the durable `enqueue_user_interaction`/`resume_session(user_response)` seam this directive routes into)

## Context

Today the only agent→human path is `ask_user`, which the framework routes to a topology `User` det-node: validation maps it to `ActionType.ASK_USER` (gated on `has_edge_to_usernode`, `response_validator.py:227`), `_translate` returns `StepResult(kind="SINGLE_INVOKE", next_agent="User", value=question)` (`real_runtime.py:258-264`), and `_interpret`→`_handle_single_invoke`→`get_det_node("User")`→`UserNode.on_single_invoke` reaches the suspend seam (`orchestrator.py:744-747`, `det_nodes.py:159-168`). The capability is offered to the agent only when the topology wired a `User` edge (the schema gate `coordination_tools.py:122`, fed by `CoordinationContext.can_ask_user`).

That is a *pre-wired* human step. Spren v0.3 Bundle 13 — Session 62 (browser re-auth) — needs the *dynamic* case: a borrowed-browser workflow whose agent *perceives mid-run* that it has been logged out and must escalate for a human to re-authenticate, then resume. Such a workflow has no `User` node, and injecting one per-workflow is topology surgery (it also changes the graph's static readability and routes resume to the node's *successor*, not the asking agent). The dynamic escalation is **general**, not Spren-specific (multi-consumer below).

ADR-012 shipped the load-bearing half: a durable suspend seam — `enqueue_user_interaction(branch, prompt, resume_agent, *, durable=True)` records the in-flight interaction in `pending_user_interaction`, the dispatch loop snapshots-and-exits (`orchestrator.py:281-285`), and `resume_session(user_response=…)` spawns a fresh branch at `resume_agent` with the response as input (`orchestrator.py:489-540`, `orchestra.py:1609-1614`). Crucially, that seam is **agent- and topology-agnostic**: `resume_branch_with_user_response` reconstructs the resume from `(branch_id, response, resume_agent)` alone, and the snapshot persists `resume_agent`. So an escalation needs no `User` node — only a way to *reach* the durable seam from a dynamically-emitted agent directive.

This is CRITICAL/TRUNK-CRITICAL: it adds a coordination directive (new `ActionType`, a `COORDINATION_TOOL_NAMES` member, a validator arm, a tool schema), a new `OrchestratorStepResult` kind with an orchestrator dispatch arm, an additive `CoordinationContext` field, and a framework-generic agent capability. Per the framework `CLAUDE.md`, an ADR is mandatory before code.

## Decision

### 1. A new coordination directive `escalate_to_user(prompt)`

A granted agent emits `escalate_to_user(prompt)` as a native coordination tool call. It travels the same emit→validate→translate→route pipeline `ask_user` does, with the gate axis changed (see §2) and a distinct route (see §3):

- **Membership**: `escalate_to_user ∈ COORDINATION_TOOL_NAMES` (`coordination_tools.py:23`) — so `is_coordination_tool` classifies it on the coordination path (`step_executor.py:1033-1040`) and it is never handed to `ToolExecutor` as a regular tool.
- **Schema**: a new `_build_escalate_user_schema()` (a `prompt: str` param, mirroring `_build_ask_user_schema`), appended by `build_schemas` when the agent is granted (§2).
- **Validation**: `ActionType.ESCALATE_USER` + a `validate_coordination_action` dispatch arm + `_validate_escalate_user` — `_validate_ask_user` with the gate axis swapped (§2), validating a non-empty `prompt`, returning `parsed_response={"prompt": prompt}`.

### 2. Availability is SCOPED — a per-agent grant, not topology, not ambient

`escalate_to_user` is offered and accepted only for an agent explicitly granted it. The grant is a framework-generic agent capability `can_escalate: bool` (default **`False`**), threaded as `CoordinationContext.can_escalate_user` (`context.py`), derived in `_build_coordination_context` (`step_executor.py:857-915`, where the agent instance is in scope at the call site `:352`), and consumed at BOTH gates:

- **schema offer** — `build_schemas(..., can_escalate_user=…)` appends the schema only when granted;
- **validation** — `_validate_escalate_user` rejects (`is_valid=False`, permission category) when `getattr(agent, "can_escalate", False)` is false.

So escalate is `ask_user` with the gate AXIS swapped from "topology edge to `User`" to "agent is granted `can_escalate`". The topology-edge axis (`has_edge_to_usernode`, `can_ask_user`) is untouched.

**Why scoped, not universal (founder decision 2026-06-26).** Every other coordination capability in the framework is *granted*, never ambient — universal availability would make escalate the sole exception, and it would let any agent in any workflow durably suspend an unattended/scheduled run waiting for a human who may never look. Scoped is also asymmetric in cost: scoped→universal is a one-line change (`can_escalate_user=True` unconditionally in `_build_coordination_context`); universal→scoped is a breaking change once agents rely on ambient availability. The S62 consumer grants the capability to its browsing agent.

### 3. A new `ESCALATE_USER` step-kind, routed by the orchestrator directly to the durable seam

`_translate` maps a validated `ESCALATE_USER` action to a new `OrchestratorStepResult(kind="ESCALATE_USER", value=prompt)` (the prompt sourced from `validation.parsed_response`, not an undefined local). `"ESCALATE_USER"` is added to the `StepKind` Literal (`orchestrator_types.py:49-55`). A new `_interpret` arm (`orchestrator.py:703-731`) routes it directly into the durable seam:

```python
if step.kind == "ESCALATE_USER":
    self.enqueue_user_interaction(
        br, step.value, resume_agent=br.current_agent, durable=True
    )
    return
```

`br.current_agent` IS the emitting agent at the escalate boundary, so resume targets that agent — stated explicitly, not derived. No topology `User` node; no parallel suspend path; the FW16 durable machinery (capture → snapshot-and-exit → `resume_session(user_response)`) is reused verbatim. An escalation produces the SAME `pending_user_interaction` shape FW16 consumes, so snapshot/restore/resume need no change.

### 4. `durable=True` is passed directly (decoupled from `UserNode` metadata)

For the `ask_user` durable path (ADR-012 §4), durability rides `UserNode.durable`, read from the spec node's `metadata["durable"]`. Escalate has no node, so the `_interpret` arm passes the literal `durable=True`. Escalate is inherently durable: the dynamic-escalation use case (a human goes away and returns, possibly across a restart) defeats the in-memory SYNC wait. There is no non-durable escalate variant in v0.3.

### 5. The instruction surface is gated on the grant, separate from the workflow-completion block

A granted agent's system prompt gets an `escalate_to_user` behavioral-contract block (when/how to escalate, that the run pauses durably and resumes it with the reply), gated on `coord.can_escalate_user` and appended in `build_complete_system_prompt` (`base.py:73-77`). It is **not** folded into `_build_workflow_completion_instructions` (`base.py:160-190`), which early-returns unless `can_terminate_workflow or can_ask_user` — folding would hide the escalate contract in exactly the S62 case (a User-less, End-edge browsing workflow). Instruction surface ↔ tool surface stay matched on the same grant, the invariant `base.py:148-155` documents.

## Rationale

- **Why a new step-kind, not reusing the `ask_user` route.** Escalate and `ask_user` are two distinct concepts — dynamic/node-less/resume-self vs pre-wired/topology-bound/resume-successor. `ask_user`'s route requires a registered `User` det-node and resumes the node's *successor* (`det_nodes.py:150-157`), both wrong for escalate. A distinct, orchestrator-interpreted directive is the day-one-correct shape; the parallelism is two concepts each with one clean path, which the Fit discipline's brake explicitly endorses (do not force two concepts into one mechanism). `_interpret` extending with a sixth arm is a legitimate use of the `StepKind` dispatch axis — distinct from the `NodeKind`/det-node registry axis, which is the one that is extension-closed on dispatch.
- **Why route through the orchestrator, not synthesize a `User` node.** The rejected alternative (a registered-but-edge-less `UserNode`) would inject a phantom `User` node into workflows that declare none and would require widening the TRUNK-CRITICAL topology reachability validator (`graph.py:1375-1382` errors on any non-End node unreachable from Start) and co-opting `_resume_agent_for`'s fallback to get resume-self implicitly. The new-kind route needs zero validator change and states resume-self explicitly.
- **Why reuse FW16's seam.** The durable suspend/resume is already agent/topology-agnostic; escalate is simply a second *producer* of the same durable interaction. A parallel suspend mechanism would be the add-parallel failure mode.

## Alternatives considered

- **Reuse the `ask_user` route via a registered, edge-less `UserNode`** — rejected: phantom node in node-less workflows; requires widening the topology reachability validator; resume-self becomes an implicit fallback rather than an explicit contract.
- **Universal (always-on) availability** — rejected by founder: makes escalate the only ambient coordination capability; risks unattended runs stalling on spurious escalation; universal→scoped would be a breaking change. (Scoped→universal stays a one-line option.)
- **Per-workflow `User`-node injection** — rejected: topology surgery; the very thing FW18 exists to avoid.
- **A Spren-only escalate tool** — rejected: a Spren tool can't reach the orchestrator's suspend seam, and it would violate SP-018 (no framework→Spren coupling). The capability is framework-generic; the consumer grants it.
- **Folding the escalate instruction into the workflow-completion block** — rejected: that block early-returns on User-less/End-edge topologies, hiding the contract in the primary consumer's case.

## Consequences

### Backward compatibility
Fully back-compat. `can_escalate` defaults `False` — no agent gains the tool by default; a default-constructed agent is neither offered `escalate_to_user` nor passes its validation, and its full prior coordination-action set is otherwise unchanged. The `ask_user`/`User`-node static path (schema gate + validation + routing + resume-successor) is untouched. The new `StepKind` member and `CoordinationContext` field are additive. The FW16 durable/resume machinery and the topology validators are reused unchanged.

### Multi-consumer
- **Spren (the immediate v0.3 consumer)**: S62 grants `can_escalate` to its browsing agent; on a detected logged-out wall the agent escalates → durable suspend → human re-auths in the main browser → `resume_session(user_response)` → the agent re-perceives a fresh authed page and continues. No `from spren` import; no "if Spren" branch (SP-018).
- **MARSYS Cloud**: an agent hitting an out-of-band operator approval it couldn't pre-wire.
- **CI integrations**: a dynamic human gate spanning jobs (process A suspends; process B resumes).
- **Framework local users**: any agent granted the capability that needs mid-task human input without a graph `User` step.

### Known limitations
- **At-least-once re-run (ADR-007/012).** The branch is `WAITING` at the escalate boundary; the resume's re-run surface is the resumed agent's first tick — no worse than ADR-012.
- **Single-pending durable (v0.3, inherited from ADR-012).** One `resume_session(user_response)` answers the one in-flight durable interaction; durable siblings serialize FIFO. Sufficient for one re-auth per run.
- **Version-lock during a human-timescale wait (inherited from ADR-007/012).** A framework upgrade during a long escalation wait makes the run unresumable — an honest failure, not silent corruption.
- **Agent must perceive the need to escalate.** Escalation is the agent's judgment; an agent that fails to recognize it should escalate won't. This is the consumer's coaching problem (S62's browse-guidance), not a framework gap; the deferred automatic-condition raiser (below) could add a backstop later.
- **Spurious escalation stalls an unattended run.** Mitigated by scoping (only granted agents can escalate) and recoverability (a stalled run is discarded/resumed by a human; no state corruption).

### The topology-driven-routing departure (ADR-003)
`escalate_to_user` suspends without a graph `User` node — a bounded, explicit departure from ADR-003: ONE interpreted directive, routed by the orchestrator, not arbitrary imperative flow. The run's static graph no longer fully predicts its human-interaction points for escalate-granted agents; this is the deliberate price of dynamic escalation, owned here and kept bounded by the deferral below.

## Deferred (the general control-directive / flow-control primitive)

Founder-requested direction (2026-06-24), **NOT built here** — recorded so the direction is on record:
- **Generalize the raiser**: an automatic condition hook (per-step/on-signal) that raises a directive — the "watchdog" (e.g. cost > X, a tool returns Y).
- **Generalize the action set**: beyond `escalate_to_user` — `pause` (durable, no input), `redirect → agent`, `fail` — a uniform control-directive vocabulary the orchestrator interprets.
- **Attachment model**: where a hook attaches (agent/node/workflow) and how it's configured.
- **Open design questions**: condition source (agent vs automatic); the action vocabulary; how dynamic directives reconcile with topology-driven routing (ADR-003); precedence when multiple hooks fire.
- **Build trigger**: a SECOND concrete consumer (anti-pattern #4/#11). `escalate_to_user` is consumer #1; it ships as a single arm, not a framework for hooks no one calls. No `pause`/`redirect`/`fail` `ActionType` values or extra `can_*` flags are pre-added now.

## Approval

This ADR requires framework-team approval before merge. Approval is recorded here by the framework lead, OR by an explicit approval message in the PR thread.

- [x] Framework lead (founder) approved the design + full scope to proceed on 2026-06-26 (the implementer's Phase-A synthesis gate; founder chose the corrected improver-A shape + SCOPED availability). Formal merge sign-off occurs at central submodule-PR integration.
