# Spren node model — building blocks

**Status**: design locked 2026-05-15; framework-ground-truth section
rewritten 2026-05-17 for ADR-008 (framework node-kind unification). The
product design — the palette category model, the Start singleton rule,
the Tools/Logic/Data inactive treatment, extensibility — is unchanged.
What changed is the framework contract underneath it: Spren no longer
owns a node taxonomy or a materialization contract; it consumes the
framework's canonical wire types directly. Supersedes the node
description in [`02-data-model.md`](./02-data-model.md).

## Principle

The visual-builder node **palette** is Spren's own UX layer, modeled on
OpenAI's Agent Builder: it groups nodes into product categories
(Agents / Core / Tools / Logic / Data) and decides what is droppable
when. That categorization is **pure frontend presentation**.

The node **model itself is the framework's**, not Spren's. Post-ADR-008
the marsys wire types (`NodeSpec`, `EdgeSpec`, `TopologySpec`,
`WorkflowDefinition`, `AgentSpec`, …) are the single source of truth, and
Spren re-exports them through a thin `spren.models` façade (SP-005 in its
strict form: there is no Spren mirror to drift). There is **no Spren
"materialization contract"** and no `spren/runs/materialize.py`
node-translation layer — that earlier design assumed Spren owned the
taxonomy. It does not. `materialize_run` is now a one-call adapter over
the framework's `pydantic_to_topology` + `pydantic_to_execution_config`.

## Framework ground truth (post-ADR-008)

marsys's canonical topology is **homogeneous**: every node is a plain
`marsys.coordination.topology.core.Node` discriminated by a single
`kind` field — there is no separate deterministic-node *input* type.

- **`NodeKind`** (`coordination/topology/core.py`) has exactly
  `{AGENT, START, END, USER}` (wire values `agent` / `start` / `end` /
  `user`). The legacy `SYSTEM` / `TOOL` members were **removed** as
  vestigial; the framework's `NodeSpec` field validator hard-raises a
  migration-message `ValueError` on a stored `system`/`tool` value.
- **`NodeSpec.kind: NodeKind = AGENT`** is the wire field
  (`coordination/topology/serialize.py`). The old Spren field was
  `node_type`; it no longer exists. `AGENT` nodes carry `agent_ref`
  (the agent **name** — the registry key, bound by `pydantic_to_topology`
  against `AgentSpec.name`); START/END/USER carry no `agent_ref`.
- **Deterministic behaviour is materialized *from* `Node.kind` at the
  analyzer seam**, never stored and never an input. `det_nodes.py`
  (`StartNode`/`EndNode`/`UserNode`/`DeterministicNode`,
  `NODE_KIND_BEHAVIOUR`) is *internal analyzer behaviour* keyed by
  `NodeKind`; `Topology.nodes` is homogeneous plain `Node`. There is no
  "build a `StartNode` instance and put it in the topology" path — that
  was the pre-ADR-008 model and is gone.
- `pydantic_to_topology(spec, tool_registry, handler_registry)`
  materializes every `NodeSpec` as a plain `Node(kind=…)`, constructing
  and binding agents once. A `WorkflowDefinition` with nodes but **no**
  `kind=START` node still deserializes — it emits a `DeprecationWarning`
  and a runtime shim synthesizes Start (the shim is **removed in v0.4**;
  an explicit Start is the durable shape).

Spren's earlier `NodeType = {user, agent, system, tool}` mirror copied
the now-removed vestigial members and modeled a taxonomy Spren does not
own. It has been deleted, not corrected — Spren consumes `NodeKind`.

## The model (palette categories — product design, unchanged)

A palette node has a **category** and a **type** within it. Category is
frontend presentation; every node that reaches the wire is a framework
`NodeSpec` with a `kind`.

| Category | Types | Becomes on the wire | v0.3 |
|---|---|---|---|
| **Agents** | generic Agent (**active**); specialized — Browser, Code, DataAnalysis, FileOperation, WebSearch, InteractiveElements, Learnable; future e.g. Guardrail (**inactive**, see P12) | generic → `NodeSpec(kind="agent")` + an `AgentSpec`; specialized cannot round-trip (framework gap) so they are not droppable | generic **active** / specialized **inactive** |
| **Core** | Start; End; User | `NodeSpec(kind="start" / "end" / "user")` — real persisted kind nodes | **active** |
| **Tools** | (a) agent-attached tool; (b) tool-as-node | (a) entries in the `AgentSpec` tool list; (b) future framework node kind | **inactive** |
| **Logic** | conditional (if/else); loop/while | future framework `NodeKind` members | **inactive** |
| **Data** | transform; set-state | future framework primitive | **inactive** |

**Inactive types render in the palette as visibly disabled
("coming soon"), never droppable.** Spren must not let a builder produce
a topology it cannot run — that is the RUN-3d failure class. Activation
follows the framework roadmap: a category goes active when its
`NodeKind` exists framework-side.

### Core specifics

- **Start**: exactly one per canvas, present by default, non-deletable.
  Seeded as a real `NodeSpec(kind="start")` in a fresh canvas; the
  single removal guard (`isStartKind`) backs the delete button, the
  xyflow delete key (`node.deletable = false`), and selection/edge-driven
  removal. marsys also enforces ≤1 Start.
- **End**: zero-or-more, user-added, not default —
  `NodeSpec(kind="end")`.
- **User**: the canvas may have **multiple** User nodes; each is a real
  `NodeSpec(kind="user")` and round-trips as such. There is **no
  "collapse to a single canonical UserNode"** — that was the deleted
  det-node-instance model. v0.3 wires no executable User handler, so a
  run that *reaches* a User node fails at the framework (expected, not a
  Spren bug); the canvas may still place, lint, and persist User nodes.
- A runnable workflow requires every agent path to reach End or User
  (marsys `validate_workflow()`). Lint surfaces this pre-flight so the
  builder guides the user rather than failing at run time.

### Specialized agents are inactive in v0.3 — a framework wire gap, not a UI deferral (P12)

> Corrected 2026-05-17. An earlier draft framed specialized agents as a
> "frontend authoring preset (templated `AgentSpec`), UX deferred to
> task #21". That was wrong and is retracted: the constraint is a
> framework serialization-fidelity gap, and a templated-preset
> approximation cannot actually *run* as the specialized agent.

Verified against framework primary source (READ-ONLY from this worktree):

- `AgentSpec` (`agents/serialize.py:50-68`, `ConfigDict(extra="forbid")`)
  mirrors only the **base** `Agent` constructor surface — there is **no**
  field naming the agent class.
- `pydantic_to_agents` (`serialize.py:158`) hard-codes `Agent(...)` (the
  base class) for every spec; no class dispatch exists.
- The specialized classes **override real execution behavior**, not just
  `__init__`: e.g. `InteractiveElementsAgent._run`
  (`browser_agent.py:332`, the core loop), `BrowserAgent._pre_step_hook`
  (`:1969`), async construction (`create_safe` / `_initialize_browser`),
  resource lifecycle (`close`/`cleanup`/`__del__`), ~30 browser-control
  methods.

So `BrowserAgent → AgentSpec → Agent` is **lossy**: a stored→materialized
workflow reconstructs a base `Agent` with the same tools/instruction but
none of the overridden behavior. A "preset" would therefore be a
non-runnable look-alike — rejected (SP-007 spirit).

**v0.3/v0.4 decision (2026-05-17, user-confirmed):** specialized-agent
palette items render **inactive** ("coming soon"), exactly like
Logic/Tools/Data — not droppable, not a preset. The generic Agent + Core
are the only active Agent-category nodes. Truly supporting
create-and-run requires a **framework change** (an `AgentSpec`
class discriminator + registry dispatch in `pydantic_to_agents`, plus an
async-construction story) — `packages/framework/` is TRUNK-CRITICAL and
READ-ONLY here, so this is a framework backlog item, filed at
[`docs/implementation/framework/v0.5-future.md`](../../implementation/framework/v0.5-future.md)
("Specialized-agent round-trip — concrete blocker"). This supersedes the
`AC-PALETTE-2` "preset" sub-clause for Session 08 and beyond.

### Tools — two distinct concepts (kept separate)

1. **Agent-attached tool** (exists; basic). The `AgentSpec` carries a
   tool list (tool registry / `AVAILABLE_TOOLS`); v0.3 has the flat
   checklist in the agent card. Deferred: a per-tool config card and an
   "add custom tool" affordance.
2. **Tool-as-node** (future). A standalone tool step in the graph —
   composed with a Logic node, a tool can be invoked deterministically
   instead of via an agent's LLM judgment. Becomes a future framework
   node kind. Same tool registry as (1), different placement; the
   abstraction must not conflate them.

## Extensibility

Adding a future deterministic node (e.g. a conditional) is, framework
side, one new `NodeKind` member + its analyzer behaviour; Spren side, one
palette type under Core/Logic + flipping its `active` flag. Because Spren
consumes the canonical `NodeSpec` directly, there is **no Spren mirror or
materializer to extend** — the regenerated client picks up the new kind.
This is why all five categories are defined now though only Agents + Core
are active in v0.3.

## What v0.3 implements

Active: the **generic Agent** node and **Core** (Start/End/User as real
`kind` nodes). This is exactly what unblocks Run (the RUN-3d fix, now
the canonical-reframe). The **specialized-agent catalog** is shown but
**inactive** (framework wire gap — P12), alongside **Tools / Logic /
Data** which are modeled but have no framework kind yet.
