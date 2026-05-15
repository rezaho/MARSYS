# Spren node model — building blocks

**Status**: design locked 2026-05-15. Supersedes the node-type description in [`02-data-model.md`](./02-data-model.md) (which described an unfaithful 4-type mirror — see "Framework ground truth" below).

## Principle

The visual-builder node palette is **Spren's own UX layer**, modeled on OpenAI's Agent Builder — NOT a 1:1 mirror of marsys primitives. Each Spren node type declares a **materialization**: the rule converting it to valid marsys runtime constructs at run time (`packages/spren/src/spren/runs/materialize.py`).

SP-005 ("Pydantic mirrors the framework exactly") continues to govern types that genuinely mirror framework structures (e.g. `ExecutionConfig`). The node palette is explicitly **not** such a mirror — it is a translation surface with a contract. This distinction is the resolution of the SP-005 concern raised during the RUN-3d review.

## Framework ground truth (why the old model was wrong)

marsys's canonical topology has exactly two node kinds, distinguished by **type** (class), not by name:

- **Agent node** — `marsys.coordination.topology.core.Node`, `node_type=AGENT`, carries `agent_ref`.
- **Deterministic node** — `DeterministicNode` subclasses (`marsys/coordination/execution/det_nodes.py`): today `StartNode`, `EndNode`, `UserNode`; future: conditional, loop, etc. `Topology.__post_init__` accepts `Node | DeterministicNode` (`core.py:170-174`) — a first-class documented input path.

`NodeType` (`core.py:20-25`) declares `{USER, AGENT, SYSTEM, TOOL}`, but **`NodeType.SYSTEM` and `NodeType.TOOL` have zero references anywhere in `packages/framework/src/marsys`** (verified by grep) — vestigial enum members. Only `AGENT` and the legacy `USER` regular-node path are real; the modern path is the `DeterministicNode` subclasses. Name-based resolution (`RESERVED_DETNODE_NAMES`) exists **only** in `parse_node`'s string branch (`converters/parsing.py:62-72`) — sugar for the human-written string DSL, not the canonical contract.

Spren's earlier `NodeType = {user, agent, system, tool}` mirror copied the vestigial members and mistook the string sugar for the model. Corrected here.

## The model

A Spren node has a **category** and a **type** within it. Each type declares a **materialization** and an **active** flag.

| Category | Types | Materializes to | v0.3 |
|---|---|---|---|
| **Agents** | generic Agent; specialized (Browser, DataAnalysis, CodeExecution, FileOperation, WebSearch, InteractiveElements, Learnable) | agent `Node(node_type=AGENT)` + agent definition carrying the agent class | **active** |
| **Core** | Start; End (Stop); User | `StartNode` / `EndNode` / `UserNode` det-node *instances* (the supported non-shim path) | **active** |
| **Tools** | (a) agent-attached tool; (b) tool-as-node | (a) entries in the agent definition's tool list; (b) future det-node | **inactive** |
| **Logic** | conditional (if/then); loop/while | future marsys `DeterministicNode` subclasses | **inactive** |
| **Data** | transform; set-state | future marsys primitive or Spren orchestration | **inactive** |

**Inactive types render in the palette as visibly disabled ("coming soon"), never droppable.** Spren must not let a builder produce a topology it cannot run — that is the RUN-3d failure class, and the reason inactive categories are shown but not buildable until their marsys primitive lands. The product owner controls activation as the framework roadmap delivers each primitive.

### Core specifics (the v0.3-concrete subset = the RUN-3d fix)

- **Start**: exactly one per canvas, present by default, non-deletable; singleton enforced in Spren by construction (marsys also enforces ≤1 — `TopologyGraph.get_start_node` raises on >1). Materializes to a `StartNode` instance.
- **End / Stop**: zero-or-more, user-added, not default. Materializes to `EndNode` instances.
- **User**: the frontend may have **multiple** User nodes (design convenience — several visual human-interaction points). At materialization they **collapse to the single canonical `UserNode`**: every edge incident to any visual User node rewires to the one `UserNode` instance. Not default; user-added.
- A runnable workflow requires every agent to reach End or User (marsys `validate_workflow()`, `graph.py:1395-1467`). Lint surfaces this pre-flight so the builder guides the user to add/wire a terminal rather than failing at run time.

### Tools — two distinct concepts (kept separate in the abstraction)

1. **Agent-attached tool** (exists; basic). The agent definition carries a tool list (backed by the tool registry / `AVAILABLE_TOOLS`). v0.3 has the flat checklist in the agent card. Deferred redesign: a **per-tool config card** (some tools, e.g. web search, have options to configure) and an **"add custom tool" affordance** anticipated in the UX now even though the backend cannot author custom tools yet.
2. **Tool-as-node** (future). A standalone tool step in the graph. Rationale: composed with a **Conditional (Logic)** node, a tool can be invoked **deterministically** (graph-driven) instead of via an agent's LLM judgment. Materializes to a future marsys det-node. Same underlying tool registry as (1), different placement (inside an agent vs as a graph node) — the abstraction must not conflate them.

## Extensibility

Adding a future framework det-node (e.g. a conditional) is: one new type under Core/Logic + one materializer dispatch entry + flip its `active` flag. No rework of the model, the Pydantic mirror, or the canvas. This is why all five categories are defined now though only Agents + Core are implemented in v0.3.

## What v0.3 implements

Concrete and active: **Agents** (existing agent node + the specialized-agent catalog as distinct palette items — see the PALETTE redesign in session 06) and **Core** (Start/End/User with the materialization above). This is exactly what unblocks Run (RUN-3d). **Tools / Logic / Data**: modeled here, rendered inactive in the palette, no materialization yet.
