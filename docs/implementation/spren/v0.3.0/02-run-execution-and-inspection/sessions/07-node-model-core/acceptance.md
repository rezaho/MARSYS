# Acceptance criteria — Session 07: Node-model building blocks + Core (Start/End/User)

Frozen at 2026-05-15T00:00:00Z. The test auditor reads ONLY this file plus the test files. Implementation source is intentionally hidden.

Context an auditor needs (no source access):
- A `WorkflowDefinition` carries a `topology` with nodes and edges. Each node has a category-based model (categories: `agents`, `core`, `tools`, `logic`, `data`) plus a type. Core types are `start`, `end`, `user`. This replaces a prior 4-value `NodeType` enum (`user`, `agent`, `system`, `tool`).
- "Materialization" = Spren translating a `WorkflowDefinition` into a marsys-framework `Topology` object used to execute a run.
- The marsys framework exposes det-node classes `StartNode`, `EndNode`, `UserNode` (each constructed with a `name=`), distinct from a plain `Node`. A correctly materialized topology registers these as det-nodes.
- A `TOPOLOGY_ERROR` (e.g. `[TOPOLOGY_ERROR] not reachable from Start`) is a framework validation failure surfaced on a run when the topology is invalid.
- The Spren linter is a pure Spren module that returns lint findings for a given workflow definition; each finding has a `LintCode`. It makes ZERO framework calls.
- v0.3 has no executable User-interaction path: a run whose agent invokes a User node fails at runtime ("no handler bound") BEFORE any provider call. The runnable shape is therefore `Start → Agent → End`. User-node materialization and collapse are still implemented and unit-testable without running.
- The framework regression baseline (per project CLAUDE.md): `pytest packages/framework/tests` collects 841, of which 764 pass.

## Functional

- AC-1: Materializing a `WorkflowDefinition` whose topology contains a Core/Start node, a Core/User node, an Agent node, and a Core/End node completes without raising an error.
- AC-2: After the AC-1 materialization, the resulting marsys `Topology` contains a `StartNode` instance, a `UserNode` instance, and an `EndNode` instance (concrete det-node instances, NOT plain `Node` objects) registered as det-nodes.
- AC-3: After the AC-1 materialization, the resulting marsys `Topology` contains exactly one `StartNode` instance, and its name equals the name Spren emitted for the Core/Start node (regression guard proving the framework legacy topology shim did not synthesize an additional Start).
- AC-4: `POST /v1/runs` on a `Start → Agent → End` visual-builder workflow (sidecar configured with a fake provider key) returns HTTP 201.
- AC-5: The run created by the AC-4 request does NOT fail with any `TOPOLOGY_ERROR`; it proceeds past topology validation and fails only at the fake provider call (the provider failure itself is out of scope and is not asserted beyond "execution reached it").
- AC-6: Materializing a single definition that contains multiple Core/User nodes produces exactly one marsys `UserNode` in the resulting topology.
- AC-7: After the AC-6 collapse, every edge that was incident to any visual Core/User node is present in the materialized topology, rewired to point at the single canonical `UserNode`, with no duplicate edges (two User nodes both edged to the same agent yield exactly one edge after collapse) and no dangling edges.
- AC-8: The Spren node model exposes no `NodeType` value `system` and no `NodeType` value `tool`; node categorization follows the category-based model (`agents|core|tools|logic|data`).
- AC-9: The generated OpenAPI schema and the generated TypeScript client regenerate successfully and contain no `system` or `tool` `NodeType` value.
- AC-10: A new workflow created via the canvas/create path contains exactly one Core/Start node by default (the seeded empty definition has a single Start).
- AC-11: Attempting to introduce a second Core/Start node into a workflow is rejected or prevented on every path that can add a node (palette drop, API create/update, Python import) — a definition never ends up with more than one Start.
- AC-12: A Core/Start node cannot be removed from the canvas: the delete button, the xyflow Backspace/Delete keycode path, and selection/edge-driven node removal all leave the Start node present (no-op for Start).
- AC-13: The reserved-name validator rejects a reserved name (`Start`, `End`, `User`) only for `agents`-category nodes; a Core node named `Start`, `End`, or `User` is accepted (PRODUCT-BUG-001 reconciled).
- AC-14: Importing a Python workflow file that declares a framework `StartNode`/`EndNode`/`UserNode`, or nodes named `Start`/`End`/`User`, produces the corresponding Core nodes (category `core`, type `start`/`end`/`user`) in the resulting Spren `WorkflowDefinition` — NOT Agent-category nodes named "Start"/"End"/"User".
- AC-15: The empty-draft predicate classifies a workflow as an empty draft only when it is a visual-builder workflow with no agent-category nodes AND no edges; a workflow consisting of only the seeded Start node (no agents, no edges) is classified as an empty draft, and a workflow with the Start node plus at least one agent node (or any edge) is NOT classified as an empty draft. This holds identically at both the list-filter site and the sweeper-deletion site.
- AC-16: The node palette renders five categories (Agents, Core, Tools, Logic, Data). Agents items and Core items are droppable and produce nodes when dropped.
- AC-17: The Tools, Logic, and Data palette categories render in a visibly disabled state ("coming soon") and cannot be dropped; attempting to drop them produces no node.
- AC-18: For a workflow definition where a non-Start/non-End node cannot reach any End or User node, the pure-Spren linter returns a finding whose `LintCode` is the distinct value `no_terminal`, with a plain-language message instructing the user to connect the node to an End or User node. Reachability used by the linter is computed from the explicit Core/Start node (not from "nodes with no incoming edges").
- AC-19: The `LintCode` type/`Literal` in the Spren lint model includes the value `no_terminal`.
- AC-20: A one-shot forward migration file exists at `packages/spren/src/spren/storage/migrations/04__*.py`. Running `MigrationsRunner` against a database seeded (via raw SQL) with a workflow row whose `definition` JSON uses the pre-Session-07 node shape leaves that row's `definition` parseable under the new node model, with correct category/type values for every node.
- AC-21: The migration file's source imports no symbol from `spren.models` (frozen-artifact discipline — verifiable by static inspection of the migration file's imports).
- AC-22: The `packages/spren/src/spren/models/lint.py` module docstring does not claim the linter wraps the framework's `TopologyGraph.validate_workflow()`; it states the linter is a pure Spren linter (Start/End/User-aware, mirroring the validate-workflow contract in logic, with no framework call).

## Non-functional

- AC-23: No file under `packages/framework/` is modified by this session. The framework regression suite (`pytest packages/framework/tests`) still collects 841 tests, of which 764 pass (baseline unchanged — any drift indicates an accidental framework touch).
- AC-24: The pure-Spren linter (`workflow_linter.py`) makes zero calls into the marsys framework — in particular it does not call `validate_workflow` / `TopologyGraph.validate_workflow()`. (Observable via the linter producing findings with no framework objects involved and no framework import/call on the lint path.)

## Out of scope

- Rich palette UX redesign (left island, vertical categories, click-to-expand detail, per-tool config cards, drag-drop polish). This session does only the minimal five-category change.
- Any Tools / Logic / Data materialization or execution — modeled and inactive only (no tool-as-node, no conditional, no transform).
- Canvas node visual design, right-rail, RUN-1 silent-swallow, RUN-3b envelope, lint-reactivity, auth.
- Making the User node executable (wiring a `CommunicationManager`/`UserNodeHandler` into Spren's run path). v0.3 has no human-in-the-loop run path; a run reaching a User node fails ("no handler bound") before any provider call. User-node execution is a separate future session — tests must NOT assert a User-containing run executes.
- Retroactively editing Session 03's frozen `acceptance.md` (AC-20 there is recorded as superseded elsewhere, not mutated). The `04__*` forward migration IS in scope (SP-006-mandated; it is not a backward-compat shim).
