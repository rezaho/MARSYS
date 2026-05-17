# Acceptance criteria — Session 08: Consume the framework canonical `WorkflowDefinition` (post-ADR-008 `NodeKind`)

Frozen at 2026-05-17T00:00:00Z. The test auditor reads ONLY this file plus the test files. Implementation source is intentionally hidden.

Context the auditor needs (self-contained): Session 08 deletes Spren's hand-rolled Pydantic wire-type mirror and its deterministic-node materializer, and makes Spren consume the marsys framework's canonical wire types via a thin re-export façade. Key external facts the criteria depend on:

- The framework canonical wire types live in `marsys.coordination.topology.serialize` (`NodeSpec`, `EdgeSpec`, `TopologySpec`, `WorkflowDefinition`), `marsys.coordination.topology.core` (`NodeKind`, `EdgeType`, `EdgePattern`), `marsys.agents.serialize` (`AgentSpec`, `MemoryRetention`), `marsys.models.serialize` (`ModelConfigSpec`, `ModelType`, `ApiProvider`), `marsys.coordination.serialize` (`ExecutionConfigSpec`, `StatusConfigSpec`, `ConvergencePolicyConfigSpec`, `TracingConfigSpec`).
- `NodeKind` has exactly the values `agent`, `start`, `end`, `user`. Legacy `system`/`tool` are removed; the framework's `NodeSpec` field validator HARD-RAISES a `ValueError` (surfaced as `pydantic.ValidationError`) on a stored `system` or `tool` value.
- The framework wire field is `NodeSpec.kind` (the old Spren field was `NodeSpec.node_type`).
- The framework resolves API credentials per-provider when `ModelConfigSpec.api_key` is `None`: `openai`→`OPENAI_API_KEY`, `openrouter`→`OPENROUTER_API_KEY`, `google`→`GOOGLE_API_KEY`, `anthropic`→`ANTHROPIC_API_KEY`, `xai`→`XAI_API_KEY`. A mapped provider with the env var absent raises `ValueError` → `pydantic.ValidationError` with a message of the form "Set the '<PROVIDER>_API_KEY' environment variable…". `openai-oauth`/`anthropic-oauth` resolve from profile files, no env var.
- A `WorkflowDefinition` with nodes but no `kind=start` node deserializes successfully and emits a `DeprecationWarning` (the framework's permissive path), it is NOT an error.
- The framework is TRUNK-CRITICAL READ-ONLY for this session. Framework regression baseline per project CLAUDE.md: `pytest packages/framework/tests` collects 841, of which 764 pass.

## Functional

- AC-1: `spren.models` re-exports the framework canonical types so that each of the following names imported from `spren.models` is the same object as the corresponding framework symbol: `NodeSpec`, `EdgeSpec`, `TopologySpec`, `WorkflowDefinition` (from `marsys.coordination.topology.serialize`); `AgentSpec`, `MemoryRetention` (from `marsys.agents.serialize`); `ModelConfigSpec`, `ModelType`, `ApiProvider` (from `marsys.models.serialize`); `ExecutionConfigSpec`, `StatusConfigSpec`, `ConvergencePolicyConfigSpec`, `TracingConfigSpec` (from `marsys.coordination.serialize`).

- AC-1b: `spren.models` re-exports `EdgeType` and `EdgePattern` as the real `Enum` types from `marsys.coordination.topology.core` (NOT the string-literal aliases from `marsys.coordination.topology.serialize`). Verifiable: `spren.models.EdgeType` and `spren.models.EdgePattern` are `enum.Enum` subclasses and are the same objects as `marsys.coordination.topology.core.EdgeType` / `EdgePattern`.

- AC-1c: `uv run --package spren python -c "import spren.models"` exits 0; the FastAPI application object constructs without error; the OpenAPI schema generates without error.

- AC-1d: The generated OpenAPI / TypeScript client for `TopologySpec` includes a `metadata` field of object type with a default (empty-object) value. This is an expected additive delta versus the pre-session Spren `TopologySpec`, not a regression.

- AC-2: Spren no longer defines its own `WorkflowDefinition`, `_validate_cross_references`, `NodeType`, `NodeCategory`, `NODE_TYPE_CATEGORY`, `CORE_NODE_TYPES`, `category_of`, `_reserved_names_are_agent_scoped`, or local `NodeSpec`/`TopologySpec`/`EdgeSpec`. Verifiable as absence: importing `NodeType`, `NodeCategory`, or `category_of` from `spren.models` fails (these names are not in `spren.models.__all__` and are not importable); and `spren.models.WorkflowDefinition` / `NodeSpec` / `TopologySpec` / `EdgeSpec` are the framework objects (per AC-1), not Spren-local classes.

- AC-2b: The Spren workflow envelope types `Workflow`, `WorkflowCreateRequest`, `WorkflowUpdateRequest`, `WorkflowListResponse`, `WorkflowProvenance` still exist and are importable from `spren.models`. The `.definition` field on the envelope types is typed as / accepts the framework `WorkflowDefinition` (composition); the envelope types are not subclasses of `WorkflowDefinition`.

- AC-3: `materialize_run` builds the runtime `Topology` via exactly one `pydantic_to_topology(definition, AVAILABLE_TOOLS, handler_registry={})` call — no separate `pydantic_to_agents` call, no hand-built `Topology`, no Spren re-implementation of `Node`/`Edge`. Observable: for a valid definition, the returned bundle's `topology` is the object returned by `pydantic_to_topology`, and each agent referenced by an agent node is registered in the framework `AgentRegistry` exactly once (no double registration).

- AC-3b: `materialize_run` builds `execution_config` via `pydantic_to_execution_config(definition.execution_config)`, then applies a Spren per-run override that sets `tracing` to a `TracingConfig` with `enabled=True`, `output_dir` = `<data_dir>/data/runs/<run_id>`, `include_message_content=True`, and sets `aggui` to an `AGGUIConfig` with `enabled=True`. Observable on the returned `execution_config`: tracing is enabled with that output directory and content inclusion; AG-UI is enabled; the non-overridden config fields (e.g. `status`, `convergence_policy`) are typed objects produced by `pydantic_to_execution_config` (not plain dicts).

- AC-3c: `RuntimeBundle` has exactly two fields: `topology` (a framework `Topology`) and `execution_config` (a framework `ExecutionConfig`). The `agents` field is DELETED — not derived, not present. Verifiable: `RuntimeBundle` instances have no `agents` attribute, and constructing/consuming a bundle does not require an agents argument. Agents remain reachable via `topology.nodes[*].agent_ref`.

- AC-3d: `materialize_run`'s keyword signature is unchanged except that the `secrets_lookup` keyword parameter is removed. `routes/runs.py` and `runs/lifecycle.py` non-test code require zero changes (façade transparency): the call site `materialize_run(definition=..., data_dir=..., run_id=...)` still works, `register_run` still consumes `bundle.execution_config`, and `_run_lifecycle` still consumes `bundle.topology`.

- AC-4: `runs/materialize.py` imports nothing from `marsys.coordination.execution.det_nodes`, and defines none of the symbols: `_materialize_node`, `_materialize_topology`, `_materialize_model_config`, `_materialize_agent`, `_materialize_status_config`, `_materialize_convergence_policy`, `_build_execution_config`, `_env_secrets_lookup`, `_CANONICAL_USER_NAME`, the `SecretsLookup` type alias, or any `secrets_lookup` parameter on `materialize_run` or any helper. No `StartNode`, `EndNode`, or `UserNode` instance is ever constructed by Spren code (these classes are never imported or instantiated anywhere in `packages/spren/`).

- AC-CRED-1: Spren imposes zero credential assumption. An API-typed agent whose `ModelConfigSpec` carries no `api_key` resolves its key via the framework per-provider env-var path: with `provider="openrouter"` the run uses `OPENROUTER_API_KEY`; with `provider="anthropic"` it uses `ANTHROPIC_API_KEY` (and analogously for the other mapped providers). Spren never reads any `SPREN_<PROVIDER>_API_KEY` variable and never pre-injects `api_key` into the spec or the framework call.

- AC-CRED-2: A genuinely missing key surfaces the framework's own `ValidationError` (message of the form "Set the '<PROVIDER>_API_KEY' environment variable…"), mapped to the existing Spren 400 error path — NOT a Spren-authored message that mentions checking a `SPREN_…` variable. The run-create / materialize-time key pre-check that previously raised when no `SPREN_…` secret was found is gone: a workflow on any provider whose standard env var is present runs even though no Spren-prefixed variable exists.

- AC-5: An end-to-end run of a canvas-shaped `WorkflowDefinition` — explicit `NodeSpec(kind="start")` → `NodeSpec(kind="agent", agent_ref=A)` → `NodeSpec(kind="end")`, edges Start→A→End, agent `provider="anthropic"` with a real dated model id (e.g. `claude-haiku-4-5-20251001`) and `ANTHROPIC_API_KEY` present in the process environment — submitted via `POST /v1/runs` reaches `status: "succeeded"` with a non-empty `final_response` and non-zero token/cost figures. (implied — confirm with user: this AC requires a live Anthropic API call; if no key is available in the test environment it can only be verified as an integration/E2E test, not a pure unit test.)

- AC-5b: The successful explicit-`kind` `Start→Agent→End` run emits NO `DeprecationWarning` about a missing Start node.

- AC-6: A canvas-shaped definition with no explicit Start node still round-trips through the API successfully (create + retrieve), exercising the framework's permissive-deserialize behaviour with its `DeprecationWarning` — it is NOT rejected. Spren adds no stricter rejection of its own for the missing-Start case.

- AC-7: A node or agent named `user`, `system`, or `tool` is accepted at the API boundary (workflow create/update succeeds). No Spren reserved-name validator exists post-façade, and the framework `NodeSpec` imposes no such name restriction.

- AC-8a: A new migration file exists at `packages/spren/src/spren/storage/migrations/05__migrate_node_kind.py`. It exports a module-level callable with the exact signature `def upgrade(conn: sqlite3.Connection) -> None`.

- AC-8b: `05__migrate_node_kind.py` opens no transaction of its own (it issues no `BEGIN`/`COMMIT`/`conn.commit()`; the runner wraps each migration in its own transaction). It imports nothing from `spren.models` (frozen-artifact discipline — operates on plain dicts/JSON only).

- AC-8c: When run by `MigrationsRunner` over stored `workflows.definition` rows, the migration, per node: renames the JSON key `node_type` to `kind`; maps a legacy value of `system` or `tool` to `agent`; leaves `agent`/`start`/`end`/`user` values intact under `kind`; drops no node and no edge (Start/End/User nodes survive as `kind` nodes); and bumps the row's `definition_version`.

- AC-8d: A frozen-baseline test seeds three row shapes — (a) a pre-`04` row whose nodes use `node_type` with `system`/`tool` values, (b) a Session-07 row whose nodes use `node_type` with values in `{agent,start,end,user}`, (c) a row where the `04` remap was already applied — runs `MigrationsRunner`, and asserts that every resulting row parses as the framework `WorkflowDefinition` with the correct `kind` values and with no node/edge dropped (Start/End/User det-nodes preserved).

- AC-9: Spren's Python importer (`importers/python_workflow.py`) emits framework `NodeSpec(kind=…)` values: a Start/End/User construct in the imported Python maps to a `NodeSpec` with the matching `kind` (`start`/`end`/`user`), NOT to an agent node named "Start"/"End"/"User". The importer imports no symbol from `det_nodes` and writes no `node_type` key.

- AC-10: The pure-Spren workflow linter (`lint/workflow_linter.py`) makes zero framework calls (it does not call `validate_workflow` or any other framework function). It reads the framework `NodeSpec.kind` field (its entry-node concept is a `kind="start"` node or the no-incoming-edge fallback). The module docstring of `models/lint.py` states it is a pure Spren linter that makes no framework call.

- AC-11a: Running `pnpm --filter @marsys/spren-web generate:types` regenerates the client such that the `NodeSpec` type exposes a `kind` field whose type is the union `"agent" | "start" | "end" | "user"`. The generated client contains no `node_type` field on `NodeSpec` and no `system` or `tool` member in the node-kind union.

- AC-11b: Every frontend read/write site is migrated from `node_type` to `kind`: `routes/workflows/$workflowId.tsx` (`workflowToReactFlow`, `reactFlowToWorkflow`, node add/insert paths, agent-form gate), `routes/workflows/index.tsx`, `components/WorkflowSnapshotAccordion/WorkflowSnapshotAccordion.tsx`, `lib/pattern-presets.ts`, `routes/workflows/-canvas/Palette.tsx`, `routes/workflows/-canvas/CanvasNode.tsx`. Verifiable as absence: no `node_type` reference remains in the web source for workflow node shape.

- AC-11c: The web TypeScript typecheck (`tsc` / the project's web typecheck command) passes after the regeneration and migration.

- AC-11d: A new canvas seeds exactly one Start node by default; that default Start node is non-deletable (it cannot be removed via the delete button, the xyflow delete key, or selection/edge-driven removal); the canvas allows zero or more (multiple) User nodes.

- AC-PALETTE-1: The palette presents a category model with these categories: **Agents** (an expandable group containing the standard Agent plus the specialized catalog: Browser, Code, DataAnalysis, FileOperation, WebSearch, InteractiveElements, Learnable, with room for future ones e.g. Guardrail); **Core** (Start — exactly one, present by default, non-deletable; End — 0..N user-added; User — 0..N); **Logic** (e.g. if/else, while — rendered as "coming soon" / non-droppable, no framework counterpart); **Tools / Data** (modeled-inactive). Start/End/User remain real `kind` nodes (droppable Core items), not a metadata affordance.

- AC-PALETTE-2: Specialized-agent palette items are frontend authoring presets only: dropping one produces a generic `kind="agent"` node plus a tool-/instruction-templated `AgentSpec`. There is no wire-level discriminator field for specialized-ness (no `agent_class`/`agent_type`/`kind` field on `AgentSpec` distinguishing Browser/Code/etc.); the framework round-trips such agents to a generic `Agent`. This is a documented v0.3 limitation.

- AC-PALETTE-3 (scope boundary — Out of scope marker): Session 08 ships only the minimal palette change required by the wire reframe (categories present; Start/End/User as real `kind` nodes; default non-deletable Start in the new-canvas seed; multiple User allowed). The full specialized-agent card/detail UX is task #21 and is OUT OF SCOPE for this session — a test asserting full card UX here would be wrong.

- AC-12a: After the session's changes, `pytest packages/framework/tests` still collects 841 tests with 764 passing (the baseline per project CLAUDE.md is unchanged).

- AC-12b: `git diff` for this session shows zero files under `packages/framework/` (no framework file is modified, added, or deleted).

- AC-13a: `docs/architecture/spren/11-node-model.md` is rewritten to the post-ADR-008 reality. Its product-design content is preserved: the AC-PALETTE category model, the Start singleton/default rule, the Tools/Logic/Data modeled-inactive treatment, and extensibility all still appear. Its "framework ground truth" section is replaced to describe `NodeKind` / `NodeSpec.kind` / the analyzer-seam materialization (no DeterministicNode-subclass-as-input model), states that Spren consumes the canonical `NodeSpec` directly, states that categories are frontend presentation, and adds a first-class section recording the specialized-agent-is-a-frontend-preset constraint (AC-PALETTE-2 / P12).

- AC-13b: `docs/architecture/spren/02-data-model.md`'s node description uses `kind` rather than `node_type`.

- AC-13c: `docs/architecture/spren/07-node-model-core.md` and the prior `08` session-plan body each carry a SUPERSEDED banner. (The current Session 08 plan body itself is the live plan, not superseded.)

- AC-13d: `docs/implementation/spren/v0.3.0/02-run-execution-and-inspection/sessions/06-ui-systematic-audit.md` re-records RUN-3d's resolution as "consume framework canonical post-ADR-008" and marks RUN-3d, RUN-3e, PALETTE-Start-End, PRODUCT-BUG-001, and RUN-2 as superseded/dissolved as appropriate.

- AC-13e: The committed migration file `packages/spren/src/spren/storage/migrations/04__remap_node_model.py` is left in place (it is NOT deleted — deleting a tracked migration would corrupt the applied-ids ledger on databases that ran it; `05` supersedes its effect).

## Agent keying (canvas ↔ framework bind)

- AC-AGENTKEY-1 [added 2026-05-17 — discovered in Phase B: the canvas keyed
  `agents` by a random `agent_<rand>` id while the framework
  `pydantic_to_topology` binds `node.agent_ref` against `AgentSpec.name`
  (serialize.py:503,509); without this, every canvas-built workflow hydrates
  with unbound agents. User-approved scope addition]: The canvas serializes
  `agents` keyed by the agent's name and sets each agent node's `agent_ref`
  to that same name (no `agent_<rand>` id scheme remains in the stored
  definition). A second agent given an already-used name is disambiguated to
  a unique name (collision dedupe), and the node's `agent_ref` tracks the
  deduped name.
- AC-AGENTKEY-2 [added 2026-05-17]: A forward migration re-keys every stored
  `workflows.definition`'s `agents` dict by `AgentSpec.name`, rewrites each
  `node.agent_ref` from the old id to the corresponding name, dedupes name
  collisions deterministically, drops no node/edge/agent, and bumps
  `definition_version`; pure-dict, no `spren.models` import; module-level
  `def upgrade(conn)`, runner-owned transaction. A frozen-baseline test seeds
  an id-keyed row (incl. a name-collision case) and asserts the migrated row
  parses as the framework `WorkflowDefinition` AND that
  `pydantic_to_topology` binds every agent node to a live `Agent`.

## Non-functional

- AC-NF-1 (correctness / data preservation): The `05` migration preserves all nodes and edges of every stored workflow definition across all seeded row shapes — no Start/End/User node and no edge is dropped (this is the SP-017-spirit no-data-loss guarantee, covered observably by AC-8c/AC-8d).

- AC-NF-2 (idempotence / ledger integrity): Re-running migrations does not re-apply `04` or `05` to a database that already recorded them (the runner tracks applied ids); a database that already ran the old `04` is not silently no-op'd by the new `05` because `05` uses a distinct prefix. Verifiable via AC-8d seed (c) plus a second runner invocation producing no further change.

- AC-NF-3 (error mapping): Materialization failures that remain after the reframe (unknown tool, unknown handler, framework validation error including the missing-credential `ValidationError`) are wrapped/mapped to the existing Spren error path (e.g. `MaterializationError` → HTTP 400), not surfaced as an unhandled 500. (Covered for the credential case by AC-CRED-2.)

## Out of scope

(Tests asserting behaviour for these in this session would be wrong.)

- RUN-1 (silent run-create swallow), RUN-3b (structured error envelope), LINT-REACTIVITY, UI-BUG-007 / ARCH-Q-001 auth, SWEEPER-1, the palette/node/right-rail visual redesign (#21–#24), the nit sweep (#18).
- RUN-3e (delivery-contract instruction suffix) — still a real open architecture decision but orthogonal; AC-5 reaching `succeeded` is only the diagnostic for whether it still reproduces. Do not assert suffix behaviour here.
- Making the User node executable (human-in-the-loop run path). v0.3 has none; a run that reaches a User node is expected to fail at the framework before any provider call. The canvas may place/lint User nodes and they round-trip as `kind="user"`, but executable-User behaviour is not tested here.
- v0.4 credential-store seam (keychain / encrypted-SQLite). v0.3 uses the framework standard env-var path only.
- Any `packages/framework/` change (the `05` migration is in scope; it is an SP-006-mandated forward migration, not a back-compat shim).
- Adding sidecar `.env` auto-load. The framework does not auto-load `.env`; this session only updates `scripts/`, the `just dev` banner, and onboarding docs to document/export the requirement — no `.env`-autoload code feature is added or tested.

## Open / needs clarification

- AC-5 (and AC-5b) require a live Anthropic API call with a valid `ANTHROPIC_API_KEY` and a real dated model id. As written it is only verifiable as an integration/E2E test against the live provider; it cannot be a hermetic unit test without contradicting AC-CRED (which forbids Spren injecting/mocking the key in product code) and SP-007 (no mocked in-codebase features in product code).

  **[resolved 2026-05-17]** Split verification, consistent with SP-007 and the prior Session-07 AC-2 pattern (fake-key sidecar isolates the topology fix from the provider call):
  - **Hermetic portion (default suite, must pass in CI):** an integration test submits the explicit-`kind` `Start→Agent→End` definition via `POST /v1/runs` against a fake-key sidecar and asserts it returns 201, reaches the provider call, fails ONLY there (no `TOPOLOGY_ERROR`, no missing-Start `DeprecationWarning` → AC-5b), and that materialization + topology validation succeeded. This is the RUN-3d/reframe regression gate.
  - **Live portion (real-system, gated OUT of the default suite):** the `succeeded` + non-empty `final_response` + non-zero cost half is verified by `scripts/scenarios/run_failure_probe.py` run against a sidecar with a real per-provider key exported. Not part of hermetic `pytest`; run as the manual/E2E acceptance step.

  This is a clarification of *how* AC-5 is verified (the plan's Testing-strategy already frames the probe as the real-system gate), not a change to its intent. Surfaced to the user in the Phase-A synthesis; the user may escalate to "live key in CI" if desired.
