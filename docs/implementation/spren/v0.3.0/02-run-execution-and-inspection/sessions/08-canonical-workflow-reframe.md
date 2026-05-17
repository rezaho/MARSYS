# Session 08: Consume the framework canonical `WorkflowDefinition` (post-ADR-008 `NodeKind`) — delete Spren's mirror + det-node materializer

> **Compaction-survival doc — written 2026-05-17, ground-up rewrite.** This *fully
> supersedes* the prior `08` (which was premised on the pre-ADR-008 framework and
> is now invalidated — see §0) and `07-node-model-core.md` (the det-node-instance
> approach, also invalidated). Self-contained so a post-compaction session resumes
> cold. Provenance tags: **[V]** verified in primary source THIS session (file:line
> I read) · **[E]** Explore-agent-reported, the plan-validator + implementer MUST
> re-confirm at code depth before relying on it · **[A]** architect/product design
> assertion · **[U]** user decision (see §8).

---

## 0. The decisive finding — why BOTH prior node-model attempts were wrong

Three node-model designs have existed for Spren. Two shipped wrong; the framework
just made the third one correct.

| Attempt | Mechanism | Why it is wrong now |
|---|---|---|
| **Session 07** (`07-node-model-core.md`) | Spren emits `StartNode`/`EndNode`/`UserNode` *instances* into `Topology.nodes` | **[V]** Post-ADR-008 `Topology.__post_init__` accepts only plain `Node` (rejects `DeterministicNode` subclass instances [E]); and `Topology`/`pydantic_to_topology` no longer have `NodeType` — Spren's `materialize.py:48` `from ...core import NodeType` is a hard `ImportError` against the rebased tree |
| **Prior Session 08** (the doc this replaces) | Spren emits agent-only nodes + `TopologySpec.metadata={entry_point,exit_points}`; rely on `orchestra._apply_legacy_topology_shim` to synthesize Start/End | **[V]** That shim is now the *deprecated* path: `WorkflowDefinition._validate_cross_references` emits a `DeprecationWarning` whenever there is **no explicit `kind=START` node** (`serialize.py:229-239`) and the shim is *removed in v0.4*. Building Spren on it ships dead-on-arrival code |
| **This Session 08** (correct) | Spren keeps Start/End/User as **real** canvas + wire nodes; serializes them as `NodeSpec(kind="start"\|"end"\|"user")`; consumes the framework's canonical `pydantic_to_topology` | **[V]** This is now the framework's blessed durable representation — proven runnable end-to-end by `packages/framework/examples/session08_workflow_roundtrip.py` (explicit `kind="start"`/`"end"` nodes → JSON → disk → reload → `pydantic_to_topology(reloaded, {}, {})` → `Orchestra.run` → `assert result.success is True`) |

**The framework change that flipped this (ADR-008 / framework "Session 08", now on
`main`, mirrored in this worktree post-rebase — full cited record in §9):** the node
taxonomy was unified. `NodeType` → `NodeKind`; members are exactly
`{AGENT, START, END, USER}` (`SYSTEM`/`TOOL` deleted as vestigial); the wire field
`NodeSpec.node_type` → `NodeSpec.kind`; `WIRE_SCHEMA_VERSION = 2`; deterministic
behaviour is materialized **from `Node.kind` at the analyzer seam**, never stored as
a subclass instance in `Topology.nodes` (ADR-008 "Option A"). `workflow_to_pydantic`
is now *total* — every `NodeKind` round-trips 1:1. There is **no dual-field shim**:
the version boundary is the contract.

**Net effect on Spren:** the canvas keeps explicit Start/End/User nodes (this was the
user's instinct all along, and is now exactly what the framework supports). Spren's
job is no longer to invent a node model or to fight the framework — it is to **stop
maintaining a hand-rolled mirror** and consume the framework's canonical wire types.
The SP-005 thesis of the prior `08` was right and survives; everything it built
*around* the deprecated metadata-shim path is deleted.

**Process root cause (binding lesson, recorded so we actually change behaviour):**
across Session 07, RUN-3d, and the prior Session 08, agents — including independent
verifiers — repeatedly produced confident, internally-consistent conclusions about
the framework's node contract by *code-tracing*, and each was wrong at a different
layer. The only thing that ever produced the truth was a **canonical working
reference run** (`session08_workflow_roundtrip.py` / the GAIA canonical test). The
discipline this enforces, per CLAUDE.md → Grounding: a premise about an external
contract is `UNVERIFIED` until confirmed against a *primary source you read this
session* or a *canonical example that actually runs* — an upstream agent's report
(including a handoff doc or this plan) is `agent-brief`, never a citation. This plan
tags every contract claim accordingly; the plan-validator must treat **[E]** items
as `UNVERIFIED` and re-derive them from source.

---

## 1. Premises this plan rests on

The frame-check target is **P0**. If P0 is false the whole plan is wrong; the
validator must look for the falsifier named in P0 before building on it.

- **P0 [V] — The correct durable representation of Start/End/User is an explicit
  `NodeSpec(kind=...)` consumed via `pydantic_to_topology`; the metadata-shim path
  and the det-node-instance path are both dead.** Falsifier to look for: a code path
  where `pydantic_to_topology` produces anything other than a plain `Node`, OR where
  an explicit `kind="start"` node triggers a deprecation/rejection. Verified absent:
  `serialize.py:471-552` materializes every `NodeSpec` as `Node(name, kind, agent_ref=runtime_binding, …)`;
  `serialize.py:229-239` deprecates only the *no-explicit-Start* case;
  `examples/session08_workflow_roundtrip.py` runs explicit `kind` det-nodes to
  `success`.
- **P1 [V]** Framework owns the canonical wire shape. `marsys.coordination.topology.serialize`
  exports `NodeSpec` (`kind: NodeKind = AGENT`, `agent_ref`, `is_convergence_point`,
  `metadata`; `model_config = extra="forbid"`; `@field_validator("kind", mode="before")
  _reject_removed_node_types` raises on `system`/`tool`), `EdgeSpec`, `TopologySpec`
  (`nodes`, `edges`, `metadata`, `rules`), `WorkflowDefinition`
  (`topology`/`agents`/`execution_config`, `extra="forbid"`,
  `_validate_cross_references` model-validator), `pydantic_to_topology(spec,
  tool_registry, handler_registry=None) -> Topology`, `workflow_to_pydantic`,
  `topology_equals`, `workflow_definition_schema`, `WIRE_SCHEMA_VERSION = 2`.
  — `serialize.py:59-71,119-261,471-552` (read this session).
- **P2 [V]** `NodeKind` (values `agent|start|end|user`) lives at
  `marsys.coordination.topology.core`; `RESERVED_NODE_NAMES` is derived from it as
  `frozenset(k.value for k in NodeKind if k is not AGENT)` → `{start,end,user}`
  (value-set FLIPPED from the pre-S08 `{user,system,tool}`). — `core.py:20-41`.
- **P3 [E]** `pydantic_to_topology` internally calls `pydantic_to_agents(spec,
  tool_registry)` (`serialize.py:500-503`), constructs each agent **once**, binds it
  to `Node.agent_ref`; `USER` nodes resolve `metadata["handler"]` via
  `handler_registry` (absent key → `UnknownHandlerError`; no key → framework
  process-wide `UserNodeHandler`). Implication: Spren must NOT also call
  `pydantic_to_agents` separately (double-construction / double `AgentRegistry`
  registration — the prior plan's P8, still live).
- **P4 [V]** `pydantic_to_agents(spec: WorkflowDefinition, tool_registry) ->
  List[Any]` (NOT `List[Agent]` — `agents/serialize.py:131`); it calls
  `runtime_model_config_from_spec(agent_spec.agent_model)` **positionally with one
  arg** (`agents/serialize.py:156`) — there is NO explicit `api_key=None,
  runnable=True` kwarg at the call site; runnable behaviour is the parameter
  default. The implementer must not lean on an explicit-kwarg contract that the
  code does not have.
- **P5 [E]** `pydantic_to_execution_config(spec: ExecutionConfigSpec) -> ExecutionConfig`
  exists (`coordination/serialize.py:280-304`); `ExecutionConfigSpec` carries full
  `status: StatusConfigSpec` + `tracing: TracingConfigSpec` + `convergence_policy`.
  This replaces Spren's hand-rolled `_build_execution_config` /
  `_materialize_status_config` / `_materialize_convergence_policy`.
- **P6 [V]** `materialize_run(*, definition, secrets_lookup=None, enable_aggui=True,
  data_dir=None, run_id=None) -> RuntimeBundle` is called at `routes/runs.py:158-162`
  with only `definition=`, `data_dir=`, `run_id=`. `register_run` consumes
  `bundle.execution_config` (→ `Orchestra(execution_config=…)`, `lifecycle.py:186-190`)
  and `_run_lifecycle` consumes `bundle.topology` (`lifecycle.py:303-307`).
  Keeping the `materialize_run` keyword signature + `RuntimeBundle` field names means
  `routes/runs.py` and `lifecycle.py` need **zero** change (façade transparency).
  Whether `bundle.agents` is consumed anywhere is a known-unknown (see §7).
- **P7 [V]** Spren currently duplicates the framework: its own `WorkflowDefinition`
  + identical `_validate_cross_references` (`models/workflow.py:29-61`), its own
  `NodeType`/`NodeCategory`/`NODE_TYPE_CATEGORY`/`CORE_NODE_TYPES`/`category_of` +
  `NodeSpec.node_type` + `_reserved_names_are_agent_scoped` validator
  (`models/topology.py:27-123`), and a det-node-emitting materializer
  (`runs/materialize.py:104-181`). All of this is the divergence SP-005 forbids and
  is now also broken against the rebased framework (P0/§0).
- **P8 [V]** `packages/framework/` is TRUNK-CRITICAL READ-ONLY. Every change is in
  `packages/spren/`, `apps/web/`, `docs/`, `scripts/`. Framework regression baseline
  (per CLAUDE.md: `pytest packages/framework/tests` collects 841, 764 pass) MUST NOT
  move — drift means an accidental framework touch.
- **P9 [V]** The committed `04__remap_node_model.py` only remaps `node_type`
  `system|tool → agent` in place; it does **not** rename `node_type → kind` and does
  **not** touch `start`/`end`. The runner records applied ids and skips them
  (`runner.py:42-46`); reusing prefix `04` on any DB that ran the old `04` is a
  silent no-op. The new migration MUST be `05__*`.
- **P10 [E]** `det_nodes.py` still exists and exports `StartNode/EndNode/UserNode/
  DeterministicNode` + `NODE_KIND_BEHAVIOUR`/`behaviour_for_kind`, but they are now
  *internal analyzer behaviour classes* keyed by `NodeKind`, NOT inputs to
  `Topology.nodes`. Spren must import none of them. — Explore-reported
  `det_nodes.py:36-220`, `analyzer.py:172-223`; implementer confirms by deleting
  Spren's import and seeing it compile via `pydantic_to_topology`.
- **P11 [V] — Credential resolution is the framework's, per-provider, and Spren
  imposes ZERO assumption.** `ModelConfig._validate_api_key` (`models/models.py:209-263`)
  resolves, when `api_key is None`, from a hard-coded per-provider map
  (`models.py:218-224`): `openai→OPENAI_API_KEY`, `openrouter→OPENROUTER_API_KEY`,
  `google→GOOGLE_API_KEY`, `anthropic→ANTHROPIC_API_KEY`, `xai→XAI_API_KEY`;
  `openai-oauth`/`anthropic-oauth` resolve from profile files via
  `OAuthCredentialStore` (no env var); a mapped provider with absent var raises
  `ValueError` → surfaced as `pydantic.ValidationError`. The framework does **not**
  auto-load `.env` (`os.getenv` only — no `dotenv` in the model/adapter path); the
  process must have exported the var. Spren's *entire* single-var assumption is one
  function: `_env_secrets_lookup` (`materialize.py:90-93`) + the api_key inject
  (`:207`) + the missing-key raise (`:202-206`) + the `SecretsLookup` alias (`:69`)
  + kwarg threading (`:187/:216/:345/:363/:369`) + the route's default-reliance
  (`routes/runs.py:158-162`, which never passes `secrets_lookup`). Deleting all of
  it ⇒ `pydantic_to_topology`→`pydantic_to_agents`→`runtime_model_config_from_spec(spec)`
  (no api_key) ⇒ framework per-provider resolution. **This dissolves WF-BUG-RUN-2**
  (its gate *is* the `:202-206` raise; with a normal `.env` carrying standard
  unprefixed names it currently 400s for every provider).
- **P12 [V] — "Specialized agent" is NOT a wire concept; it is a frontend
  authoring preset.** Framework `AgentSpec` (`agents/serialize.py:50-68`,
  `extra="forbid"`) has no `agent_class`/`agent_type`/`kind` discriminator;
  `agent_to_pydantic` never records `type(agent)`; `pydantic_to_agents` hard-codes
  reconstruction to base `Agent` (`agents/serialize.py:158`). `BrowserAgent →
  AgentSpec → Agent` round-trips **lossily** to a generic `Agent` (only tool
  *names* survive, and only if the consumer `tool_registry` maps them). Therefore
  Spren's palette "specialized agent" items (Browser, Coding, … future Guardrail)
  must compile to a generic `kind="agent"` node + a tool-/instruction-templated
  `AgentSpec` — never a Spren wire field the framework lacks (SP-005/SP-018). The
  subclass-`__init__`-only behaviours (e.g. `headless`, `search_mode`, vision
  config) are NOT expressible in v0.3; making specialized-ness a wire concept is a
  *framework* backlog item, not Spren's to invent.

---

## 2. Goal

Delete Spren's hand-rolled Pydantic mirror **and** its det-node materializer; make
Spren **consume the framework canonical** wire types (`marsys.*.serialize`) through a
thin `spren.models` re-export façade, and rewrite `materialize_run` to a single
`pydantic_to_topology(definition, AVAILABLE_TOOLS, handler_registry={})` call +
`pydantic_to_execution_config` + a Spren-only per-run tracing/AG-UI override. The
visual builder keeps **real Start / End / User nodes** that serialize as
`NodeSpec(kind=…)`. Net: a canvas-built `Start → Agent → End` workflow runs
end-to-end through the framework's blessed path, the mirror-divergence failure class
becomes structurally impossible (Spren no longer owns the shape), and both prior
node-model attempts are retired.

---

## 3. Acceptance criteria

- **AC-1** `spren.models` re-exports the framework canonical types: `NodeSpec`,
  `EdgeSpec`, `TopologySpec` from `marsys.coordination.topology.serialize`;
  `WorkflowDefinition` from the same; `AgentSpec`/`MemoryRetention` from
  `marsys.agents.serialize`; `ModelConfigSpec`/`ModelType`/`ApiProvider` from
  `marsys.models.serialize`; `ExecutionConfigSpec`/`StatusConfigSpec`/
  `ConvergencePolicyConfigSpec`/`TracingConfigSpec` from `marsys.coordination.serialize`;
  and the runtime enums `EdgeType`/`EdgePattern` from
  **`marsys.coordination.topology.core`** (NOT `…serialize`, which has only
  `EdgeTypeLiteral`/`EdgePatternLiteral` string literals — `serialize.py:102-103`;
  the importer's enum-identity/isinstance/attribute-allowlist depends on these
  being real `Enum`s — `python_workflow.py:39-41,774`). `uv run --package spren
  python -c "import spren.models"` succeeds; the FastAPI app builds; OpenAPI
  generates. **Known wire/OpenAPI delta to expect (not a regression):** framework
  `TopologySpec` adds `metadata: Dict[str,Any]` that Spren's local one lacked
  (`serialize.py:196` vs old `topology.py:144-149`) — the generated-TS diff will
  show this new field; it is additive + default-factory, allowed.
- **AC-2** Spren's own `WorkflowDefinition` + `_validate_cross_references`
  (`models/workflow.py`) and its `NodeType`/`NodeCategory`/`NODE_TYPE_CATEGORY`/
  `CORE_NODE_TYPES`/`category_of`/`_reserved_names_are_agent_scoped`/local
  `NodeSpec`/`TopologySpec`/`EdgeSpec` (`models/topology.py`) are **deleted** (not
  re-exported under old names that mirror framework structures). The Spren workflow
  *envelope* types (`Workflow`, `WorkflowCreateRequest`, `WorkflowUpdateRequest`,
  `WorkflowListResponse`, `WorkflowProvenance`) are kept and point `.definition` at
  the framework `WorkflowDefinition` (composition, no subclass).
- **AC-3** `materialize_run` builds the runtime `Topology` via exactly **one**
  `pydantic_to_topology(definition, AVAILABLE_TOOLS, handler_registry={})` call (no
  separate `pydantic_to_agents` call, no hand-built `Topology`, no `Node`/`Edge`
  re-implementation); `execution_config` via `pydantic_to_execution_config(definition.execution_config)`
  then the Spren per-run override sets `tracing = TracingConfig(enabled=True,
  output_dir=<data_dir>/data/runs/<run_id>, include_message_content=True)` and
  `aggui = AGGUIConfig(enabled=True)`. No `Agent` is constructed twice / registered
  twice in `AgentRegistry`. **`RuntimeBundle` is reduced to `{topology,
  execution_config}` — the `agents` field is deleted, NOT derived** (improver:
  zero production readers — `lifecycle.deregister` only pops `_active_runs`, never
  reads `bundle.agents`; its docstring rationale is dead; dropping it removes the
  double-registration risk class entirely; agents remain reachable via
  `topology.nodes[*].agent_ref`). `materialize_run`'s keyword signature (minus the
  deleted `secrets_lookup` kwarg the route never passed) is otherwise unchanged →
  `routes/runs.py:158-162` + `lifecycle.py` non-test code byte-identical.
- **AC-4** `materialize.py` imports nothing from
  `marsys.coordination.execution.det_nodes` and contains no
  `_materialize_node`/`_materialize_topology`/`_materialize_model_config`/
  `_materialize_agent`/`_materialize_status_config`/`_materialize_convergence_policy`/
  `_build_execution_config`/`_env_secrets_lookup`/`_CANONICAL_USER_NAME`, no
  `SecretsLookup` type alias, and no `secrets_lookup` parameter on `materialize_run`
  or any helper; no `StartNode`/`EndNode`/`UserNode` instance is ever constructed
  by Spren.
- **AC-CRED** Spren imposes **zero credential assumption**. An API-typed agent
  whose `ModelConfigSpec` carries no `api_key` resolves its key via the framework's
  per-provider env-var path (P11) — `provider="openrouter"` reads
  `OPENROUTER_API_KEY`, `"anthropic"` reads `ANTHROPIC_API_KEY`, etc.; Spren never
  reads `SPREN_<PROVIDER>_API_KEY` and never pre-injects `api_key`. A genuinely
  missing key surfaces the framework's own `ValidationError` ("Set the
  '<PROVIDER>_API_KEY' environment variable…") at materialize time, mapped to the
  existing Spren 400 path — NOT a Spren "checked SPREN_… env var" message. The
  WF-BUG-RUN-2 run-create pre-check (the `materialize.py:202-206` raise) is gone:
  a workflow on any provider whose standard env var is present runs without a
  Spren-prefixed variable existing.
- **AC-5** End-to-end: a canvas-shaped `WorkflowDefinition` — explicit
  `NodeSpec(kind="start")` → `NodeSpec(kind="agent", agent_ref=A)` →
  `NodeSpec(kind="end")` with edges Start→A→End, `provider="anthropic"`, a real
  dated model id (e.g. `claude-haiku-4-5-20251001`), `ANTHROPIC_API_KEY` in env —
  run via `POST /v1/runs` reaches `status:"succeeded"` with non-empty
  `final_response` and non-zero tokens/cost, and emits **no** `DeprecationWarning`
  about a missing Start node. (Probe: `scripts/scenarios/run_failure_probe.py`
  updated to the explicit-`kind` shape + dated model id.) **Verification split
  (acceptance.md, resolved 2026-05-17):** the hermetic suite asserts the run
  reaches the provider call with no `TOPOLOGY_ERROR` / no missing-Start
  `DeprecationWarning` against a fake-key sidecar (the regression gate, in CI);
  the `succeeded`+cost half is the live probe with a real per-provider key, gated
  OUT of the default `pytest` suite (SP-007; Session-07 AC-2 precedent).
- **AC-6** A canvas-shaped definition with **no** explicit Start node still
  round-trips through the API (it is the framework's permissive-deserialize +
  `DeprecationWarning` case, not an error) — i.e. Spren does not add its own
  stricter rejection; the framework owns that policy.
- **AC-7** PRODUCT-BUG-001 dissolves: a node or agent named `user`/`system`/`tool`
  is accepted at the API boundary (no Spren reserved-name validator exists
  post-façade; the framework's `NodeSpec` has none).
- **AC-8** New one-shot forward migration `packages/spren/src/spren/storage/migrations/05__migrate_node_kind.py`,
  exporting a module-level `def upgrade(conn: sqlite3.Connection) -> None` (the
  exact entrypoint name `runner.py:80-90` requires) that opens **no** transaction
  of its own (the runner wraps every migration in `BEGIN`/`COMMIT` —
  `runner.py:65-77`); pure-dict, **no `spren.models` import** (frozen-artifact
  discipline). Per stored `workflows.definition`, per node: rename key `node_type
  → kind`; map a legacy `system`/`tool` value → `agent` — this is **mandatory for
  correctness, not defensive**: the framework `_reject_removed_node_types`
  validator (`serialize.py:143-166`) HARD-RAISES on a stored `system`/`tool`, so a
  never-`04` DB is unloadable without this mapping; leave `agent`/`start`/`end`/
  `user` intact under `kind`; **no node/edge dropped** (Start/End/User survive as
  `kind` nodes); bump `definition_version`. Frozen-baseline test seeds (a) a
  pre-`04` row (`node_type` w/ `system`/`tool` — load-bearing, not an edge case),
  (b) a Session-07 row (`node_type ∈ {agent,start,end,user}`), (c) a
  `04`-already-applied row; runs `MigrationsRunner`; asserts every row parses as
  the framework `WorkflowDefinition` with correct `kind` and no dropped det-nodes.
- **AC-9** Spren's Python importer (`importers/python_workflow.py`) emits framework
  `NodeSpec(kind=…)` (Start/End/User → the matching `kind`, not an Agent named
  "Start"); it imports no `det_nodes` symbol.
- **AC-10** The pure-Spren linter still makes **zero** framework calls (SP-018);
  it reads the framework `NodeSpec.kind`; `models/lint.py`'s docstring states the
  truth (pure Spren linter, no `validate_workflow` call).
- **AC-11** Frontend consumes the regenerated client: `pnpm --filter
  @marsys/spren-web generate:types` produces `NodeSpec.kind` with `NodeKind =
  "agent"|"start"|"end"|"user"` (no `node_type`, no `system`/`tool`); every read/
  write site is migrated (`$workflowId.tsx` `workflowToReactFlow`/`reactFlowToWorkflow`,
  `index.tsx:149`, `WorkflowSnapshotAccordion.tsx:113,124`, `pattern-presets.ts:98`,
  `Palette.tsx`, `CanvasNode.tsx`); the canvas keeps a default non-deletable Start
  and allows multiple User nodes; web typecheck passes.
- **AC-12** Framework regression baseline unchanged (`packages/framework/tests`
  841/764 per CLAUDE.md); `git diff` shows **zero** `packages/framework/` files.
- **AC-PALETTE** The palette **category model is preserved and locked** (it kept
  getting lost to compaction; this AC is its durable home until `11-node-model.md`
  carries it). The model: **Agents** (an OpenAI-Agent-Builder-style expandable
  group: the standard Agent + today's specialized catalog [Browser, Code,
  DataAnalysis, FileOperation, WebSearch, InteractiveElements, Learnable] + a place
  for future ones e.g. Guardrail) · **Core**: Start (exactly one, present by
  default in a new canvas, non-deletable), End (0..N, user-added), User (0..N) ·
  **Logic** (if/else, while — modeled, rendered "coming soon"/non-droppable, no
  framework counterpart yet) · **Tools / Data** (modeled-inactive per the existing
  5-category design). Per **P12**, specialized-agent items are **frontend
  authoring presets** that compile to a generic `kind="agent"` node + a
  tool-/instruction-templated `AgentSpec` — there is no wire discriminator and the
  framework round-trips them to a generic `Agent` (a documented v0.3 limitation).
  **Session 08 ships only the MINIMAL palette change** required by the wire
  reframe: categories present, Start/End/User remain real `kind` nodes, a default
  non-deletable Start in `new.tsx`'s seed, multiple User allowed. The full
  specialized-agent card/detail UX is **task #21** (unchanged), now with this model
  as its binding spec.
- **AC-13** Doc reconciliation (not deletion): `docs/architecture/spren/11-node-model.md`
  is rewritten to the post-ADR-008 reality — its product design (the AC-PALETTE
  category model, Start singleton/default, Tools/Logic/Data modeled-inactive,
  extensibility) is **preserved**; only its "framework ground truth" section is
  replaced (DeterministicNode-subclasses → `NodeKind`/`NodeSpec.kind`/analyzer-seam;
  Spren consumes the canonical `NodeSpec` directly; categories are frontend
  presentation; the verified **P12** specialized-agent-is-a-preset constraint is
  added as a first-class section). `02-data-model.md`'s node_type description →
  `kind`; `07-node-model-core.md` + the prior `08` body get a SUPERSEDED banner;
  `06-ui-systematic-audit.md` re-records RUN-3d's resolution as "consume framework
  canonical post-ADR-008" and marks RUN-3d/3e/PALETTE-Start-End/PRODUCT-BUG-001/
  **RUN-2** superseded/dissolved. `04__remap_node_model.py` is **left in place**
  (committed migration; runner tracks it; deleting it corrupts the applied-ids
  ledger on DBs that ran it — `05` supersedes its effect).

## 4. Out of scope

- RUN-1 (silent run-create swallow), RUN-3b (structured error envelope),
  LINT-REACTIVITY, UI-BUG-007/ARCH-Q-001 auth, SWEEPER-1, palette/node/right-rail
  visual redesign (#21–#24), the nit sweep (#18) — all independent of this reframe;
  full cited register in §10 so they survive compaction.
- **RUN-3e (delivery-contract instruction suffix).** Real, still open, architecture
  decision pending — but orthogonal to the wire-shape reframe. AC-5's
  `Start→Agent→End` reaching `succeeded` is the gate that tells us whether RUN-3e
  still reproduces on the canonical path; do not fold the suffix work in here.
- Making the **User node executable** (human-in-the-loop run path). v0.3 has none
  (the framework binds a `UserNodeHandler` only when a `communication_manager` is
  wired; Spren wires none). The canvas can place/lint User nodes and they
  round-trip as `kind="user"`, but a run that *reaches* one fails at the framework
  before any provider call. Documented as a known v0.3 limitation; a separate
  interactive-runs session owns it.
- v0.4 credential-store seam (keychain / encrypted-SQLite). v0.3 uses the framework
  standard env-var path. Record as a v0.4 follow-up so SP-007 is honored (v0.4 adds
  a *seam*, never patches `runtime_model_config_from_spec`).
- Any `packages/framework/` change. (The `05__*` migration is in scope — it is
  SP-006-mandated forward migration, not a back-compat shim.)

## 5. Files in scope — REPLACE / DELETE / REWRITE / KEEP

**REPLACE body → framework re-export façade:**

| Path | Action |
|---|---|
| `packages/spren/src/spren/models/topology.py` | Delete all local types/validators; `from marsys.coordination.topology.serialize import NodeSpec, EdgeSpec, TopologySpec`; `from marsys.coordination.topology.core import NodeKind, EdgeType, EdgePattern` (the runtime enums — `serialize` has only the string literals; the importer needs real `Enum`s). No `NodeType`/`NodeCategory`/`category_of`. |
| `packages/spren/src/spren/models/workflow.py` | Delete Spren `WorkflowDefinition` + `_validate_cross_references`; `from marsys.coordination.topology.serialize import WorkflowDefinition`; keep envelope types (`Workflow`/`WorkflowCreateRequest`/`WorkflowUpdateRequest`/`WorkflowListResponse`/`WorkflowProvenance`) pointing `.definition` at the framework type. |
| `packages/spren/src/spren/models/agent.py` | `from marsys.agents.serialize import AgentSpec, MemoryRetention` (delete the Spren mirror body). |
| `packages/spren/src/spren/models/model_config.py` | `from marsys.models.serialize import ModelConfigSpec, ModelType, ApiProvider`. |
| `packages/spren/src/spren/models/execution_config.py` | `from marsys.coordination.serialize import ExecutionConfigSpec, ConvergencePolicyConfigSpec, StatusConfigSpec, TracingConfigSpec`. |
| `packages/spren/src/spren/models/__init__.py` | Re-export framework symbols; **drop `NodeCategory` and `NodeType`** from `__all__` (breaking, intended per SP-006 — palette categorization is now a frontend concern). Keep all non-mirror Spren exports (run/trace/tool/file/error/lint/artifact). |
| `packages/spren/src/spren/runs/materialize.py` | Delete every `_materialize_*`/`_build_execution_config`/`_env_secrets_lookup`/`_CANONICAL_USER_NAME`, the `SecretsLookup` alias + `secrets_lookup` kwarg threading, and the `det_nodes`/`core.NodeType` imports. Rewrite `materialize_run`: `topology = pydantic_to_topology(definition, AVAILABLE_TOOLS, handler_registry={})`; `ec = pydantic_to_execution_config(definition.execution_config)`; per-run `ec.tracing`/`ec.aggui` override; return `RuntimeBundle(topology, ec)` (**`agents` field dropped — not derived**, AC-3). NO `api_key` injection anywhere — the framework resolves per-provider (P11). Keep `MaterializationError` for the cases that remain (wrap `UnknownToolError`/`UnknownHandlerError`/`ValidationError`). |
| `packages/spren/src/spren/runs/materialize.py` `RuntimeBundle` | `@dataclass` reduced to `topology: Topology` + `execution_config: ExecutionConfig`; delete the `agents` field and its dead "so the lifecycle coordinator can deregister" docstring (no such path exists — `lifecycle.deregister` only pops `_active_runs`). |
| `packages/spren/src/spren/importers/python_workflow.py` | Emit framework `NodeSpec(kind=…)`; Start/End/User → the matching `kind`; remove any det-node import / `node_type` write. |
| `packages/spren/src/spren/models/lint.py` | Correct the module docstring to the truth (pure Spren linter, zero framework calls). |
| `packages/spren/src/spren/lint/workflow_linter.py` | Confirm it reads framework `NodeSpec.kind` (the entry-node concept = a `kind="start"` node, or no-incoming-edge fallback). Zero framework calls (SP-018). |

**ADD:**
- `packages/spren/src/spren/storage/migrations/05__migrate_node_kind.py` — the
  forward migration (AC-8). Pure-dict, no `spren.models` import, no self-managed
  transaction (runner wraps).

**REWRITE (docs — reconcile, do not delete):**
- `docs/architecture/spren/11-node-model.md` → post-ADR-008 reality (AC-13).
- `docs/architecture/spren/02-data-model.md` node_type→`kind` description.
- SUPERSEDED banners on `07-node-model-core.md` and the prior `08` body; RUN-3d/3e/
  PRODUCT-BUG-001/PALETTE-Start-End re-recorded in `06-ui-systematic-audit.md`.

**DELETE:** nothing (the prior `08` plan said delete `11-node-model.md` and
`04__remap_node_model.py` — both reversed: the doc is rewritten; the committed
migration stays for ledger integrity, `05` supersedes it).

**Web (after TS regen) — migrate `node_type`→`kind`, keep Start/End/User as nodes:**
`apps/web/src/routes/workflows/$workflowId.tsx` (`workflowToReactFlow` :683/:699,
`reactFlowToWorkflow` :744/:745, `addNode` :318/:330/:332, pattern insert :435/:457,
agent-form gate :363), `index.tsx:149`,
`components/WorkflowSnapshotAccordion/WorkflowSnapshotAccordion.tsx:113,124`,
`lib/pattern-presets.ts:98`, `routes/workflows/-canvas/Palette.tsx` (categories;
Start/End/User stay droppable Core items; Tools/Logic/Data inactive),
`-canvas/CanvasNode.tsx:18,62,65,71`, `new.tsx` `EMPTY_DEFINITION` (keep one
default Start — it is now a real persisted `kind="start"` node, not a shim trick),
`lib/api.ts:24` + `lib/api-types.generated.ts` (regenerate). Centralize an
"is Start" guard across every removal path (delete button, xyflow `deleteKeyCode`,
selection/edge-driven removal) so the default Start is non-deletable; allow
multiple User nodes.

**Credential-path cleanup (P11) — full consumer graph to clear:**
- `runs/materialize.py` — the source (covered in the REPLACE table above).
- `scripts/scenarios/marsys_direct_probe.py` (`:18-22` docstring telling operator
  to `export SPREN_ANTHROPIC_API_KEY`, `:147` import of `_env_secrets_lookup`,
  `:150-153` direct call) and `scripts/scenarios/run_failure_probe.py` (`:6-10`,
  `:115-116` `SPREN_ANTHROPIC_API_KEY` setup) → switch to the standard provider
  var; `run_failure_probe.py` also moves to the explicit-`kind` shape + dated model
  id (already in scope).
- Tests: `test_runs_materialize.py` — delete/rewrite `test_default_secrets_lookup_reads_env`
  (`:201-213`), `test_default_secrets_lookup_handles_dashed_provider` (`:216-228`),
  `test_materialize_raises_when_secret_missing` (`:156-168`) — they assert the
  removed convention; carry the **RUN-3a typed-config regression intent** into the
  new path's tests (assert `pydantic_to_execution_config` yields typed
  `status`/`convergence_policy`, and one `AgentRegistry` registration per agent),
  and rewrite the `node_type=`/`bundle.agents` constructions. `test_routes_runs.py`
  (`:58-65` `fake_materialize` stub signature, `:69-72` `RuntimeBundle(...)`,
  `:144` `SPREN_OPENAI_API_KEY` setenv) → align to the reduced `RuntimeBundle` +
  standard env var. (`test_models_workflow.py:40-53` already asserts the correct
  key-free end-state — leave it.)
- Docs: `06-ui-systematic-audit.md` `SPREN_*` mentions re-recorded under AC-13;
  any onboarding/`just dev` banner that implies a Spren-prefixed var → standard.

**KEEP untouched:** `runs/lifecycle.py`, `routes/runs.py` (façade-transparent — P6);
`RuntimeBundle`; `pyproject.toml` `marsys[aggui]` core dep; `scripts/` harness
(only `run_failure_probe.py` updated to the explicit-`kind` shape + dated model id);
the pure-Spren-linter no-framework-call architecture; RUN-1/RUN-3b paths.

## 6. Approach & sequencing (minimal-churn)

1. **Façade first.** Rewrite `spren/models/*` to re-export the framework canonical
   types; drop `NodeType`/`NodeCategory`. Run `uv run --package spren python -c
   "import spren.models"` + `pytest packages/spren/tests` — transient breakage
   expected only in the must-rewrite tests; proves the façade compiles + OpenAPI
   generates.
2. **`materialize_run` swap.** Single `pydantic_to_topology(definition,
   AVAILABLE_TOOLS, handler_registry={})`; derive `agents` from bound nodes;
   `pydantic_to_execution_config` + per-run tracing/AG-UI override. **Verify
   register-once semantics first** (P3 — `pydantic_to_topology` already constructs
   + registers agents; confirm Spren never double-registers; confirm whether
   `bundle.agents` has any consumer — if none, derive-from-topology or drop the
   field, surface either way). Credential path is now the framework standard env
   var (`ANTHROPIC_API_KEY`) — update Spren integration-test env setup
   (`SPREN_*` → standard) to mirror the canonical GAIA test.
3. **`05__migrate_node_kind.py`** + frozen-baseline test (AC-8) — the three seed
   shapes; assert post-migration framework-parseable, no dropped det-nodes.
4. **Importer** → framework `NodeSpec(kind=…)`; **lint** confirm transparent + fix
   docstring.
5. **Web last (after TS regen).** Migrate every `node_type` site to `kind`; keep
   Start/End/User as real Core nodes; default non-deletable Start; multiple User
   allowed; pattern presets seed an explicit Start.
6. **Probe + verify.** `run_failure_probe.py` → explicit-`kind` `Start→Agent→End` +
   dated model id; AC-5 end-to-end; `implementation-reviewer` on the full diff;
   confirm framework baseline 841/764 + zero `packages/framework/` diff.
7. **Docs (AC-13)** reconciled as the final step.

## 7. Risks / known unknowns

| Risk | Mitigation / detection |
|---|---|
| Agent double-construction / double `AgentRegistry` registration (P3, prior P8) | Single `pydantic_to_topology`; derive `agents` from bound nodes; verify register-once against `agents/serialize.py` + `agents/registry.py` BEFORE finalizing; integration test asserts one registration per agent |
| `bundle.agents` has a hidden consumer | grep `lifecycle.py`/`routes`/tests for `.agents`; if unused, drop or derive — surface the finding, don't silently change the dataclass |
| Credential path change breaks Spren integration tests (`SPREN_*` → `ANTHROPIC_API_KEY`) | Mirror the canonical GAIA test env setup; this is a deliberate user-facing v0.3 change (§8 D1) |
| `[E]` premises P3/P4/P5/P10 wrong at code depth | Plan-validator re-derives every `[E]` from source; implementer confirms `pydantic_to_agents`/`pydantic_to_execution_config`/`det_nodes` signatures before relying on them |
| Migration `05` corrupts stored definitions / mis-handles a never-`04` DB | Frozen-baseline test with all three seed shapes; pure-dict, no model import; defensive `system|tool→agent` even though `04` may have run |
| OpenAPI→TS regen drift breaks unrelated web types | Regenerate, diff the generated client, expect only node-shape deltas (`kind`, no `node_type`/`system`/`tool`); web typecheck |
| Web silently loses Start/End on `node_type→kind` rename | AC-11 + a Playwright pass: build `Start→Agent→End`, confirm default Start non-deletable, Run reaches no-`TOPOLOGY_ERROR` |
| Editing `06`/`07`/`11` frozen-ish artifacts | §8 D2: tombstone/banner not delete; `04` migration left in place for ledger integrity |

## 8. Decisions — most of the prior plan's escalations DISSOLVE

The framework change collapsed the prior plan's 5-way escalation. Recorded
resolutions; only **D1** wants explicit user confirmation (and it is forced by
SP-005, so the recommendation is strong).

- **D-DISSOLVED — Entry/exit UX (prior #1).** There is no "how does the canvas
  designate entry/exit without Start/End nodes" question: Start/End/User **are**
  real canvas + wire nodes (`kind`). The palette keeps them. The user's original
  instinct ("why can't the canvas have Start/End?") was correct and is now the
  literal framework design.
- **D-DISSOLVED — Drop-user-nodes / data loss (prior #3a).** USER is a first-class
  `kind`. Migration is a `node_type→kind` field rename; **no node is dropped**.
  The SP-017-spirit data-loss concern evaporates.
- **D-RESOLVED — Frozen-artifact handling (prior #2).** Tombstone, don't delete:
  `07`/prior-`08` get SUPERSEDED banners; `11-node-model.md` is rewritten (its
  product design survives; only its "framework ground truth" section was wrong);
  `04__remap_node_model.py` stays committed (deleting a tracked migration corrupts
  the applied-ids ledger). No frozen `acceptance.md` is mutated.
- **D1 [RESOLVED — user-confirmed 2026-05-17] — v0.3 credential model.** Spren
  imposes **zero** credential assumption: delete the entire `_env_secrets_lookup`/
  `SecretsLookup`/`secrets_lookup`/api_key-inject machinery; defer fully to the
  framework's **per-provider** resolution (P11 — verified table in §9). NOT a
  single `ANTHROPIC_API_KEY` lookup — the user explicitly corrected this: each
  provider reads its own standard var (OpenRouter→`OPENROUTER_API_KEY`, …). Forced
  by SP-005. Side effects: WF-BUG-RUN-2 dissolves (P11); the failure message on a
  truly-absent key becomes the framework's clear per-provider `ValidationError`.
  v0.4 keychain seam = separate follow-up (SP-007 — adds a *seam*, never patches
  `runtime_model_config_from_spec`).
- **D2 [NOTE — consequence, scope deliberately NOT expanded].** The framework does
  not auto-load `.env` (P11 — `os.getenv` only). Post-reframe the sidecar relies on
  the launching process having exported the provider var (today nothing in Spren
  loads `.env`; framework benchmark scripts load it themselves). "Use the same
  logic as the framework" ⇒ Spren does **not** add sidecar `.env`-autoload here
  (that is a separate feature, not this reframe). In scope: update `scripts/` +
  `just dev` banner + onboarding to export/document the requirement (doc/tooling,
  not a code feature). A future "sidecar auto-loads project `.env`" is an
  explicit, separate decision if the convenience is wanted — recorded so it is not
  silently assumed either way.
- **D3 [LOCKED — user-restated 2026-05-17] — palette category model.** See
  **AC-PALETTE**. The 5-category model (Agents w/ standard + specialized catalog +
  future · Core Start/End/User · Logic · Tools · Data) stands; Start default +
  non-deletable in a new canvas; specialized agents are frontend presets (P12, a
  verified framework constraint, not a choice). Session 08 = minimal palette change
  only; full card UX = task #21 with this as its binding spec. Locked here + in the
  `11-node-model.md` rewrite (AC-13) so it stops being lost to compaction.
- **D4 [RESOLVED — user-approved 2026-05-17, Phase-B discovery] — canvas agent
  keying.** The canvas keyed `agents` by a random `agent_<rand>` id (==
  `agent_ref`) while the framework `pydantic_to_topology` binds `agent_ref`
  against `AgentSpec.name` (`serialize.py:503,509`) — so every canvas-built
  workflow would hydrate with unbound agents post-reframe. The plan's implicit
  "definitions bind agents" premise was false for the canvas path (importer +
  canonical example already name-key). User chose: **canvas keys agents by
  name + `agent_ref=name` (drop the id scheme, dedupe name collisions)** AND a
  **forward migration re-keys stored `agents` by name + rewrites `agent_ref`**.
  Scope addition (AC-AGENTKEY-1/2). Shipped as a separate `06__rekey_agents_by_name`
  migration (leaves the verified `05` untouched; one responsibility each).
- **D-NOTE — Model-id constraint.** Standard `anthropic` adapter passes the model
  name verbatim → workflows must use dated ids (`claude-haiku-4-5-20251001`, …),
  not friendly aliases. Unaffected by ADR-008; not a blocker; constrain the model
  picker in the #21 palette/agent-form work.

---

## 9. Framework-side change record (ADR-008 — what changed, cited; user-requested)

Provenance: **[V]** = read in primary source this session; **[E]** = Explore-reported,
re-confirm before relying. Framework is READ-ONLY from this worktree; this section
exists so a future Spren session knows the contract without re-deriving it.

- **`NodeType` → `NodeKind`; members `{AGENT, START, END, USER}` (values
  `agent/start/end/user`); `SYSTEM`/`TOOL` deleted as vestigial.** **[V]**
  `coordination/topology/core.py:20-33`.
- **`RESERVED_NODE_NAMES = frozenset(k.value for k in NodeKind if k is not AGENT)`
  → `{start,end,user}`** — value-set flipped from pre-S08 `{user,system,tool}`,
  same symbol + import path. **[V]** `core.py:39-41`.
- **`Node` dataclass: `name`, `kind: NodeKind = AGENT`, `agent_ref`,
  `is_convergence_point`, `metadata`.** **[E]** `core.py:59-95`. `Topology.__post_init__`
  accepts only homogeneous plain `Node` (rejects `DeterministicNode` instances).
  **[E]** `core.py:168-185` — *the falsifier the implementer must personally
  confirm by reading these lines.*
- **Wire `NodeSpec.kind: NodeKind = AGENT`** (no `node_type`); `extra="forbid"`;
  `@field_validator("kind", mode="before") _reject_removed_node_types` raises a
  migration-message `ValueError` on stored `system`/`tool`. **[V]**
  `coordination/topology/serialize.py:119-166`, removed-set `:86`.
- **`WIRE_SCHEMA_VERSION = 2`; no dual-field shim — the version boundary is the
  contract.** **[V]** `serialize.py:75-81`; `workflow_definition_schema()` embeds
  `x-wire-schema-version` `:560-589`.
- **`WorkflowDefinition._validate_cross_references`** is permissive: a definition
  with nodes but no `kind=START` node deserializes with a `DeprecationWarning`
  (runtime shim synthesizes Start; **shim removed v0.4**); validates `agent_ref ∈
  agents` (AGENT nodes only) and every edge endpoint ∈ node names. **[V]**
  `serialize.py:219-261`.
- **`workflow_to_pydantic` is total** — `Topology.nodes` is homogeneous `Node`;
  deterministic behaviour is materialized from `Node.kind` at the analyzer seam,
  never stored; every `NodeKind` (incl. START/END/USER) serializes 1:1; the prior
  Session-04 det-node serialization rejection is intentionally removed (ADR-008
  Decision 8). **[V]** `serialize.py:336-361`.
- **`pydantic_to_topology(spec, tool_registry, handler_registry=None) -> Topology`**
  materializes every `NodeSpec` as a plain `Node(kind=…)`; internally calls
  `pydantic_to_agents(spec, tool_registry)` once; `USER` node `metadata["handler"]`
  resolved via `handler_registry` (missing → `UnknownHandlerError`). **[V]**
  `serialize.py:471-552`. `exceptions.py` now exports only `UnknownToolError` +
  `UnknownHandlerError`; `NonSerializableTopologyError` deleted. **[V]**
  `exceptions.py:1-26`.
- **`det_nodes.py` retained** as internal analyzer behaviour
  (`StartNode/EndNode/UserNode/DeterministicNode`, `NODE_KIND_BEHAVIOUR`,
  `behaviour_for_kind`), keyed by `NodeKind`, materialized at
  `analyzer.py:172-223` via `graph.register_det_node`. NOT an input to
  `Topology.nodes`. **[E]** `det_nodes.py:36-220`.
- **Consume-path signatures (the API Spren targets):** `pydantic_to_agents(spec:
  WorkflowDefinition, tool_registry) -> List[Agent]` **[E]** `agents/serialize.py:128-176`;
  `runtime_model_config_from_spec(spec, api_key=None, *, runnable=True)` resolves
  the provider env var when runnable **[E]** `models/serialize.py:103-166`;
  `pydantic_to_execution_config(spec: ExecutionConfigSpec) -> ExecutionConfig`
  **[E]** `coordination/serialize.py:280-304`; `ExecutionConfigSpec` carries full
  `status`/`tracing`/`convergence_policy` **[E]** `:93-154`.
- **Credential resolution (verified primary source — P11):** `runtime_model_config_from_spec`
  delegates to `ModelConfig._validate_api_key` (`models/models.py:209-263`).
  Per-provider map `models.py:218-224`:

  | provider | env var (when no `api_key`) | absent → |
  |---|---|---|
  | `openai` | `OPENAI_API_KEY` | `ValueError`→`ValidationError` |
  | `openrouter` | `OPENROUTER_API_KEY` | `ValueError`→`ValidationError` |
  | `google` | `GOOGLE_API_KEY` | `ValueError`→`ValidationError` |
  | `anthropic` | `ANTHROPIC_API_KEY` | `ValueError`→`ValidationError` |
  | `xai` | `XAI_API_KEY` | `ValueError`→`ValidationError` |
  | `openai-oauth` | *(none)* — `OAuthCredentialStore`, `~/.codex/auth.json` | no-op at validate; resolved at adapter init |
  | `anthropic-oauth` | *(none)* — `OAuthCredentialStore`, `~/.claude/.credentials.json` | no-op at validate; resolved at adapter init |

  No `.env` auto-load anywhere in the model/adapter path (`os.getenv` only —
  `models.py:231`).
- **Specialized agents are NOT a wire concept (verified — P12):** the specialized
  classes exist (`marsys/agents/{browser_agent,code_execution_agent,…}.py`) and
  differ only by `__init__` tool-injection/prompt defaults; `AgentSpec`
  (`agents/serialize.py:50-68`, `extra="forbid"`) has no class discriminator;
  `pydantic_to_agents` hard-codes reconstruction to base `Agent`
  (`agents/serialize.py:158`); `BrowserAgent → AgentSpec → Agent` is lossy.
- **Canonical proof references:**
  `packages/framework/examples/session08_workflow_roundtrip.py` (explicit-`kind`
  det-nodes → JSON → disk → reload → `pydantic_to_topology(reloaded,{},{})` →
  `Orchestra.run` → `assert result.success is True`) **[V, read; not executed this
  session]**; `packages/framework/benchmarks/GAIA/test_parallel_tracing_canonical_anthropic.py`
  (the prior canonical agent-only proof). The framework-agent handoff
  `tmp/spren/framework-S08-wire-contract-handoff.md` is an `agent-brief` — NOT a
  citation; every claim above was re-derived from primary source or marked **[E]**.

---

## 10. Complete Spren issue register (cited; user-requested, compaction-survival)

The original audit that spawned all of this is `06-ui-systematic-audit.md`. This is
the verified register (extracted + cross-checked this session). Source of nuance for
each: that file. Status as of 2026-05-17.

### Fixed (shipped)
- **RUN-3a** — run-create 500: Spren passed plain dicts where the framework
  `Orchestra` expects typed `StatusConfig`/`ConvergencePolicyConfig`
  (`AttributeError: 'dict' has no attribute 'enabled'`). Chain
  `routes/runs.py` create_run → `runs/lifecycle.py:186` register_run →
  `Orchestra(...)` → `orchestra.py:304`. Fixed Spren-side
  (`materialize.py` typed reconstruction + 4 regression tests). **Note: this fix
  lives in the `_materialize_*` code this Session 08 deletes — the *reason* it
  existed (typed config) is now satisfied by `pydantic_to_execution_config`; the
  regression intent must carry into the new path's tests.**
- **RUN-3c** — `ModuleNotFoundError: ag_ui` (`orchestra.py:347`); fixed by
  promoting `packages/spren/pyproject.toml` core dep `marsys` → `marsys[aggui]`
  (SP-004). KEEP.
- **ARCH-Q-001 / UI-BUG-007 (decision-locked)** — token lost on reload;
  `capabilities.tsx:38-52,91-108`, `main.tsx:24` StrictMode, `AuthGate.tsx`.
  Decision LOCKED: localStorage, resolve order Tauri-inj → `#token=` fragment
  (always wins) → localStorage; 401 → clear + AuthGate. Subsumes UI-BUG-006.
  Tracked as task #16.

### Superseded / dissolved (by this reframe)
- **RUN-3d** — "visual builder emits no Start node → framework validation fails."
  Root cause re-recorded: Spren maintained a divergent mirror + a det-node
  materializer; the framework now owns the canonical wire shape (ADR-008).
  Resolved by this Session 08. Anchors: `materialize.py` (deleted), framework
  `serialize.py` (consumed).
- **RUN-3e** — "agents never deliver to End / framework hides branch error."
  *Not dissolved* — still a real open architecture decision (delivery-contract
  instruction suffix derived from outgoing edges; `materialize.py` `_materialize_agent`
  was the proposed site, now relocated since that fn is deleted). Out of scope
  here (§4); AC-5 reaching `succeeded` is the diagnostic for whether it still
  reproduces on the canonical path. Secondary half (surface
  `result.branch_results` real error instead of generic "insufficient arrivals",
  `lifecycle.py:312-338` vs framework `orchestra.py:1184`) = task #14 (RUN-3b).
- **PRODUCT-BUG-001** — reserved-name validator over-rejected; auto-dissolves (no
  Spren validator post-façade; framework `NodeSpec` has none) — AC-7.
- **WF-BUG-PALETTE-1** (Start/End not buildable) — resolved *differently* than the
  prior plan intended: Start/End ARE palette/canvas nodes (`kind`), not a metadata
  affordance. PALETTE-2/3/4 (interaction model, categories, card UI) remain open
  under #21.

### Open — independent of the reframe
- **RUN-1 [CRITICAL]** — run-create failure silently swallowed; `RunButton.tsx:122-126`
  only `console.error` (vs terminal-state toast `:91-97`). Task #13.
- **RUN-3b** — run-failure cause hidden behind generic string; `lifecycle.py:312-338`
  reads only `result.success/error/final_response`; framework `OrchestraResult`
  carries `branch_results`; `orchestra.py:1184` overwrites with generic. Surface
  failed-branch reason + structured envelope. Task #14.
- **RUN-2** — *DISSOLVED by this reframe* (moved here from open). Its gate was the
  Spren-side key pre-check `materialize.py:202-206` (`secrets_lookup` returns
  `None` → `MaterializationError` → 400) — deleted by P11. Post-reframe a workflow
  on any provider whose standard env var is exported runs; a truly-absent key
  surfaces the framework's clear per-provider `ValidationError`. The residual
  "no in-product key entry UI" is genuinely Session 10 (Settings) and is unrelated
  to this gate; it is not a v0.3-run blocker once D1 lands.
- **LINT-REACTIVITY [CRITICAL]** — lint findings don't refresh after edits until
  reload; `-canvas/LintChip.tsx` + canvas-edit→lint trigger; agent-form Apply path
  prime suspect. Task #15. (Also fixes UI-BUG-010 stale ✓.)
- **SWEEPER-1** — draft-sweeper FK failure: empty `visual_builder` draft WITH a
  run → `sqlite3.IntegrityError` every sweep; `storage/workflows.py` predicate two
  sites `:108-114` + `:134-141`, `runs.workflow_id` FK no `ON DELETE`
  (`migrations/02__create_runs.py:27`), caught `draft_sweeper.py:60-65`.
  Pre-existing, not a reframe regression. Task #25. (Note: this reframe does NOT
  reshape the empty-draft predicate — the prior Session-07 "Start-seeded draft"
  predicate concern is moot because Spren no longer special-cases node shape for
  drafts; SWEEPER-1's `NOT EXISTS (runs…)` fix stands alone.)
- **PRODUCT-BUG-002** — sidecar silently picks random port if requested taken →
  token/port mismatch; `__main__.py:32-43` `_resolve_port`. Fail-loud or
  parse-ready-line-only. Nit-sweep / #18.
- **TOOLING-BUG-001** — `just dev` Windows recipe leaks sidecar/Vite on exit
  (`Stop-Job` doesn't deliver stdin `shutdown\n`; `__main__.py:62-72`); orphans
  hold 5173/8765. Low priority; #18.
- **Nits (#18):** UI-BUG-001 (invalid `#token=` no feedback), UI-BUG-002 (dev-jargon
  auth error), UI-BUG-003 (wordmark magenta inconsistency — decision-pending),
  UI-BUG-004 (period cue faint), UI-BUG-005 (`--ink-faint` on sub-18px footer,
  violates AC-138), UI-BUG-006 (fragment ignored on soft-nav — subsumed by #16),
  UI-BUG-008 (sidebar stale "coming soon" + leaks "Session NN"), UI-BUG-009
  (palette-added node off-screen, no auto-pan), UI-BUG-010 (lint chip stale ✓ —
  fold into #15), UI-CANDIDATE-001/002 (InputBar chevrons; temporal-anchor time —
  low confidence, verify in a UI pass).

### Open — design (dependent on #21 palette redesign)
- **NODE-1** (#22) canvas node visuals; **RIGHTRAIL-1** (#23) config card collides
  with presence orb / Apply unreachable; **RIGHTRAIL-2** (#24) right-rail visual
  cohesion; **PALETTE-2/3/4** subsumed by the #21 redesign spec in `06`.

**Positives already verified (do not re-audit):** orb state machine, stub send,
sidebar/cmdk, /runs page, workflow list, create flow, agent config form (tools from
real `/v1/tools`), lint detection+panel, Save (PUT+GET+toast).

---

## 11. This session's work + lessons (user-requested record)

**What happened, in order:**
1. Built a checked-in autonomous assessment harness (`scripts/`) and walked the
   `06` audit scenarios; produced the full `06-ui-systematic-audit.md` register
   (§10).
2. Ran the no-YOLO multi-agent fix pipeline on the Run trunk: **RUN-3a** fixed
   (typed-config reconstruction, Spren-side, TRUNK-CRITICAL untouched), **RUN-3c**
   fixed (`marsys[aggui]` core dep). Committed.
3. Hit **RUN-3d**: investigator + independent verifier both concluded "emit
   `DeterministicNode` instances Spren-side." **Both were wrong.** Session 07 was
   written on that premise.
4. User forced a canonical working reference; the prior Session 08 was written
   reframing to "agent-only nodes + `entry_point`/`exit_points` metadata + legacy
   shim" — internally consistent, also wrong.
5. User revealed the framework gap was real and had been **fixed framework-side by
   a separate agent (ADR-008)**; instructed: rebase, verify the handoff against
   primary source (not as truth), rewrite Session 08 ground-up.
6. Rebased `feature/spren-umbrella` onto `origin/main` (safety tag
   `pre-rebase-s08-backup`; clean). Verified the post-ADR-008 contract against
   primary source (§9). Confirmed both prior node-models invalidated; wrote this
   plan.

**Lessons (binding):**
- *A canonical working reference beats any amount of code-tracing or agent
  consensus.* Two independent agents agreeing is not verification — it is
  correlated error when both trace the same ambiguous code. The fix is an external
  re-derivation from a primary source / a run, not more agents.
- *Provenance is load-bearing.* A handoff doc, a prior plan, an upstream agent's
  report = `agent-brief`, never a citation. This plan tags every contract claim
  **[V]/[E]** so the next session does not re-launder them.
- *The framework-vs-Spren-mirror divergence was the real defect class*, not any
  single node bug. SP-005 in its pure form (consume, don't mirror) is the
  structural fix; it only became cleanly possible after ADR-008.
- *Escalating early was right.* The "framework change first" recommendation was
  correct and the user had already executed it; most of the prior plan's 5-way
  escalation then dissolved (§8).

---

## 12. References
- **Canonical proof:** `packages/framework/examples/session08_workflow_roundtrip.py`;
  `packages/framework/benchmarks/GAIA/test_parallel_tracing_canonical_anthropic.py`.
- **Framework canonical (consumed):** `marsys/coordination/topology/serialize.py`,
  `.../core.py`, `.../exceptions.py`, `marsys/agents/serialize.py`,
  `marsys/models/serialize.py`, `marsys/coordination/serialize.py`.
- **ADR-008** (framework worktree — node-kind unification; the change record §9
  summarizes).
- `docs/architecture/spren/08-design-principles.md` — SP-005/006/007/017/018/019.
- `docs/architecture/spren/11-node-model.md` — to be rewritten (AC-13).
- `docs/implementation/spren/v0.3.0/02-run-execution-and-inspection/sessions/06-ui-systematic-audit.md`
  — audit register (§10 is the verified extract).
- `07-node-model-core.md`, prior `08` body — SUPERSEDED (§0).
- Pipeline this session: 3 parallel Explore maps (issue register / framework
  consume-path / web blast-radius) + primary-source verification of §9.
