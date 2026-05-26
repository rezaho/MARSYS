# Acceptance criteria — Framework Session 09: Specialized-Agent Serialization (`kind`/`params` AgentSpec contract)

Frozen at 2026-05-17T00:00:00Z. The test auditor reads ONLY this file plus the test files. Implementation source is intentionally hidden.

Context for the auditor (so criteria are self-contained): the framework owns a canonical cross-process wire format. Workflows are serialized to a `WorkflowDefinition` (JSON). An agent serializes via `agent_to_pydantic` (object → `AgentSpec` pydantic model) and hydrates via `pydantic_to_agents` / `pydantic_to_topology` (pydantic/JSON → live agent objects). Before this session, every specialized `Agent` subclass flattened to a base `Agent` on round-trip (all subclass-specific config silently dropped). This session adds a `kind` discriminator + a typed per-subclass `params` spec, a closed `AGENT_KIND_REGISTRY`, and makes the hydrate entrypoints async. The 5 in-scope specialized subclasses are: `WebSearchAgent`, `BrowserAgent`, `CodeExecutionAgent`, `DataAnalysisAgent`, `FileOperationAgent`. "Round-trip" below means: construct the agent → `agent_to_pydantic` → serialize to JSON → deserialize → hydrate back to a live agent object.

## Functional

### Specialized-agent round-trip (subclass identity preserved)

- AC-1: Round-tripping a `WebSearchAgent` produces a live agent that is an instance of `WebSearchAgent` (not a flattened base `Agent`).
- AC-2: Round-tripping a `BrowserAgent` produces a live agent that is an instance of `BrowserAgent` (not a flattened base `Agent`).
- AC-3: Round-tripping a `CodeExecutionAgent` produces a live agent that is an instance of `CodeExecutionAgent` (not a flattened base `Agent`).
- AC-4: Round-tripping a `DataAnalysisAgent` produces a live agent that is an instance of `DataAnalysisAgent` (not a flattened base `Agent`).
- AC-5: Round-tripping a `FileOperationAgent` produces a live agent that is an instance of `FileOperationAgent` (not a flattened base `Agent`).
- AC-6: For each of the 5 in-scope subclasses, the hydrated instance has its specialized tools/instructions/methods intact (i.e. equivalent to what a freshly constructed instance of that subclass with the same declarative config exposes — the specialized behaviour is rebuilt by the class on hydrate, not carried on the wire).
- AC-7: For each of the 5 in-scope subclasses, the serialized `AgentSpec` carries a `kind` value identifying that subclass, and a `params` object holding that subclass's declarative configuration.

### Additive base surface (load-bearing regression guard)

- AC-8: A round-trip of any of the 5 in-scope subclasses preserves every base `AgentSpec` field that round-trips for a base `Agent` — at minimum: `name`, `goal`, `instruction`, `model`, `tools`, `allowed_peers`, `is_convergence_point`, the memory configuration field(s) (`memory_*`), `plan_config`, and `schemas`. None of these may regress, be dropped, or be reset to defaults when an agent is routed through the specialized (`kind`/`params`) path. (Test must populate these base fields with non-default values on a subclass instance and assert they survive the round-trip.)

### `kind="agent"` default — base `Agent` characterization pin

- AC-9: `AgentSpec.kind` exists with default value `"agent"`.
- AC-10: Serializing a base `Agent` (one not in the 5 specialized subclasses) yields an `AgentSpec` whose `kind == "agent"`.
- AC-11: A base `Agent` round-trips with behaviour unchanged from before this session — every base field equal in/out (characterization pin). No specialized `params`-driven reconstruction is applied to a `kind="agent"` spec.
- AC-12: No migration is performed on existing base-`Agent` specs: a base-`Agent` spec/JSON is behaviourally identical pre- and post-change (the `kind="agent"` default makes existing specs equivalent; tests must not require any spec rewrite/upgrade step for base agents).

### Closed registry — `AGENT_KIND_REGISTRY`

- AC-13: A module-level `AGENT_KIND_REGISTRY` mapping a wire-kind string to an `Agent` subclass exists and is the single source resolving `kind` → class on hydrate.
- AC-14: Each of the 5 in-scope specialized subclasses declares its own stable wire key as a class attribute `WIRE_KIND` (a `ClassVar[str]`).
- AC-15: The reverse mapping (class → kind) used by serialization is derived by inverting `AGENT_KIND_REGISTRY` — it is not a second hand-maintained map. (Observable: the reverse lookup is consistent with the forward registry for every entry; no class→kind pair exists that is not the inverse of a registry entry.)
- AC-16: A CI/unit test exists that fails the build if a class declaring `WIRE_KIND` is absent from `AGENT_KIND_REGISTRY`, or if a registry entry's class lacks/contradicts a matching `WIRE_KIND` (bidirectional `WIRE_KIND` ↔ registry consistency). A forgotten specialized agent fails this test, not a user's load.
- AC-17: `kind` resolution does NOT use reflection / `Agent.__subclasses__()` and does NOT use `marsys.agents.__all__` as the registry source. (Observable: a specialized-like `Agent` subclass that is importable/in `__all__` but NOT in `AGENT_KIND_REGISTRY` is not resolvable by hydrate — resolution is registry-membership-gated, and AC-16's test flags it.)

### `params` — typed, declarative, no secrets, no runtime objects

- AC-18: `params` is a typed per-subclass spec (a distinct `*ParamsSpec` type per subclass), not an open/untyped `dict`.
- AC-19: For every one of the 5 subclasses, the serialized JSON of a round-tripped spec contains NO credential/secret value (e.g. `WebSearchAgent`'s Google / NCBI / search API keys). Constructing a subclass instance with secret kwargs set must not cause those secret values to appear anywhere in the serialized output.
- AC-20: For every one of the 5 subclasses, the serialized JSON contains NO non-JSON/runtime object and NO field carrying one (specifically: no `RunFileSystem`/filesystem object, no `CodeExecutionConfig`/`code_config`, no `ModelConfig` object as `params` content). These are reconstructed by the subclass `__init__` on hydrate, not transported.
- AC-21: For every one of the 5 subclasses, the serialized JSON contains NO machine-specific absolute filesystem path (e.g. a resolved `base_directory`/`root_dir`/cwd-derived path). A subclass that defaults a directory to the process cwd when none is given must not embed that resolved path on the wire.
- AC-22: The serialized `params` content for every subclass is JSON-safe (serializes and deserializes through plain JSON without custom object encoders).

### Async hydrate

- AC-23: `pydantic_to_agents` is an async function (`await`-able coroutine), and there is NO separate sync variant / sync wrapper of it shipped as a public entrypoint — async is the single hydrate entrypoint.
- AC-24: `pydantic_to_topology` is an async function (`await`-able coroutine), and there is NO separate sync variant / sync wrapper of it shipped as a public entrypoint.
- AC-25: Round-tripping a `BrowserAgent` through the async hydrate entrypoint yields a browser-READY `BrowserAgent` instance (equivalent to one built via the subclass's async `create_safe` construction path), NOT a sync-constructed browser-unready instance. (Test must assert the browser-readiness state, not merely the type — AC-2 covers type.)
- AC-26: The async hydrate dispatch builds each specialized subclass using the framework's async-construction idiom: a class exposing an async `create_safe` is constructed via `await create_safe(...)`; a class without it is constructed via the plain constructor. (Observable via AC-25 for the `create_safe` branch and AC-1/AC-3/AC-4/AC-5 for the plain-constructor branch.)
- AC-27: The in-repo sync hydrate callers are migrated to the async entrypoint and pass: the Session-08 round-trip example (`examples/session08_workflow_roundtrip.py`) runs successfully end-to-end, and the previously-sync hydrate test callsites execute against the async entrypoint and pass. (No remaining in-repo caller invokes a removed sync hydrate API.)

### Field rename & schema version

- AC-28: `AgentSpec` exposes the agent's model configuration under the field name `model` (the field formerly named `agent_model` is renamed to `model`). A round-trip preserves the model configuration via `model`.
- AC-29: There is no `agent_model` field remaining on `AgentSpec` and no dual-field/alias shim accepting both `agent_model` and `model` (the version boundary is the contract; no backward-compat alias).
- AC-30: `WIRE_SCHEMA_VERSION` equals `3`.
- AC-31: A serialized `WorkflowDefinition` embeds the schema version as `x-wire-schema-version` with value `3`.

### `LearnableAgent` / `BaseLearnableAgent` — explicitly unchanged

- AC-32: `agent_to_pydantic` raises `TypeError` when given a `LearnableAgent` (behaviour unchanged from before this session — it does NOT silently flatten and does NOT gain a serialization path).
- AC-33: `agent_to_pydantic` raises `TypeError` when given a `BaseLearnableAgent` (behaviour unchanged — no new `BaseAgent`-level serialization path is introduced for it).

## Non-functional

- AC-34: Full framework test suite is green relative to the Session-08-merged baseline: zero NEW test failures introduced by this session's changes (regression guard). (A test asserting this is not expected; this is the suite-level pass bar the auditor should confirm is represented as the regression gate.)

## Out of scope

(Tests asserting these in THIS session would be wrong.)

- `LearnableAgent` / `BaseLearnableAgent` becoming serializable — they must remain unserializable exactly as today (see AC-32/AC-33; do NOT write a test expecting a successful `LearnableAgent` round-trip).
- Spren mirror update and Spren stored-data migration (contracted Spren-side follow-up; no Spren files are touched or tested here).
- Session-10 credential-resolution / `runnable` / dual-`ModelConfig` collapse (ADR-010).
- New specialized agent classes beyond the 5 in scope; edge/pattern semantics; v0.4 shim/heuristic removal.
- Backward-compat for the old `agent_model` field name or for `WIRE_SCHEMA_VERSION` 2 (the schema bump is an intentional break; no migration shim — see AC-29).

## Open / needs clarification

- (Implied — confirm with user) AC-15/AC-16: the plan/ADR specify the reverse map is "derived by inversion" and a CI test asserts bidirectional consistency, but do not name the exact public symbol/function for the reverse map or the CI test. Criteria are phrased behaviourally (consistency observable, build-failing test exists) so they survive the implementation's naming choice; flag only if the auditor needs an exact symbol name.
- (Implied — confirm with user) AC-34: the "zero new failures vs S08-merged baseline" is a suite-level gate stated in plan §7/§8 and ADR Risks; it is not expressible as a single unit test. Auditor should treat it as the regression bar (full suite must pass), not as a discrete testable assertion file.
