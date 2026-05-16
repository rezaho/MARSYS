# Session 07 — Canonical WorkflowDefinition run-path: reproduce & fix

**Status:** DONE — reproduced, root-caused, contract redesigned, verified
**Role:** implementer (investigative reproduce-then-fix; no pre-written architect plan — this doc IS the tracking contract per user request)
**Branch:** feature/tracing-streaming
**Created:** 2026-05-16

---

## 1. Goal

A real user reports problems when defining a workflow with the **canonical**
representation (the Session-04 `WorkflowDefinition` Pydantic wire shape:
`TopologySpec` + `AgentSpec` + `ExecutionConfigSpec`, hydrated via
`pydantic_to_topology`) instead of the string-notation topology dict.

Concretely:

1. Build a canonical-version twin of
   `packages/framework/benchmarks/GAIA/test_parallel_tracing.py` — same
   agents, same topology, same tracing — but expressed through
   `WorkflowDefinition` / `AgentSpec` / `TopologySpec` and hydrated with
   `pydantic_to_topology`.
2. Reproduce the reported failure by running it end-to-end through
   `Orchestra.run()` exactly like the string-notation test runs.
3. Root-cause it (bug-fix protocol).
4. Fix it so the canonical-version test runs to the same successful outcome
   as the string-notation baseline.

This is reproduce-first. We do not assume the bug's mechanism; we observe it.

---

## 2. Premise Ledger

Load-bearing premises this task rests on. `Source` token per CLAUDE.md
grounding discipline. A premise may not enter a conclusion until
`CONFIRMED@<cite>` or `RISK-LOGGED`.

| # | Premise | Source | Status |
|---|---------|--------|--------|
| P1 | Another user hits issues using the canonical definition for topology + agents | user-stated (relayed third-hand; symptom unspecified) | UNVERIFIED — must reproduce empirically; not assumed true |
| P2 | `test_parallel_tracing.py` (string-notation) runs to success — the baseline | user-stated | UNVERIFIED — confirm baseline before claiming canonical is broken |
| P3 | "canonical version" = the Session-04 `WorkflowDefinition` wire shape (`TopologySpec`/`AgentSpec`) hydrated via `pydantic_to_topology` | my-recall (inferred) — strongly evidenced @ `coordination/topology/serialize.py:1-35` ("Canonical Pydantic wire shape for a runnable marsys workflow"); `WorkflowDefinition` carries both `topology` and `agents`, matching the user's "topology and agents" phrasing | RISK-LOGGED — proceeding on this interpretation; surfaced to user. If wrong (e.g. they meant object-notation dict or raw `PatternConfig`), test design must change |
| P4 | The canonical run path is: build `WorkflowDefinition` → `pydantic_to_topology(spec, tool_registry)` → `Orchestra.run(topology=<Topology>, …)`. A `WorkflowDefinition` cannot be passed to `Orchestra.run` directly | primary-source | CONFIRMED@`coordination/orchestra.py:426-441` (`_ensure_topology` returns a `Topology` as-is; raises `TypeError` for anything not `Topology`/`PatternConfig`/`dict` — `WorkflowDefinition` is none of these) |
| P5 | `Agent.__init__` auto-registers into the global `AgentRegistry`, so agents materialized by `pydantic_to_agents` are registry-resolvable by name | primary-source | CONFIRMED@`agents/agents.py:253` (`self.name = AgentRegistry.register(...)` in `BaseAgent.__init__`) |

## 3. Frame check

Most load-bearing premise: **P1 — the bug exists in the canonical run
path.** Concrete observation that would *falsify* the frame: the
canonical-version test runs end-to-end with the same success outcome as the
string-notation baseline (same `result.success`, comparable trace tree, no
exception). The reproduction step exists precisely to look for that
falsifier *before* any RCA. If the canonical test passes cleanly, P1 is
false and the task pivots to "could not reproduce — get the reporter's exact
recipe."

We do **not** pre-commit to a root cause. (Candidate mechanisms exist —
e.g. `runtime_model_config_from_spec` builds the runtime `ModelConfig` via
`ModelConfig.model_construct(api_key=None, …)` at `models/serialize.py:140`,
which bypasses every `model_validator` including `_validate_api_key` and
provider/base-url derivation; whether that is *the* fault is for the
observed failure + investigator to establish, not this section to assert.)

---

## 4. Scope

**In scope**
- New script `packages/framework/benchmarks/GAIA/test_parallel_tracing_canonical.py`.
- Whatever framework fix the reproduced + root-caused failure requires
  (likely in the Session-04 serialize layer:
  `models/serialize.py` / `agents/serialize.py` /
  `coordination/topology/serialize.py`, or the Orchestra hydration path).

**Out of scope (unless RCA proves otherwise)**
- Redesigning the canonical wire shape.
- The string-notation path itself.
- Spren-side mirror cleanup (tracked separately in Session 04).

---

## 5. Approach

1. Write the canonical twin test. Mirror exactly: same three agents
   (Coordinator/Researcher/FactChecker), same goals/instructions/
   memory-retention, same `claude-haiku-4.5` `anthropic-oauth` model,
   same flows, same `entry_point`/`exit_points` (into `TopologySpec.metadata`,
   matching what `StringNotationConverter` does @ `string_converter.py:54-65`),
   same `TracingConfig`. Hydrate via `pydantic_to_topology(spec, {})`.
2. Confirm baseline (P2): run the string-notation test as a script.
3. Reproduce (P1): run the canonical test as a script. Capture the exact
   failure verbatim.
4. RCA via bug-fix protocol: ground → `failure-root-cause-investigator`
   (provenance-separated input) → self-review → independent review if
   warranted.
5. Apply the architecture-fitting fix. Re-run canonical + string +
   touched-module regression suite.
6. Fill §6 and §7 below; flip status to DONE.

---

## 6. Root-cause analysis

### 6.1 Reproduction results (empirical)

| Run | Config | Outcome |
|-----|--------|---------|
| Baseline string-notation (`test_parallel_tracing.py`) | `anthropic-oauth`, `oauth_profile=marsys-2` | **Success** — 15 steps, 24.0s, coherent answer. P2 CONFIRMED |
| Canonical twin (`test_parallel_tracing_canonical.py`) | same `anthropic-oauth` | **Success** — 14 steps, 21.0s, coherent answer. Frame falsifier observed: simple oauth case is NOT broken |
| Deterministic canonical defect probe | `provider="anthropic"` (standard API key), `ANTHROPIC_API_KEY` set in env | **Defect reproduced** — see below |

The broad premise P1 ("canonical is broken") is **false for oauth, true
for standard-API-key providers**. The oauth canonical run passed because
oauth providers structurally dodge the two bypassed validators (see 6.2).
Narrowing the reproduction to the realistic non-oauth case confirms the
defect.

Deterministic probe (no LLM cost), `ANTHROPIC_API_KEY` present in env:

```
Normal/string path   ModelConfig(provider='anthropic', ...)
  base_url = 'https://api.anthropic.com/v1'
  api_key  = '<read from ANTHROPIC_API_KEY env>'

Canonical path       ModelConfigSpec(provider='anthropic') -> runtime_model_config_from_spec()
  base_url = None
  api_key  = None
```

### 6.2 Root cause

`pydantic_to_agents` (`agents/serialize.py:156`) materializes each agent's
runtime model config via `runtime_model_config_from_spec(agent_spec.agent_model)`
with **no `api_key` argument**. That function's no-key branch
(`models/serialize.py:140`) builds the config with
`ModelConfig.model_construct(api_key=None, **fields)`.

`model_construct` is Pydantic's validator-bypassing constructor. It skips
**every** `ModelConfig` validator, including the two that make a config
runnable:

- `_set_base_url_from_provider` — `@model_validator(mode="before")`
  (`models.py:183-207`). Maps `provider` → `base_url` via
  `PROVIDER_BASE_URLS`. Bypassed ⇒ `base_url` stays `None` when the spec
  author set only `provider` (the normal canonical authoring pattern).
- `_validate_api_key` — `@model_validator(mode="after")`
  (`models.py:209-263`). For standard providers reads
  `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY` / `XAI_API_KEY`
  / `OPENROUTER_API_KEY` from the environment. Bypassed ⇒ `api_key` stays
  `None` even though the env var is set.

Net: agents built from a canonical `WorkflowDefinition` with any
standard-API-key provider get a `ModelConfig` with no endpoint and no
credential, and fail at first LLM call. **Why oauth dodged it:** for
`*-oauth` providers `_validate_api_key`'s relevant branch is a no-op
(`models.py:245-248`) and the oauth adapter resolves both endpoint and
credential from `oauth_profile` at client-init (`models.py:524-536`),
independent of these validators — so bypassing them costs nothing on the
oauth path. Standard providers have no such independent recovery.

This is not a flaw in `model_construct` itself. The no-key path was a
deliberate Session-04 choice so storage-time materialization (community
templates, unconfigured machines) does not raise from `_validate_api_key`
(documented `models/serialize.py:1-17`). The defect is that
`pydantic_to_agents` is a **runtime materialization** path (it builds live
`Agent`s that must execute), yet it routes through the inspection-grade
no-key branch and never attempts env-var credential resolution or
provider→base_url derivation.

### 6.3 Bug-fix protocol

Public-contract serialize layer with cross-package consumers (Spren,
MARSYS Cloud, community templates per `models/serialize.py:1-17`) ⇒
high-stakes ⇒ full protocol: ground → `failure-root-cause-investigator`
→ self-review → independent `root-cause-solution-reviewer` → validate
gate (surface to user before applying; must NOT reintroduce the
storage-time raise). Findings recorded in §7.

---

## 7. Proposed solution

**User decision (2026-05-16): redesign the Session-04 contract, not just
patch the call site.** The root cause is not merely the call site — it is
that `runtime_model_config_from_spec`'s **silent default is the
non-runnable inspection branch**. Any runtime caller that takes the path of
least resistance gets a broken config. Flipping the default removes the
footgun at the source and makes the bug structurally impossible.

### 7.1 New contract

`runtime_model_config_from_spec` gains a keyword-only `runnable: bool = True`
axis, orthogonal to `api_key`:

```python
def runtime_model_config_from_spec(
    spec: ModelConfigSpec,
    api_key: Optional[str] = None,
    *,
    runnable: bool = True,
) -> ModelConfig:
```

- `runnable=True` (**new default**): full validation via the normal
  `ModelConfig(...)` constructor — `_set_base_url_from_provider` derives
  `base_url`; `_validate_api_key` resolves the env-var key for standard
  providers and **raises `ValueError` if missing** (exact parity with a
  directly-constructed `ModelConfig` / string-notation); oauth branch is a
  no-op (works as the proven string-notation baseline does). This is the
  execution contract.
- `runnable=False`: inspection/storage — `ModelConfig.model_construct(...)`,
  validators bypassed, never raises, every non-secret field preserved, NOT
  runnable. For loading / validating / displaying stored specs on machines
  without credentials (community templates, MARSYS Cloud pre-deploy
  validation, Spren persistence). This is the *former* default, now an
  explicit, honest opt-in.

`api_key` is unchanged and orthogonal: explicitly supplied → used (with
`runnable=True`, validated — AC-34).

### 7.2 Why this fixes the bug with no call-site change

`pydantic_to_agents` (`agents/serialize.py:156`) calls
`runtime_model_config_from_spec(agent_spec.agent_model)` — with the new
default it now gets a runnable config. **The default flip *is* the fix.**
Only `models/serialize.py` (branching + module/function docstrings) changes
in production code; `models/models.py` is untouched (Session-04 out-of-scope
constraint at `04-...acceptance.md:138` honored), `agents/agents.py`
untouched, no public signature break (purely additive keyword-only param).

### 7.3 Frozen-acceptance amendment (Session 04)

Surfaced and user-approved. Amended in
`docs/implementation/framework/sessions/v0.3.0/04-workflow-serializer/acceptance.md`
with dated `[amended 2026-05-16 — Session 07]` notes:

- **AC-9** — signature becomes
  `runtime_model_config_from_spec(spec, api_key=None, *, runnable=True)`.
- **AC-33** — the round-trip/field-preservation guarantee now holds for
  `runtime_model_config_from_spec(spec, runnable=False)` (was the bare
  default). Storage/inspection intent preserved, now explicit.
- **AC-34** — unaffected (explicit key still validated); clarifying note
  added.

### 7.4 Test changes (contract change, NOT test silencing)

`tests/agents/test_serialize.py::test_runtime_model_config_from_spec_with_no_api_key_succeeds`
encoded the *old* default. Rewritten to assert the inspection contract via
the explicit `runnable=False`. New tests added for the new default:
env-var key resolution succeeds (runnable), and missing env var raises
`ValueError` (parity with string-notation). This is a user-approved
contract change re-pinned by tests — not anti-pattern #9 (skip/silence).

### 7.5 Independent-review concerns folded in

- Spren mis-citation corrected: the reason not to mutate the inspection
  branch is the **framework-internal** storage contract (AC-33 + the pinned
  test), not Spren (Spren never calls the runtime fn — only imports the
  `ModelConfigSpec` type). The redesign keeps the inspection branch intact;
  it only stops being the silent default.
- Env-var-missing path: must **raise clearly** (let the validated
  `ModelConfig` constructor raise — same `ValueError` string-notation
  emits), never swallow into a silent `api_key=None` (would reintroduce
  the bug one layer deeper) and never a bare `except` (anti-pattern #13).
- One-line WHY comment at the oauth no-op rationale in the function
  docstring (non-obvious hidden coupling: oauth adapter resolves
  endpoint+credential from `oauth_profile` independently).

---

## 8. Acceptance criteria

Frozen separately at `07-canonical-workflow-repro/acceptance.md` before the
fix is applied (the in-flight reproduction is investigative; acceptance
covers the deliverables — the test file existing & passing, behavioral
parity with the baseline, root-cause documented, regression suite green).

---

## 9. Outcome & verification

**Production change (one file):**
`packages/framework/src/marsys/models/serialize.py` — module-docstring last
paragraph, `runtime_model_config_from_spec` signature + docstring, and the
final branching. `models/models.py` and `agents/agents.py` untouched
(Session-04 out-of-scope honored); `pydantic_to_agents` /
`pydantic_to_topology` signatures + call sites unchanged (the default flip
alone fixes it); no speculative `model_credentials` seam.

**Tests:**
- `benchmarks/GAIA/test_parallel_tracing_canonical.py` — new canonical
  twin; now asserts `result.success` + non-empty response (not an
  assertion-free script).
- `tests/agents/test_serialize.py` — old default-pinning test rewritten →
  `..._inspection_mode_preserves_fields_no_raise` (`runnable=False`); new:
  `..._default_runnable_resolves_env`,
  `..._default_runnable_raises_on_missing_env` (+ direct-`ModelConfig`
  parity), `..._default_matches_direct_modelconfig`,
  `..._oauth_default_no_env_var_needed`. Load-bearing autouse fixture
  commented.

**Verification:**
- Deterministic (no LLM): canonical default == string-notation path for
  `provider="anthropic"` (`base_url` + `api_key` parity); `runnable=False`
  preserves fields, `api_key=None`, never raises; missing env var raises
  `ValueError` (parity with string-notation).
- E2E: canonical oauth twin → `result.success=True`, coherent answer,
  exit 0 (with assertions). String baseline P2 confirmed earlier; its code
  path is untouched by this change (constructs `ModelConfig` directly,
  never calls `runtime_model_config_from_spec`) so not re-run.
- Regression: `tests/agents/test_serialize.py`,
  `tests/coordination/topology/`,
  `tests/integration/test_workflow_definition_round_trip.py` — 92 passed.
- Independent commit-before-exposure review of the root cause + fix:
  converged; verdict SOUND-WITH-CONCERNS, all concerns folded in.
- Implementation review of the delta: Critical 0, Important 0.

**Design-smell resolution:** the "unsafe silent default" the user flagged
is fixed at the source (default is now `runnable=True`; the non-runnable
inspection contract is an explicit opt-in). No follow-up debt — the
footgun is gone, not deferred.
