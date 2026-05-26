# Framework Session 10: Runtime credential resolution — collapse the storage/runtime ModelConfig split

Status: PLAN (Phase A). Created 2026-05-17. Owner: implementer session.

> **Supersedes a deferred Session-04 decision.** Session 04 explicitly put "Editing `ModelConfig` to remove or alter the `_validate_api_key` validator" *out of scope* and introduced `ModelConfigSpec` as a tactical dodge (`04-workflow-serializer/acceptance.md:142`, AC-67; the Session-04 plan's own "Lessons" calls `ModelConfig` *"storage-hostile"*, `04-workflow-serializer.md:569`). Session 10 makes the strategic fix Session 04 deferred. The supersession is recorded in **ADR-010** and via dated amendments to the Session-04 `acceptance.md`.

---

## The big picture (background — read first)

A workflow definition must be serialized, saved, validated, and shared **without** API credentials (Spren persistence, MARSYS Cloud pre-deploy validation, community templates, machines configured with only a subset of provider keys). Today the framework cannot construct a runtime `ModelConfig` in those contexts: `ModelConfig._validate_api_key` (`models/models.py:209-263`) is a `@model_validator(mode="after")` that, for `type="api"` with no key passed and the provider env var absent, **raises `ValueError` at Pydantic-construction time** (`models/models.py:241-244`). Worse, on a machine that *does* have the env var set, construction silently pulls the live secret into the object (`models/models.py:235`) — a credential-leak vector if such an object is ever persisted.

Session 04 worked around this by introducing `ModelConfigSpec` (`models/serialize.py:49-75`): a field-by-field mirror with **no `api_key` field and no validator**. Session 07 then added a `runnable: bool` flag to `runtime_model_config_from_spec` so the same function could either build a real (raising) `ModelConfig` or a `model_construct`'d non-validated one (`models/serialize.py:103-166`). The result is a dual-path serializer plus a duplicated type (Spren mirrors it again in `packages/spren/src/spren/models/model_config.py`).

The original API contract was always: *no `api_key` → resolve the provider env var at runtime → error only if neither is reachable*. That env-var fallback is **already fully implemented** (`models/models.py:218-244`) and `api_key` is **already `Optional[str] = None`** (`models/models.py:108`). The only defect is **timing**: resolution + the raise happen at *Pydantic construction* instead of at *model materialization* ("at runtime"). Moving that one step downstream removes the entire reason the dual-path serializer and the storage/runtime split exist.

## Goal

Move API-key resolution (env-var lookup + the "credential unreachable" error) from `ModelConfig` Pydantic-construction to model materialization (the runnable-model factory), so that:

1. Constructing a `ModelConfig` for an `api` model with no key and no env var **never raises** and **never reads/stores a secret**.
2. The credential is resolved — and the error raised — only when a runnable model is actually built from that config.
3. `runtime_model_config_from_spec` collapses to a **single path** (no `runnable` flag, no `model_construct` bypass).
4. `ModelConfigSpec` is **kept** as the storage wire shape — its no-`api_key`-field property is a *structural* guarantee that a credential cannot be written to disk (the user's explicit choice; behavioral excludes are one missed `.model_dump()` from a leak).

Outcome: storage/validation paths construct real validated `ModelConfig`/specs on credential-less machines with no special path; the serializer is one function; the Spren-side mirror deletion (separate follow-up PR) becomes trivial.

## Scope

### In scope (framework only)

- `packages/framework/src/marsys/models/models.py` — remove credential resolution + raise from `_validate_api_key`; add a single pure resolver helper.
- `packages/framework/src/marsys/agents/agents.py` — call the resolver at the model-materialization seam (`_create_model_from_config`, the `api` branch).
- `packages/framework/src/marsys/models/serialize.py` — collapse `runtime_model_config_from_spec` to one path; delete the `runnable` param and the `model_construct` branch; update module/function docstrings.
- Tests under `packages/framework/tests/` covering the new behavior (see Testing strategy).
- `docs/architecture/framework/decisions/ADR-010-runtime-credential-resolution.md` — new ADR.
- Dated amendments to `docs/implementation/framework/sessions/v0.3.0/04-workflow-serializer/acceptance.md` (AC-9, AC-33, AC-67, Out-of-scope line) — same dated-reversal convention Session 08 used for AC-59 (`...:117`).
- `packages/framework/CHANGELOG.md` — `## [Unreleased]` entry.

### Explicitly out of scope (surface, do not bundle)

- **Spren-side mirror deletion** (`packages/spren/src/spren/models/model_config.py` and re-import). This is the *already-committed coordinated follow-up PR* (`04-workflow-serializer/acceptance.md:144`; project memory `project_spren_modelconfig_mirror.md`). Session 10 *enables* it; it does not perform it. Anti-pattern #8 (no smuggled cross-package changes).
- `_set_base_url_from_provider` (`models/models.py:183-207`) — a separate, non-credential validator. Untouched. (It derives `base_url` from `provider`; needs only stored fields; harmless for storage. Removing the `model_construct` branch means it now also runs on the previously-bypassed path — intended and correct.)
- TRUNK-CRITICAL files (`coordination/orchestra.py`, `coordination/execution/orchestrator.py`, `coordination/execution/real_runtime.py`, `coordination/topology/graph.py`, `coordination/validation/response_validator.py`) — untouched.
- Migrating the OAuth credential mechanism (`models.py:524-538`) — unchanged; the resolver no-ops for `*-oauth` providers exactly as `_validate_api_key` does today.

## File targets & approach

### 1. `models/models.py` — single resolver, no construction-time raise

- **Delete** the credential logic from `_validate_api_key` (`209-263`). Net effect: constructing `ModelConfig(type="api", ...)` with no key never reads env, never mutates `api_key`, never raises. Remove the now-dead `@model_validator(mode="after") _validate_api_key` entirely (the OAuth/no-provider warnings it emitted move to the resolver, or are dropped if redundant — decide during implementation; warnings are not contract).
- **Add** a module-level pure function, single source of truth for the provider→env-var map and the OAuth set:

  ```python
  def resolve_api_key(model_type: str, provider: str | None, api_key: str | None) -> str | None:
      """Resolve the credential at model-materialization time.
      - non-api type, or api_key already supplied, or OAuth provider → return api_key unchanged.
      - api type, no key: read the provider env var; return it if present.
      - api type, no key, no env var, known provider → raise ValueError (same message as today).
      """
  ```
  Message parity with the current `models.py:241-244` text ("Set the `<ENV>` environment variable or provide `api_key` directly.").
- The `PROVIDER_BASE_URLS` / env-var map / oauth set must not be duplicated — one definition, referenced by both `_set_base_url_from_provider` and `resolve_api_key`.

### 2. `agents/agents.py` — call resolver at the materialization seam

- In `_create_model_from_config` (`agents.py:2861`), the `api` branch (`2920-2941`), replace `api_key = config.api_key` (`2921`) with `api_key = resolve_api_key("api", config.provider, config.api_key)`. This is the precise chokepoint where a runnable `BaseAPIModel` is built from a `ModelConfig`; `BaseAPIModel.__init__` (`models.py:488`) already handles the OAuth branch independently (`models.py:524-538`), so OAuth (`resolve_api_key` → `None`) is unaffected.
- Confirm during A2/validation that `_create_model_from_config` is the **only** path that materializes a runnable API model from a `ModelConfig` (no bypass). If another path exists, the resolver must cover it too — surface before implementing.

### 3. `models/serialize.py` — one path

- `runtime_model_config_from_spec(spec, api_key=None)` → always `return ModelConfig(api_key=api_key, **common_fields)`. Delete the `*, runnable: bool = True` param and the `if not runnable: ModelConfig.model_construct(...)` branch (`159-162`). Construction no longer raises, so the bypass has no reason to exist.
- Rewrite the module docstring (`serialize.py:1-23`) and the function docstring (`109-139`): the spec is storage-safe by *structure* (no `api_key` field); materialization happens at model-build time, not here; constructing a `ModelConfig` from a spec is always safe and non-raising; the config becomes runnable when a model is materialized from it (and *that* raises if the credential is unreachable).

### 4. Session-04 acceptance amendments (surface explicitly)

Append dated notes (do **not** rewrite existing lines) to `04-workflow-serializer/acceptance.md`:
- AC-9 / AC-33: signature loses `runnable`; the no-raise property no longer needs an opt-in because construction never raises.
- AC-67 ("`models/models.py` is not edited"): reversed — Session 10 deliberately edits it. Cite ADR-010.
- Out-of-scope line 142 ("Editing `ModelConfig` … `_validate_api_key`"): superseded by Session 10.
Pattern: identical to the `[REVERSED 2026-05-17 — Session 08; ADR-008 ...]` note at `acceptance.md:117`.

### 5. ADR-010 + CHANGELOG

- `ADR-010-runtime-credential-resolution.md` — Y-statement; supersedes the Session-04 line-142 deferral and AC-67; states the structural-vs-behavioral credential-safety rationale for keeping `ModelConfigSpec`.
- CHANGELOG `## [Unreleased]`: ModelConfig no longer resolves/raises for credentials at construction; resolution moved to model materialization; `runtime_model_config_from_spec` signature simplified (`runnable` removed).

## Testing strategy

Levels: unit (fast, no network), integration (serializer round-trip, agent materialization with fake env key), live (`tests/models/test_provider_integration.py`, env-gated — **not** in blast radius; those build `BaseAPIModel` directly with explicit keys, not via `ModelConfig._validate_api_key`).

New / changed tests:
- **Construction no longer raises**: `ModelConfig(type="api", name="x", provider="anthropic")` with no `api_key` and no `ANTHROPIC_API_KEY` constructs successfully; `.api_key is None`; no secret read.
- **Error moved to materialization**: building a runnable model from that config (via `_create_model_from_config` / `Agent(...)`) with no key/env/oauth raises `ValueError` with the parity message.
- **Env fallback still works** — at materialization, with the provider env var set, the resolved key reaches the adapter.
- **OAuth providers** (`anthropic-oauth`, `openai-oauth`): construct and materialize with no key and no env var, no raise (resolver no-ops; OAuth path owns credentials).
- **Single-path serializer**: `runtime_model_config_from_spec(spec)` has no `runnable` kwarg (call with it → `TypeError`); preserves every non-secret field incl. `oauth_profile`; does not raise even with no credentials reachable.
- **Structural safety preserved (regression)**: existing `test_model_config_spec_has_no_api_key_field` / `..._json_round_trip_omits_api_key` stay green; spec still has no `api_key` field; AC-30/AC-35/AC-31/AC-32 unaffected.
- **Reworked**: the explicit `runnable=False` test in `packages/framework/tests/agents/test_serialize.py` (~L203) — rewrite to assert the single-path contract. This removes the `runnable=False` inspection API: a deliberate, user-approved API simplification, surfaced here per anti-pattern #1 (not a silent test deletion).
- **Stale-comment fix**: `_api_key_env` fixture docstring in `tests/coordination/topology/test_serialize.py:61-67` (references `ModelConfig._validate_api_key`) updated to name the materialization-time resolver. Behavior holds (fixture still needed: agent materialization still resolves env at build time).
- **Regression gate**: full framework suite, zero new failures vs. baseline.

## Premise Ledger

| # | Premise | Source | Status |
|---|---------|--------|--------|
| P1 | A no-`api_key` counterpart to `ModelConfig` exists (`ModelConfigSpec`), duplicated in Spren | user-stated | CONFIRMED@`models/serialize.py:49-75`, `packages/spren/src/spren/models/model_config.py:34` |
| P2 | Original vision = api_key optional, else provider env var, else error | user-stated | CONFIRMED@`models/models.py:108-110` (optional) + `209-263` (env fallback + error) |
| P3 | Today `api_key` in `ModelConfig` is *required* | user-stated | **REFUTED**@`models/models.py:108` — `Optional[str] = Field(None, ...)`. Already optional; vision already shipped. No work needed on optionality. |
| P4 | The counterpart exists *because* api_key is required | user-stated | **REFUTED** — it exists because `_validate_api_key` *raises at construction* (`models/models.py:241-244`) and Session 04 deferred fixing that (`04-workflow-serializer/acceptance.md:142`, AC-67; `04-workflow-serializer.md:569` "storage-hostile") |
| P5 | Removing the construction-time raise lets the dual serializer path + storage/runtime split collapse | derived | CONFIRMED structurally: only caller of `runtime_model_config_from_spec` is `agents/serialize.py:156` (default); no production `runnable=False` caller; `model_construct` used only at `serialize.py:162` |
| P6 | `_create_model_from_config` (api branch) is the single runnable-model materialization seam | my-recall | UNVERIFIED — validator must confirm no other path builds a runnable API model from a `ModelConfig` bypassing it |
| P7 | Nothing depends on `_validate_api_key` populating `config.api_key` post-construction (pre-materialization) | derived | UNVERIFIED — grep shows only `agents.py:2921` + adapters (post-materialization) read `.api_key`; validator to confirm no other reader |
| P8 | Amending a prior session's frozen `acceptance.md` via dated notes is the sanctioned convention here | primary-source | CONFIRMED@`04-workflow-serializer/acceptance.md:117` (Session 08 / ADR-008 precedent) |
| P9 | Spren mirror deletion is an already-committed *separate* follow-up, not this session | primary-source | CONFIRMED@`04-workflow-serializer/acceptance.md:144` + memory `project_spren_modelconfig_mirror.md` |

### Frame check

Most load-bearing premise: the user's framing "make api_key optional → counterpart loses its reason → remove it." Falsifier sought: the `api_key` field declaration and the validator body. Found `models/models.py:108` (`Optional[str] = None`) and `:209-263` (env fallback already implemented, raise at construction). **The frame is partially wrong**: optionality is already done; the real defect is *resolution timing*. Reframed (user chose Option B): defer resolution to materialization; keep `ModelConfigSpec` for structural credential-leak safety; collapse the dual serializer path. This frame is internally consistent with primary source and supersedes — does not contradict — Session 04 (which deferred this exact fix).

## Risks

1. **Behavior-timing change (primary risk).** Code/tests expecting `ModelConfig(type="api", ...)` to raise *at construction* on missing creds will now see the error later (at materialization). Blast radius bounded by the validator; mitigation = explicit tests asserting the new timing + full regression gate. 3-cycle root-cause protocol on any failure.
2. **Prior-frozen-acceptance edit.** Editing Session 04's `acceptance.md` is allowed only via the dated-amendment convention (P8). Risk = doing it wrong/silently. Mitigation = exact Session-08/AC-59 pattern, surfaced in B5.
3. **Hidden materialization path (P6).** If a runnable API model can be built from a `ModelConfig` without going through `_create_model_from_config`, that path would skip resolution. Mitigation = validator confirms the seam before implementation; surface if found.
4. **Scope creep into Spren.** Tempting to "finish" by deleting the Spren mirror. Do not (P9, anti-pattern #8). Surface as the follow-up.

## Open questions (resolve at A5 with user)

- **Q1 — Resolution site.** Materialization-time (`_create_model_from_config`, when the runnable model is built) vs. first-`.run()`-call. Recommendation: **materialization-time** — fail when you build the runnable thing, not deep inside a request; matches the existing seam and "at runtime" intent. (Lean: adopt; confirm.)
- **Q2 — Spren scope.** Confirm Session 10 is framework-only and the Spren mirror deletion stays the documented follow-up PR. (Lean: yes, per P9.)
- **Q3 — Keep non-raising env warnings?** `_validate_api_key` currently `warnings.warn`s for unknown-provider / no-provider cases. Move to resolver or drop? (Lean: move the still-meaningful one to the resolver; warnings are not contract.)

## Done = Session 04's acceptance amended, ADR-010 written, single-path serializer, construction never raises/leaks, error at materialization, full regression green, Spren follow-up surfaced (not done).
