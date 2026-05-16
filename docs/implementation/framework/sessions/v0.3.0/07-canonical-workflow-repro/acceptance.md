# Session 07 — Frozen acceptance criteria

Frozen 2026-05-16, after reproduction + RCA + independent review, before
the fix is applied. The test-coverage-auditor sees only this file and the
test files — never the implementation.

Scope: the canonical `WorkflowDefinition` run-path defect (standard-API-key
providers materialize unrunnable `ModelConfig`) and the user-approved
Session-04 contract redesign that fixes it (`runtime_model_config_from_spec`
default flips to `runnable=True`; inspection becomes explicit
`runnable=False`).

## Deliverable: canonical-version test

- **AC-1** A script `packages/framework/benchmarks/GAIA/test_parallel_tracing_canonical.py`
  exists. It defines the same workflow as `test_parallel_tracing.py`
  (Coordinator → {Researcher, FactChecker} parallel, converging back to
  Coordinator) but expresses topology + agents through the canonical wire
  shape (`WorkflowDefinition`, `AgentSpec`, `TopologySpec`, `NodeSpec`,
  `EdgeSpec`, `ModelConfigSpec`) and hydrates a runnable `Topology` via
  `pydantic_to_topology`. It does NOT use the string-notation topology dict.
- **AC-2** Run as a script with the `anthropic-oauth` / `oauth_profile`
  config (same as the string-notation baseline), it completes the workflow
  with `result.success is True` and a non-empty final response — the same
  outcome class as the string-notation baseline.

## Contract: runtime materialization (`runtime_model_config_from_spec`)

- **AC-3** Default call (no `api_key`, no `runnable`) for a standard-API-key
  provider (`openai`/`anthropic`/`google`/`xai`/`openrouter`) **with the
  provider env var set** returns a runnable `ModelConfig`: `base_url`
  derived from the provider (non-None, matches the provider endpoint) and
  `api_key` resolved from the env var (equals the env value).
- **AC-4** Default call for a standard-API-key provider **with the provider
  env var unset** raises `ValueError` (same failure class a directly
  constructed `ModelConfig(provider=..., api_key=None)` raises). It does NOT
  return a config with `api_key=None`/`base_url=None`, and does NOT swallow
  the error.
- **AC-5** `runtime_model_config_from_spec(spec, runnable=False)` returns a
  `ModelConfig` whose every non-secret field equals the spec's (including
  `oauth_profile`), with `api_key is None`, and does **not** raise even when
  no credential is reachable and the env var is unset (inspection/storage
  contract — former AC-33 intent, now explicit opt-in).
- **AC-6** `runtime_model_config_from_spec(spec, api_key="sk-...")` returns
  a validated `ModelConfig` carrying the supplied key with `base_url`
  populated (Session-04 AC-34 parity — unchanged).
- **AC-7** For an `*-oauth` provider, the default (runnable) call returns a
  config without requiring any standard-provider env var (oauth key branch
  is a no-op; `oauth_profile` preserved). The canonical oauth e2e test
  (AC-2) exercising this path succeeds.

## Structural / contract-integrity

- **AC-8** The bug is fixed by the default flip alone: `pydantic_to_agents`
  (`agents/serialize.py`) and `pydantic_to_topology`
  (`coordination/topology/serialize.py`) keep their existing signatures and
  call sites — no new parameter threaded, no `model_credentials` param
  added (no speculative seam without a caller).
- **AC-9** `packages/framework/src/marsys/models/models.py` and
  `packages/framework/src/marsys/agents/agents.py` are not edited.
  Production change is confined to
  `packages/framework/src/marsys/models/serialize.py` (branching +
  docstrings).
- **AC-10** Session-04 frozen acceptance
  (`docs/implementation/framework/sessions/v0.3.0/04-workflow-serializer/acceptance.md`)
  is amended with dated, Session-07-attributed notes on AC-9 (new
  signature) and AC-33 (preservation guarantee now under `runnable=False`);
  AC-34 still holds and is noted as unaffected.
- **AC-11** The pre-existing test that pinned the old default
  (`test_runtime_model_config_from_spec_with_no_api_key_succeeds`) is
  rewritten to pin the new contract (inspection via explicit
  `runnable=False`); it is not deleted, skipped, or marked xfail. New tests
  cover AC-3 and AC-4.
- **AC-12** Framework regression for the touched modules stays green:
  `tests/agents/test_serialize.py`,
  `tests/coordination/topology/`,
  `tests/integration/test_workflow_definition_round_trip.py`.

## Reproduction evidence (must remain demonstrable)

- **AC-13** A deterministic, no-LLM check demonstrates the fix: a
  standard-provider `ModelConfigSpec` (e.g. `provider="anthropic"`, no
  `base_url`) with the provider env var set, run through the default
  `runtime_model_config_from_spec`, yields the same `base_url` and
  `api_key` as the equivalent directly-constructed `ModelConfig` (the
  string-notation path) — i.e. the pre-fix `base_url=None`/`api_key=None`
  divergence is gone.
