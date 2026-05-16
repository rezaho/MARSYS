# Session 07 — Consolidated frozen acceptance criteria

Consolidated 2026-05-16 from the three per-phase frozen sets (originally
Sessions 07/08/09). The immutable per-phase snapshots remain in git history
at PRs **#39 / #41 / #42** — this file is the merged, by-defect view. The
test-coverage-auditor reads only this file and the test files, never the
implementation. Each criterion was frozen *before* its fix was applied.

---

## Defect 1 — canonical workflow / `runtime_model_config_from_spec` contract

- **D1-AC-1** A script `benchmarks/GAIA/test_parallel_tracing_canonical.py`
  exists, defines the same workflow as `test_parallel_tracing.py`
  (Coordinator → {Researcher, FactChecker} parallel, converging back) but
  via the canonical wire shape (`WorkflowDefinition`/`AgentSpec`/
  `TopologySpec`/`NodeSpec`/`EdgeSpec`/`ModelConfigSpec`) hydrated by
  `pydantic_to_topology`; it does NOT use the string-notation dict.
- **D1-AC-2** Run with the `anthropic-oauth`/`oauth_profile` config, it
  reaches `result.success is True` with a non-empty final response (same
  outcome class as the string-notation baseline).
- **D1-AC-3** Default `runtime_model_config_from_spec(spec)` (no `api_key`,
  no `runnable`) for a standard provider (`openai`/`anthropic`/`google`/
  `xai`/`openrouter`) **with the env var set** returns a runnable
  `ModelConfig`: `base_url` provider-derived (non-None), `api_key` = env
  value.
- **D1-AC-4** Same with the env var **unset** raises `ValueError` (same
  failure class as a directly-constructed `ModelConfig(provider=...,
  api_key=None)`); does NOT return `api_key=None`/`base_url=None`, does NOT
  swallow.
- **D1-AC-5** `runtime_model_config_from_spec(spec, runnable=False)`
  returns a `ModelConfig` whose every non-secret field equals the spec's
  (incl. `oauth_profile`), `api_key is None`, and does **not** raise even
  with no credential reachable (inspection/storage contract — explicit
  opt-in).
- **D1-AC-6** `runtime_model_config_from_spec(spec, api_key="sk-...")`
  returns a validated `ModelConfig` with the key and `base_url` populated
  (Session-04 AC-34 parity — unchanged).
- **D1-AC-7** For an `*-oauth` provider the default (runnable) call returns
  a config without any standard-provider env var (oauth branch no-op;
  `oauth_profile` preserved); the OAuth e2e (D1-AC-2) exercising this
  succeeds.
- **D1-AC-8** Fixed by the default flip alone: `pydantic_to_agents` /
  `pydantic_to_topology` keep existing signatures and call sites; no new
  parameter, no `model_credentials` seam.
- **D1-AC-9** `models/models.py` and `agents/agents.py` not edited;
  production change confined to `models/serialize.py`.
- **D1-AC-10** Session-04 frozen acceptance amended with dated,
  Session-07-attributed notes on AC-9 (signature) and AC-33 (preservation
  now under `runnable=False`); AC-34 noted unaffected.
- **D1-AC-11** The pre-existing `…_with_no_api_key_succeeds` test is
  rewritten to pin the new contract via explicit `runnable=False` (not
  deleted/skipped/xfail); new tests cover D1-AC-3 and D1-AC-4.
- **D1-AC-12** Framework regression green: `tests/agents/test_serialize.py`,
  `tests/coordination/topology/`,
  `tests/integration/test_workflow_definition_round_trip.py`.
- **D1-AC-13** Deterministic no-LLM check: a standard-provider
  `ModelConfigSpec` (no `base_url`) with the env var set, via the default
  `runtime_model_config_from_spec`, yields the same `base_url` + `api_key`
  as the equivalent directly-constructed `ModelConfig` — the pre-fix
  `None`/`None` divergence is gone.

## Defect A — Anthropic tool-call `arguments` JSON string

- **A-AC-1** `AnthropicAdapter.harmonize_response`, given a `tool_use`
  block whose `input` is an object, returns a `HarmonizedResponse` whose
  `tool_calls[0].function["arguments"]` is a `str` and `json.loads` of it
  equals the original `input` (faithful, not `repr`/`str`).
- **A-AC-2** Same with empty `input` (`{}`): `arguments` is the string
  `"{}"`, never a dict or `None`.
- **A-AC-3** End-to-end
  `test_parallel_tracing_canonical_anthropic.py` (standard `anthropic`,
  `ANTHROPIC_API_KEY` from repo `.env`) reaches `result.success is True`,
  non-empty response, exit 0; the pre-fix `TypeError`/`message_id` crash
  does not occur.
- **A-AC-4** No OAuth regression: `test_parallel_tracing_canonical.py`
  still reaches `Success: True`.

## Defect C — no OpenAI `name` leak into Anthropic request

- **C-AC-1** `format_request_payload`, given a plain assistant/user
  message carrying a message-level `name` (as `memory.to_llm_dict`
  produces), emits Anthropic messages whose keys are a subset of
  `{role, content}` (no `name`/OpenAI-only keys); the tool /
  assistant-with-tool_calls branches still emit wire-legal shapes.
- **C-AC-2** Fix confined to the adapter seam: only
  `adapters/anthropic.py` changes in production code (Defects A+C are two
  localized changes in that one file). `ToolCallMsg`'s string check
  (`memory.py`) is NOT loosened; no `try/except`; no `_v2`/variant file.
- **C-AC-3** Touched-module regression green (adapter/model + agent
  serialize suites that exercise harmonized tool calls).

## Defect B1 — `MessageError` absorbs diagnostic kwargs

- **B1-AC-1** Constructing base `MessageError` with any of
  `message_id`/`invalid_data`/`expected_format`/`validation_path` returns
  a usable `MessageError` (no `TypeError`); values present in
  `error.context` (size-bounded, like sibling subclasses).
- **B1-AC-2** Realistic path —
  `Message(role="assistant", content="x", tool_calls=["bad"])` and the
  other `memory.py` guards — raises a usable `MessageError` (was the
  `message_id` `TypeError`). The ~14 `memory.py` raise sites are
  UNCHANGED (fix in the base class only).
- **B1-AC-3** No sibling regression: `MessageFormatError` /
  `MessageContentError` / `SchemaValidationError` still expose their own
  diagnostic attributes (e.g. `MessageFormatError(...).expected_format`
  is the supplied value, NOT clobbered to `None`); the base sets no such
  attributes.
- **B1-AC-4** `AgentFrameworkError.to_dict()` / `.context` shape not
  broken — diagnostics added as `context` keys (enrichment), nothing
  removed. `AgentFrameworkError`, `memory.py`, the adapters not modified
  for B1.
- **B1-AC-5** The bug-encoding `pytest.raises((MessageError, TypeError))`
  tuples (7 sites) tightened to `pytest.raises(MessageError)`; the
  `(MessageError, KeyError, TypeError)` site drops only the `TypeError`
  bug arm, keeps `KeyError`. No skip/xfail.
- **B1-AC-6** Touched-module regression green:
  `tests/agents/test_exceptions.py`, `tests/memory/`,
  `tests/agents/test_memory.py`.
- **B1-AC-7** No standard-anthropic e2e regression (B1 is on the
  exception path every workflow runs);
  `test_parallel_tracing_canonical_anthropic.py` still `Success: True`.

## Defect B2 — async API 4xx provider body surfaced

- **B2-AC-1** `_CapturedErrorResponse` exposes exactly `status_code:int`,
  idempotent cached `json()->dict` (raises `ValueError` when no body —
  callers already swallow that), case-insensitive `headers`.
- **B2-AC-2** Joint R1+R2: `AnthropicAdapter.handle_api_error(<aiohttp
  ClientResponseError>, response=<shim carrying a 400
  invalid_request_error body>)` returns an `ErrorResponse` whose `error`
  is the provider's real message (NOT "Bad Request") and
  `classification.category == "invalid_request"`, `is_retryable False`.
  Fails on pre-fix code; passes only with both R1 and R2.
- **B2-AC-3** A body-less 400 (response=None, status from the exception)
  → message falls back to the reason phrase, classification
  `invalid_request` (correct — a 400 is an invalid request regardless;
  pre-fix wrongly `unknown`). A true connection error (no status) stays
  `unknown` with the exception text (no regression).
- **B2-AC-4** Fix confined to `adapters/base.py` (shim + in-frame capture)
  + the anthropic 400 arm in `agents/exceptions.py`. `from_provider_response`
  signature, the 6 provider branches, the sync path, the 8 OAuth-streaming
  callers, and `memory.py` all UNCHANGED. No `try/except` mask beyond the
  documented non-JSON-body fallback; no variant file.
- **B2-AC-5** Live-API demo `benchmarks/GAIA/test_anthropic_4xx_error_body.py`
  forces a real Anthropic 400 and prints a clear B2 PASS verdict (real
  `temperature` message + `invalid_request`).
- **B2-AC-6** Touched-module regression green (models + agents);
  standard-anthropic e2e no success-path regression.

## Side-finding (pre-existing, unrelated — not a chain defect)

- **S-AC-1** `tests/memory/test_memory_manual.py::test_message_with_images`
  creates `tmp/screenshots/` (`mkdir(parents=True, exist_ok=True)`) before
  writing, fixed as its own one-line commit, NOT bundled into any defect
  fix. (It failed identically on clean HEAD pre-change — genuinely
  pre-existing.)
