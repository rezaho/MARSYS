# Session 07 — Canonical workflow on the standard-API-key path: reproduce & fix the whole chain

**Status:** DONE — all defects reproduced, root-caused, fixed, merged to `main`
**Role:** implementer (investigative reproduce-then-fix; this doc IS the tracking contract per user request)
**Branch:** feature/tracing-streaming → `main`
**Created:** 2026-05-16 · **Consolidated:** 2026-05-16 (was Sessions 07/08/09 — one connected investigation, merged here)

> **Why one doc:** this started as "a user reports problems with the
> canonical workflow definition" and turned out to be a *chain of five
> independent defects plus one pre-existing side-finding*, every one of
> them on the standard-API-key path and structurally masked by OAuth.
> Each fix exposed the next (an onion). It was one investigation; it is
> recorded as one session. The three per-phase frozen-acceptance snapshots
> (Sessions 07/08/09) remain in git history at PRs **#39 / #41 / #42**;
> their criteria are consolidated in `07-canonical-workflow-repro/acceptance.md`.

---

## 1. Goal

A real user reported problems defining a workflow with the **canonical**
representation (the Session-04 `WorkflowDefinition` Pydantic wire shape:
`TopologySpec` + `AgentSpec` + `ExecutionConfigSpec`, hydrated via
`pydantic_to_topology`) instead of the string-notation topology dict.

Reproduce-first (do not assume the mechanism), root-cause via the bug-fix
protocol (ground → investigator → independent commit-before-exposure
review), fix each defect at the architecturally correct seam, and make the
canonical path run end-to-end exactly like the string-notation baseline —
on the **standard API-key** path, which is what real users use.

---

## 2. The chain (overview)

| # | Defect | Where | Status |
|---|--------|-------|--------|
| 1 | Canonical workflow materialized an **unrunnable `ModelConfig`** (no `base_url`/`api_key`) for *all* standard-API-key providers (not Anthropic-specific) | `models/serialize.py` | ✅ `main` (PR #39) |
| A | Anthropic adapter emitted tool-call `arguments` as a **dict, not a JSON string** → masked `TypeError` at step 1 | `adapters/anthropic.py` | ✅ `main` (PR #41) |
| C | OpenAI message-level **`name` leaked into the Anthropic request** → HTTP 400 | `adapters/anthropic.py` | ✅ `main` (PR #41) |
| B1 | Base **`MessageError` `TypeError`-masked** every malformed-message error (the masker behind A) | `agents/exceptions.py` | ✅ `main` (PR #42) |
| B2 | Async path **discarded provider 4xx bodies** → opaque "Bad Request" (hid C's real message) | `adapters/base.py` + `agents/exceptions.py` | ✅ `main` (PR #42) |
| side | Pre-existing, unrelated: `test_message_with_images` missing `mkdir` | `tests/memory/test_memory_manual.py` | ✅ `main` (PR #42) |
| f/u | Logged architecture follow-up (not done): `from_provider_response` transport-object coupling | — | 📋 separate session |

**The OAuth-masking theme:** OAuth providers structurally bypass every one
of defects 1/A/C/B1/B2 (different credential resolution, different adapter
response/request handling). That is why the reporter (standard API key)
hit them and every OAuth test stayed green.

---

## 3. Premise Ledger (entry investigation)

`Source` token per CLAUDE.md grounding discipline; a premise may not enter
a conclusion until `CONFIRMED@<cite>` or `RISK-LOGGED`.

| # | Premise | Source | Status |
|---|---------|--------|--------|
| P1 | A user hits issues using the canonical definition for topology + agents | user-stated (relayed third-hand; symptom unspecified) | CONFIRMED — reproduced empirically (Defect 1; A/C/B1/B2 are the chain it fronted) |
| P2 | `test_parallel_tracing.py` (string-notation) runs to success — the baseline | user-stated | CONFIRMED@run (15 steps, success) |
| P3 | "canonical version" = the Session-04 `WorkflowDefinition` wire shape hydrated via `pydantic_to_topology` | my-recall (inferred) @ `coordination/topology/serialize.py:1-35` | CONFIRMED — interpretation held |
| P4 | Canonical run path: `WorkflowDefinition` → `pydantic_to_topology(spec, tool_registry)` → `Orchestra.run(topology=<Topology>, …)`; `WorkflowDefinition` cannot be passed to `Orchestra.run` directly | primary-source | CONFIRMED@`coordination/orchestra.py:426-441` |
| P5 | `Agent.__init__` auto-registers into the global `AgentRegistry` | primary-source | CONFIRMED@`agents/agents.py:253` |

**Frame check.** Most load-bearing premise: P1 — the bug exists in the
canonical run path. Falsifier: the canonical twin runs end-to-end with the
same success outcome as the string-notation baseline. The reproduction step
existed precisely to look for that falsifier before any RCA. Result: the
OAuth canonical twin **passed** (falsifier observed for OAuth) — narrowing
to the realistic standard-API-key case confirmed Defect 1, which then
fronted the rest of the chain. No root cause was pre-committed at any hop.

---

## 4. Defect 1 — canonical workflow builds an unrunnable `ModelConfig`

**Scope note (accuracy):** this defect is **not** Anthropic-specific. It
hits every standard-API-key provider (`openai`/`anthropic`/`google`/
`xai`/`openrouter`). Defects A and C below *are* the Anthropic-adapter-
specific ones.

**Reproduction.** Baseline string-notation `test_parallel_tracing.py`
(OAuth) → Success. Canonical twin (OAuth) → **Success** (frame falsifier).
Deterministic no-LLM probe with `ANTHROPIC_API_KEY` set:

```
Normal/string path   ModelConfig(provider='anthropic', ...)
  base_url = 'https://api.anthropic.com/v1' ; api_key = '<from env>'
Canonical path       ModelConfigSpec(provider='anthropic') -> runtime_model_config_from_spec()
  base_url = None ; api_key = None
```

**Root cause.** `pydantic_to_agents` (`agents/serialize.py:156`)
materialized each agent's runtime config via
`runtime_model_config_from_spec(agent_spec.agent_model)` with **no
`api_key`**. That function's no-key branch (`models/serialize.py:140`)
used `ModelConfig.model_construct(api_key=None, **fields)`, which bypasses
**every** validator — including `_set_base_url_from_provider`
(`models.py:183-207`, provider→`base_url`) and `_validate_api_key`
(`models.py:209-263`, env-var credential). Standard providers got no
endpoint and no credential. OAuth dodged it: `_validate_api_key`'s oauth
branch is a no-op (`models.py:245-248`) and the oauth adapter resolves
both from `oauth_profile` at client-init (`models.py:524-536`). The no-key
branch was a deliberate Session-04 choice for storage-time materialization
(`models/serialize.py:1-17`); the defect is that `pydantic_to_agents` is a
**runtime** path routing through the inspection-grade branch.

**Fix (user-decided: redesign the contract, not patch the call site).**
The real flaw is that `runtime_model_config_from_spec`'s *silent default*
was the non-runnable inspection branch — a footgun for any runtime caller.
Added a keyword-only `runnable: bool = True` axis (orthogonal to
`api_key`):

- `runnable=True` (**new default**): normal `ModelConfig(...)` constructor
  — derives `base_url`, resolves the env credential, **raises `ValueError`
  if missing** (string-notation parity); oauth branch no-op.
- `runnable=False`: `model_construct(...)` — validators bypassed, never
  raises, fields preserved, NOT runnable. The former default, now an
  explicit, honest opt-in (storage/inspection: community templates,
  MARSYS Cloud pre-deploy, Spren persistence).

`pydantic_to_agents` calls `runtime_model_config_from_spec(...)` with no
args — **the default flip *is* the fix**, no call-site change. Production
change confined to `models/serialize.py` (branching + docstrings);
`models/models.py`, `agents/agents.py`, `pydantic_to_agents`/
`pydantic_to_topology` signatures all untouched; no speculative
`model_credentials` seam. Session-04 frozen acceptance amended (dated,
Session-07-attributed) on AC-9 (signature) / AC-33 (preservation now under
`runnable=False`) / AC-34 (unaffected). Bug-encoding test rewritten to pin
the new contract via explicit `runnable=False` + new env-resolve /
raise-on-missing / parity tests (not test-silencing — a user-approved
contract change re-pinned). Independent commit-before-exposure review:
SOUND-WITH-CONCERNS, all concerns folded in.

---

## 5. Defect A — Anthropic tool-call `arguments` emitted as a dict

Surfaced by exercising the now-working standard-`anthropic` path: after a
**successful** Anthropic `tool_use` response the framework crashed with
`TypeError: AgentFrameworkError.__init__() got an unexpected keyword
argument 'message_id'` at step 1.

**Root cause.** `adapters/anthropic.py` `harmonize_response` set tool-call
`arguments` to the raw `tool_use.input` **dict**. The canonical contract at
the `ToolCallMsg` boundary is a JSON **string** (`memory.py:179` typed
`str`, enforced `:191-192`, consumed via `json.loads` at
`coordination_tools.py:55` / `design-principles.md:68`). Every other
producer (`anthropic_oauth`, `google` `json.dumps`, `openai` native,
`openrouter`, local `models/utils.py`) emits a string; standard
`anthropic` was the sole violator. The raised `ValueError` was then
*masked* into the confusing `message_id` `TypeError` by **Defect B1**.

**Fix.** `adapters/anthropic.py` → `json.dumps(block.get("input", {}))`
(+ WHY comment), at the adapter seam per DP-006. No `ToolCallMsg`
loosening, no `try/except`. Regression: `tests/models/test_adapter_harmonize.py`
(JSON-string + faithful round-trip). Investigator + independent
commit-before-exposure review: SOUND.

---

## 6. Defect C — OpenAI `name` leaked into the Anthropic request → HTTP 400

With A fixed the workflow ran 1→15 steps, then HTTP 400. The adapter only
captured `error='Bad Request'` (that opacity is **Defect B2**); real body
captured via temporary instrumentation (reverted):
`messages.N.name: Extra inputs are not permitted`.

**Root cause.** `format_request_payload`'s plain-message `else` branch did
`cleaned_msg = msg.copy()`, carrying the OpenAI message-level `name`
(agent identity from `memory.to_llm_dict`) into the request. Anthropic's
Messages API permits only `role`/`content`. The OAuth twin
(`anthropic_oauth.py` regular-message branch) rebuilds a clean
`{role, content}` dict — which is why OAuth never hit it. (The
investigator's initial multi-tool-call hypothesis was **refuted** by the
real 400 body — grounding working as intended.)

**Fix.** Rebuild the `else` branch to emit only `{role, content}`,
mirroring the proven OAuth twin. Regression added in
`tests/models/test_adapter_harmonize.py` (AC-7: wire-legal keys; tool /
assistant-with-tool_calls branches unaffected). Self-reviewed vs primary
source + the OAuth twin; localized single-file fix verified by the API's
own error message.

---

## 7. Defect B1 — base `MessageError` `TypeError`-masks malformed-message errors

The masker behind Defect A, and the reason C's 400 cost a full
investigation.

**Reproduction (deterministic, no LLM).**
`MessageError('x', invalid_data={'x':1}, expected_format='ToolCallMsg')` →
`TypeError: ... unexpected keyword argument 'invalid_data'`; and
`Message(role='assistant', content='x', tool_calls=['bad'])` →
`Message.__post_init__` guard → `TypeError: ... 'message_id'` — the **exact**
original symptom from the first standard-anthropic run, proving B1 masked A.

**Root cause (`exceptions.py:121-128`).** Base `MessageError.__init__`
popped only `error_code` and blind-forwarded `**kwargs` to
`AgentFrameworkError.__init__`'s closed 7-param signature
(`exceptions.py:52-61`). It is the *only* class in the `MessageError`
family that violates the family invariant — every sibling
(`MessageFormatError` `:141-164`, `MessageContentError` `:177-200`, …)
already folds diagnostic kwargs into `context` and forwards only legal
kwargs. The ~14 `memory.py` raise sites are correct; the base class was the
defect.

**Fix.** Rewrote the base `MessageError.__init__` to fold
`message_id`/`invalid_data`/`expected_format`/`validation_path` into a
size-bounded `context` and forward only clean kwargs. The 14 `memory.py`
sites are unchanged. 8 bug-encoding `pytest.raises((MessageError,
TypeError))` tuples tightened → `MessageError` (the `(…, KeyError,
TypeError)` site drops only the bug arm, keeps the legitimate `KeyError`)
so the regression cannot silently return.

**Implementer-caught regression both review agents missed:** the proposed
shape also set the diagnostics as `self.*` attributes on the base. Verified
empirically that this clobbers siblings —
`MessageFormatError(...).expected_format` became `None` (the base ran
`self.expected_format = None` *after* the sibling set the real value, since
siblings consume—don't forward—that kwarg), failing
`test_exceptions.py::test_message_format_error`. Resolution: base folds into
`context` only, sets **no** attributes (frame check established zero
consumers read base-`MessageError` diagnostic attributes). This is the
grounding discipline + implementer-as-second-pair-of-eyes catching what an
investigator AND an independent reviewer both missed.

---

## 8. Defect B2 — async path discards provider 4xx bodies

The reason Defect C surfaced as opaque "Bad Request" instead of the real
Anthropic message.

**Root cause (two coupled, both required).**
- **R1** `adapters/base.py` async path passed `response=None` to
  `handle_api_error` while the live aiohttp response was still in scope;
  aiohttp's `ClientResponseError` carries only the reason phrase, and
  `.json()` is a coroutine the sync mapper calls synchronously — so the
  4xx body was discarded.
- **R2** `from_provider_response`'s anthropic branch had no generic-4xx arm
  (only matched 400 on `"credit balance"`), so even with a body a 400
  classified `UNKNOWN`.

**Fix (shim — user-chosen after a scoped design fork; see §9).** R1: read
the body in-frame and wrap it in `_CapturedErrorResponse`, shaped exactly
like the `requests.Response` the **sync** path already passes to
`handle_api_error` — zero change to `from_provider_response` or the 6
provider branches; sync path untouched. `.json()` is idempotent/cached
(the anthropic wrapper re-reads it post-classification when the error is
non-critical) and headers are a case-insensitive `CIMultiDict` snapshot.
R2: add a generic anthropic 400 arm (`INVALID_REQUEST`, terminal) at
parity with the openrouter / anthropic-oauth arms; `message` is already set
from the body upstream so the arm only classifies.

The prior proposed mechanism ("pass the parsed dict as `response`") was
**refuted by independent commit-before-exposure review** —
`from_provider_response` extracts the body via `hasattr(response,'json')`
and `if response:`, so a bare dict is discarded/falsy and B2 would stay
broken. Re-specced to the shim; investigator + independent review
converged SOUND-WITH-CONCERNS (idempotent `.json()`, CIMultiDict headers,
joint R1+R2 regression — all folded in). Verified against the **live
Anthropic API**: a forced 400 now reports `temperature: range: 0..1` with
classification `invalid_request`, instead of `Bad Request`/`unknown`.
Regression: `tests/models/test_async_error_body.py` (joint R1+R2 guard).

---

## 9. Architectural follow-up — LOGGED, not done (user-directed)

`ModelAPIError.from_provider_response` is coupled to a **transport
object**: it duck-types `response.status_code` / `response.json()` /
`response.headers` (`hasattr`-gated, truthiness-gated). That coupling is
why the sync (`requests`) and async (aiohttp coroutine `.json()`) paths
don't compose and why `_CapturedErrorResponse` must exist to bridge them.

The clean shape is an explicit `APIErrorContext(status_code, body,
headers)` consumed by `handle_api_error` / `from_provider_response`, with
the exception/raw-stream fallback retained for the 8 OAuth-**streaming**
callers that legitimately have no HTTP response. Primary-source
enumeration during B2 showed `from_provider_response` is a polymorphic
factory called **~14 times in 3 usage shapes** across **7
`handle_api_error` definitions**, incl. OAuth-streaming sites
(`anthropic_oauth.py:787,800,846,859`, `openai_oauth.py:364,374,810,820`)
with no response/status/headers. A full-strip would force unrelated,
lightly-tested streaming error paths to fabricate empty context — smuggled
scope for zero B2 benefit. The user chose to **keep the shim and log the
smell**. This is a scoped cross-provider error-handling contract change for
its **own architecture session**, not a bug-fix tail. In-code pointer:
`_CapturedErrorResponse`'s docstring in `adapters/base.py`.

---

## 10. Side-finding — pre-existing, unrelated (NOT part of the chain)

`tests/memory/test_memory_manual.py::test_message_with_images` wrote a PNG
into `packages/framework/tmp/screenshots/` without
`mkdir(parents=True, exist_ok=True)` → `FileNotFoundError` on a clean
checkout. Verified it fails identically on clean HEAD with all chain
changes stashed — genuinely pre-existing, caused by none of this work.
Fixed as its **own one-line commit**, deliberately not bundled into any
defect fix (anti-pattern #8: no smuggled scope).

---

## 11. Outcome & verification

**Production changes (by file):**
- `models/serialize.py` — Defect 1: `runnable` axis + docstrings.
- `adapters/anthropic.py` — Defect A: `json.dumps` tool-args; Defect C:
  `{role, content}` rebuild of the plain-message branch.
- `agents/exceptions.py` — Defect B1: `MessageError.__init__` context-fold;
  Defect B2: anthropic generic-400 arm.
- `adapters/base.py` — Defect B2: `_CapturedErrorResponse` shim + in-frame
  body capture.
- `tests/memory/test_memory_manual.py` — side-finding `mkdir`.

**Tests added/changed:** `benchmarks/GAIA/test_parallel_tracing_canonical.py`
(canonical OAuth twin), `…_canonical_anthropic.py` (standard-API twin),
`test_anthropic_4xx_error_body.py` (B2 demo), `tests/models/test_adapter_harmonize.py`,
`tests/models/test_async_error_body.py`, plus contract-pinning rewrites in
`tests/agents/test_serialize.py` and tightenings in
`tests/{memory/test_memory.py,agents/test_memory.py}`.

**Verification:**
- Standard-`anthropic` e2e `test_parallel_tracing_canonical_anthropic.py`
  → `Success: True` (12–14 steps) — the chain is traversed end-to-end.
- OAuth e2e `test_parallel_tracing_canonical.py` → `Success: True`
  (no success-path regression — the B2 shim only builds on non-200).
- B2 live-API demo `test_anthropic_4xx_error_body.py` → `B2 WORKING`
  (real Anthropic message + `invalid_request`, not "Bad Request").
- Deterministic: Defect-1 parity probe; B1 before→`TypeError`/
  after→usable `MessageError`; B2 joint R1+R2 guard.
- Regression: 456 passed / 0 failed (agents + models + memory) on the
  final rebased tip.
- Every defect went through investigator + independent
  commit-before-exposure review; the B1 sibling-clobber was caught by
  implementer verification both agents missed.

**Merged:** PR #39 (Defect 1), PR #41 (A + C), PR #42 (B1 + B2 + side).
`origin/main` @ `9b89356`.

---

## 12. Acceptance

Consolidated and frozen at `07-canonical-workflow-repro/acceptance.md`
(criteria from the three per-phase frozen sets, organized by defect). The
original per-phase snapshots remain immutable in git history at PRs
#39 / #41 / #42. The test-coverage-auditor reads only that file + the test
files — never the implementation.
