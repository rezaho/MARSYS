# Session 09 — `MessageError` kwarg mismatch + API 4xx body swallowing (Root B)

**Status:** IN PROGRESS — grounding/reproduction
**Role:** implementer (bug-fix protocol; the latent defect surfaced + deferred by Sessions 07/08)
**Branch:** feature/tracing-streaming (Sessions 07 & 08 merged to main)
**Created:** 2026-05-16

---

## 1. Goal

"Root B" — the latent defect that *masked* Session-08's Root A (turned a
useful `ValueError`/`MessageError` into an opaque `TypeError`) and made
Session-08's Root C cost a full investigation (the API 400 body was hidden).
Two sub-defects:

- **B1 — exception-API mismatch.** `memory.py` raises the **base**
  `MessageError(..., message_id=, invalid_data=, expected_format=,
  validation_path=)` at ~14 sites. `MessageError.__init__`
  (`exceptions.py:121-128`) pops only `error_code` and blindly forwards the
  rest via `**kwargs` to `AgentFrameworkError.__init__`
  (`exceptions.py:52-61`), which accepts only
  `message/error_code/agent_name/task_id/context/user_message/suggestion`.
  Any of those diagnostic kwargs ⇒ `TypeError: AgentFrameworkError.__init__()
  got an unexpected keyword argument '<name>'`. So whenever a real
  message/tool-call validation guard fires, the intended `MessageError` is
  destroyed and replaced by a misleading `TypeError`.
- **B2 — diagnostics black hole.** The async API error path
  (`adapters/base.py` ~585/591) calls `handle_api_error(e, response=None)`,
  and the `anthropic` branch in `exceptions.py` (~781-797) has no generic
  4xx handling, so an API 400's descriptive JSON body degrades to the
  opaque aiohttp reason phrase `"Bad Request"`.

Goal: a malformed message/tool-call yields a **usable** `MessageError`
(diagnostic data preserved), and API 4xx errors surface the provider's
actual message.

## 2. Premise Ledger

| # | Premise | Source | Status |
|---|---------|--------|--------|
| P1 | `MessageError.__init__` forwards unknown `**kwargs` to `AgentFrameworkError.__init__` which rejects them | primary-source | CONFIRMED@`exceptions.py:121-128` + `:52-61` |
| P2 | `memory.py` raises base `MessageError(..., invalid_data=, expected_format=, message_id=)` at ~14 sites | primary-source | CONFIRMED@`memory.py:282-348` (grep: invalid_data/expected_format at :285-348; message_id/validation_path at other sites) |
| P3 | The intended design is "absorb extras into context" — siblings already do it | primary-source | CONFIRMED@`exceptions.py:141-164` (`MessageFormatError`) + `:177-200` (`MessageContentError`) fold extras into `context`, forward only clean kwargs |
| P4 | B1 is latent on the success path (guards only fire on malformed input) | primary-source (Session 08 e2e green) | CONFIRMED — standard-anthropic e2e Success after Root A/C; B1 dormant |
| P5 | B2: async error path passes `response=None`; no generic-4xx anthropic branch | agent-brief (Session-08 investigators) | UNVERIFIED — re-confirm from `adapters/base.py` + `exceptions.py` this session |

## 3. Frame check

Most load-bearing: P3 — "the fix is to make the base `MessageError` absorb
the diagnostic kwargs into `context`, like its siblings, NOT to strip the
kwargs at the ~14 call sites (which would discard diagnostic intent)."
Falsifier to look for: a consumer that already reads
`exc.invalid_data`/`exc.message_id` as attributes (then absorbing into a
generic `context` dict would silently break it). The investigator must grep
for attribute access on these exceptions before choosing the shape. The
async-400 sub-defect (B2) must be re-derived from `base.py`/`exceptions.py`
this session — Session-08 reports are `agent-brief` here.

## 4. Reproduction

Deterministic, no LLM, two ways (CONFIRMED):

- Direct: `MessageError('bad tool call', invalid_data={'x':1},
  expected_format='ToolCallMsg')` → `TypeError: AgentFrameworkError.__init__()
  got an unexpected keyword argument 'invalid_data'`.
- Realistic path: `Message(role='assistant', content='x',
  tool_calls=['not-a-toolcall-dict'])` → `Message.__post_init__` guard →
  `MessageError(..., message_id=..., ...)` → `TypeError: ... unexpected
  keyword argument 'message_id'` — the **exact** original symptom from the
  first standard-anthropic run, proving B1 is what masked Session-08 Root A.

## 5. Root-cause analysis

Investigator + independent commit-before-exposure reviewer converged:

- **B1 (irreducible @`exceptions.py:121-128`):** base `MessageError.__init__`
  is the *only* class in the `MessageError` family that violates the
  family invariant (siblings absorb diagnostic kwargs into `context` +
  forward only `AgentFrameworkError`-legal kwargs). It blind-forwards
  `**kwargs` into a 7-param whitelist → `TypeError` on any diagnostic
  kwarg, destroying the real `MessageError`. Independent review: **B1
  SOUND** (root, seam, blast radius, frame check all confirmed; no
  cross-package/`__all__` consumer; zero attribute-read consumers).
- **B2 (root sound, fix mechanism REFUTED):** async path discards the live
  response body (`base.py:585,591` `response=None`) + anthropic branch has
  no generic-400 arm (`exceptions.py:791`). The independent review caught
  that the investigator's proposed B2 fix ("pass the parsed dict as
  `response`") does NOT work — `from_provider_response` extracts the body
  via `hasattr(response,'json')` (`exceptions.py:747`), so a bare dict is
  silently discarded and B2 stays broken. B2 needs a response-shaped shim
  or an explicit pre-parsed-body parameter — **sent back; separable.**

**Implementer-caught regression (both agents missed it):** the
investigator's B1 shape *also* set the diagnostics as `self.*` attributes
on the base. Empirically verified that this clobbers siblings:
`MessageFormatError(...).expected_format` became `None` (base runs
`self.expected_format = None` *after* the sibling set the real value,
because siblings consume—don't forward—that kwarg), failing
`test_exceptions.py::test_message_format_error`. Resolution: base folds
diagnostics into `context` only, does **not** set attributes (frame check
already established zero consumers read base-`MessageError` attributes).

## 6. Proposed solution & status

- **B1 — APPLIED & VERIFIED.** `exceptions.py` base `MessageError.__init__`
  rewritten: pop `message_id`/`invalid_data`/`expected_format`/`validation_path`,
  fold into size-bounded `context`, pop `error_code`, forward clean kwargs;
  no attribute-setting (no sibling clobber). 14 `memory.py` sites unchanged.
  8 bug-encoding `pytest.raises` tuples tightened (7 → `MessageError`; the
  KeyError site drops only the `TypeError` bug arm). Verified: deterministic
  repro before→`TypeError` / after→usable `MessageError`; siblings intact
  (`test_exceptions.py` green); memory suites 245 passed (1 pre-existing
  unrelated `test_message_with_images` FileNotFound — fails identically on
  clean HEAD, NOT in scope, surfaced separately).
- **B2 — APPLIED & VERIFIED (shim approach, user-chosen).** Re-specced
  after the independent review refuted the prior "pass a parsed dict"
  mechanism. Investigator + independent commit-before-exposure review
  converged SOUND-WITH-CONCERNS; all concerns folded in (idempotent cached
  `.json()`; case-insensitive `CIMultiDict` header snapshot; joint R1+R2
  regression test). Fix: `_CapturedErrorResponse` shim in `base.py` (read
  the 4xx body in-frame, shaped like the sync path's `requests.Response`)
  + anthropic generic-400 arm in `from_provider_response`
  (`INVALID_REQUEST`, terminal). `models/serialize.py` / `memory.py` /
  the other 5 adapters untouched. Verified: `test_async_error_body.py`
  (4 tests incl. the joint R1+R2 guard) green; touched-module regression;
  standard-anthropic e2e no-regression.
  - **Mid-flight design fork (user-decided):** the deeper "(b)" refactor —
    replace the shim by making `handle_api_error`/`from_provider_response`
    take extracted `(status, body, headers)` — was scoped during
    implementation. Primary-source enumeration showed `from_provider_response`
    is a polymorphic factory called **~14 times in 3 shapes**, incl. 8
    OAuth-**streaming** call sites that have no HTTP response/status/headers
    at all (`anthropic_oauth.py:787,800,846,859`,
    `openai_oauth.py:364,374,810,820`). Full-strip would force unrelated,
    lightly-tested streaming error paths to fabricate an empty context —
    smuggled scope (anti-pattern #8) for zero B2 benefit. Surfaced with the
    corrected blast radius; **user chose: keep the shim**, log the smell.

### Architectural follow-up (LOGGED, not done — user-directed)

`ModelAPIError.from_provider_response` is coupled to a *transport object*:
it duck-types `response.status_code` / `response.json()` / `response.headers`
(`hasattr`-gated, truthiness-gated). That coupling is why the sync path
(`requests.Response`) and the async path (aiohttp, coroutine `.json()`)
don't compose, and why `_CapturedErrorResponse` must exist to bridge them.
The clean shape is an explicit `APIErrorContext(status_code, body, headers)`
value object consumed by `handle_api_error` / `from_provider_response`,
with the exception/raw-stream fallback retained for the 8 OAuth-streaming
callers (which legitimately have no response). This is a cross-provider
error-handling contract change (~14 call sites, 3 usage shapes, 7
`handle_api_error` definitions) — a scoped architecture session, NOT a
bug-fix tail. Tracked here; `_CapturedErrorResponse`'s docstring is the
in-code pointer.
- **Pre-existing side-finding (NOT Root B, NOT fixed):**
  `tests/memory/test_memory_manual.py::test_message_with_images` writes to
  `packages/framework/tmp/screenshots/` without `mkdir(parents=True,
  exist_ok=True)` → `FileNotFoundError` on a clean checkout. One-line test
  fix; deliberately not bundled (anti-pattern #8). Surfaced for a separate
  change.

**Status: B1 done & verified. B2 root-caused, fix re-spec pending. One
unrelated pre-existing test-harness bug surfaced.**

## 7. Acceptance

Frozen at `09-messageerror-kwarg-mismatch/acceptance.md` before the fix.
