# Session 08 — Standard-`anthropic` response path: `message_id` TypeError

**Status:** IN PROGRESS — reproduced, investigating
**Role:** implementer (bug-fix protocol; standalone, surfaced by Session 07's Anthropic-API test)
**Branch:** feature/tracing-streaming
**Created:** 2026-05-16

---

## 1. Goal

Session 07 fixed canonical workflows materializing an unrunnable
`ModelConfig` on standard-API-key providers. Exercising the now-working
standard-`anthropic` path end-to-end (the Anthropic-API canonical test)
surfaced a **second, independent** bug: after a *successful* Anthropic
model response, the framework crashes processing it with

```
LLM call failed: AgentFrameworkError.__init__() got an unexpected keyword argument 'message_id'   (TypeError)
```

OAuth never hit this (different adapter response handling), which is why no
prior test caught it. Standard API keys are the common user case, so this
is very likely part of "the issues users hit with canonical." Find the true
root cause, fix it so the Anthropic-API canonical test reaches
`Success: True`, no regression to the oauth path.

## 2. Premise Ledger

| # | Premise | Source | Status |
|---|---------|--------|--------|
| P1 | The Anthropic API call itself **succeeds** (valid `tool_use` response, `invoke_agent` for Researcher+FactChecker, 3076 tokens, `finish_reason='tool_use'`) | primary-source | CONFIRMED@`results/traces_parallel_canonical_anthropic_test/_run.log` (the `Model claude-haiku-4-5-20251001 response: role='assistant' ... tool_calls=[...]` line) |
| P2 | Session-07 fix works on the real standard-API path: credential resolved from env | primary-source | CONFIRMED@`_run.log` (`Read API key for provider 'anthropic' from env var 'ANTHROPIC_API_KEY'`) |
| P3 | The crash is a `TypeError: AgentFrameworkError.__init__() got an unexpected keyword argument 'message_id'`, raised while processing the successful response, then wrapped by the agent into a `MODEL_ERROR` dict | primary-source | CONFIRMED@`_run.log` line 94 + wrap site `agents/agents.py:2993` (`"error": f"LLM call failed: {e}"`, stringifies → original stack lost) |
| P4 | OAuth path does not hit this | primary-source | CONFIRMED — `test_parallel_tracing_canonical.py` (oauth) runs to `Success: True`; only the `anthropic` (non-oauth) provider triggers it |
| P5 | The offending `message_id=` kwarg is NOT in `adapters/anthropic.py` | primary-source | CONFIRMED@grep (no `message_id` in `adapters/anthropic.py`); it is injected in the agent-side response/tool-call handling, not the adapter |
| P6 | A `MessageError`/`CoordinationError`/`StateError`-family exception (subclasses forwarding `**kwargs` to `AgentFrameworkError.__init__`, e.g. `exceptions.py:1212,1374`) is constructed with `message_id=` somewhere in the standard-`anthropic` tool_use response handling | my-recall (hypothesis) | UNVERIFIED — investigator to confirm the exact raise site from a real stack |

## 3. Frame check

Most load-bearing premise: P3 — "a successful response is crashing the
post-response handling via an error constructed with an unsupported
`message_id` kwarg." Falsifier to look for: the model response is actually
*malformed* (so an error is legitimately raised and the real bug is only
that the error constructor has a kwarg typo) — vs the response is valid and
the handling path itself is wrong. P1 (valid `tool_use` content logged)
already weighs against "malformed response." The investigator must still
get the **real traceback** (the wrap at `agents.py:2993` discards it) before
committing to a root cause — pin the exact raise site, do not infer it.

## 4. Reproduction

`python packages/framework/benchmarks/GAIA/test_parallel_tracing_canonical_anthropic.py`
with `ANTHROPIC_API_KEY` set (loaded from repo `.env`). Deterministic:
fails every run at step 1 (~2.7s, real round-trip) with the P3 TypeError;
the test asserts `result.success is True` so it exits non-zero. Pre-existing
secondary papercut also found: the standard `anthropic` adapter has no
`MODEL_ALIASES` table (the oauth one does), so the friendly alias
`claude-haiku-4.5` 404s — the test uses the canonical dated ID
`claude-haiku-4-5-20251001`. That asymmetry is noted, NOT in scope here.

## 5. Root-cause analysis

**Root A (the message_id TypeError — FIXED & VERIFIED).** `adapters/anthropic.py:299`
emitted tool-call `arguments` as a Python dict (`block.get("input", {})`).
The canonical contract at the `ToolCallMsg` boundary is a JSON **string**
(`memory.py:179` typed `str`, enforced `:191-192`, consumed via
`json.loads` at `coordination_tools.py:55` / `design-principles.md:68`).
Every other producer (`anthropic_oauth`, `google` `json.dumps`, `openai`
native, `openrouter`, local `models/utils.py`) emits a string; standard
`anthropic` was the only violator. The raised `ValueError` was then *masked*
as the surfaced `TypeError` by **Root B**: `memory.py`'s `MessageError(...)`
calls (~14 sites) pass `message_id=`/`invalid_data=`/`expected_format=`
kwargs that `AgentFrameworkError.__init__` (`exceptions.py:52-61`) rejects.
Both an investigator and an independent commit-before-exposure reviewer
converged: **verdict SOUND**, Root A is the irreducible root for the goal,
the adapter is the architecturally correct seam (DP-006), Root B is a real
but separate latent architect-level bug — explicitly NOT bundled.

**Root B (latent, OUT of scope — surfaced).** The `memory.py`
`MessageError`/subclass constructions vs the `AgentFrameworkError` kwarg
API. Dormant after Root A on the success path (the `ValueError` guard never
fires), but any genuinely-malformed tool/agent call on any provider still
hits the `TypeError` mask. Recommended remedy: widen the exception API
(`message_id`/`invalid_data`/`expected_format` → fold into `context`),
mirroring `MessageFormatError`/`ToolCallError`. Architect-level, separate PR.

**Root C (newly exposed, distinct — NOT yet fixed).** With Root A fixed the
workflow runs 1→15 steps; agents harmonize many `tool_use` turns correctly.
It then fails when the standard-`anthropic` adapter builds the **next
multi-turn request** after an assistant turn carrying multiple tool calls
(2× `plan_update` + `invoke_agent`): Anthropic returns **HTTP 400 Bad
Request** (`_run.log` 15:21:17; the adapter only captured `error='Bad
Request'`, not the body). A request-construction defect in the standard
adapter's tool-history → Anthropic-messages conversion (OAuth-masked).
Distinct from Root A/B; needs its own investigation (real 400 body required
— deeper instrumentation than the log gives).

## 6. Proposed solution & status

- **Applied (Root A):** `adapters/anthropic.py:299` →
  `json.dumps(block.get("input", {}))` (+ WHY comment). Regression test
  `tests/models/test_adapter_harmonize.py` (2 cases). No `ToolCallMsg`
  loosening, no `try/except`, Root B not bundled.
- **Verified:** AC-1/AC-2 (unit, JSON-string + faithful round-trip) PASS;
  touched-module + adapter suites 90 passed (AC-6); the `message_id`
  TypeError is gone and the run progresses 1→15 steps.
- **AC-3 NOT met:** the Anthropic-API e2e does **not** reach
  `Success: True` — blocked by **Root C** (a different, newly-exposed bug),
  not by a Root-A regression. AC-4 (oauth no-regression) still to confirm.
- **Open decision (user):** how deep to pursue the standard-`anthropic`
  onion (Root C, then possibly more) vs. land Root A (a real, verified,
  independently-correct improvement: 1→15 steps, contract-conformant,
  tested) and track Root B + Root C as separate follow-ups.

**Root C (FIXED & VERIFIED).** Real Anthropic 400 body (captured via
temporary instrumentation, reverted): `messages.N.name: Extra inputs are
not permitted`. The standard adapter's `format_request_payload`
plain-message `else` branch (`anthropic.py:172-180`) did
`cleaned_msg = msg.copy()`, leaking the OpenAI-family message-level `name`
(set legitimately by `memory.to_llm_dict` for agent identity) into the
Anthropic payload; Anthropic permits only `role`/`content`. The investigator
**refuted** the original multi-tool-call hypothesis from the real body
(grounding working as intended). Fix: rebuild the `else` branch to emit only
`{role, content}`, mirroring the proven oauth twin
(`anthropic_oauth.py` regular-message branch). Self-reviewed against primary
source + the oauth twin; localized single-branch single-file fix verified by
the API's own error message — independent review not warranted (contrast
Root A, which touched a TRUNK-STABLE contract).

**Final status: DONE.** The standard-`anthropic` chain is fully fixed.
End-to-end `test_parallel_tracing_canonical_anthropic.py` → `Success: True`
(12 steps, coherent answer, exit 0). OAuth no-regression
`test_parallel_tracing_canonical.py` → `Success: True` (15 steps).
Touched-module + adapter regression 92 passed. Acceptance AC-1..AC-7 met.
**Root B remains** a surfaced, latent, architect-level separate item (the
`memory.py` `MessageError`/`AgentFrameworkError` kwarg mismatch + the async
error path swallowing 400 bodies — both recommended as their own PRs, NOT
bundled here).

The chain oauth masked, now fully traversed: Session 07 (ModelConfig
materialization) → Root A (tool-arg dict→JSON-string) → Root C (message
`name` leak). Root B latent/dormant.

## 7. Acceptance

Frozen at `08-anthropic-response-message-id-typeerror/acceptance.md` before
the fix is applied.
