# Session 08 — Frozen acceptance criteria

Frozen 2026-05-16, after reproduce + RCA + independent review (both agents
converged SOUND). The test-coverage-auditor sees only this file and the
test files — never the implementation.

Scope: the standard-`anthropic` adapter emitting tool-call `arguments` as a
dict instead of the canonical JSON string (Root A). Root B (the
`memory.py` `MessageError` kwarg/`AgentFrameworkError` mismatch that *masked*
Root A as a `TypeError`) is a separate latent architect-level bug,
explicitly OUT of scope here and surfaced to the user.

- **AC-1** `AnthropicAdapter.harmonize_response`, given a raw response with a
  `tool_use` block whose `input` is an object, returns a `HarmonizedResponse`
  whose `tool_calls[0].function["arguments"]` is a `str` (not a dict), and
  `json.loads` of it equals the original `input` object (faithful, not
  `repr`/`str`).
- **AC-2** Same, with empty `input` (`{}`): `arguments` is the string `"{}"`,
  never a dict or `None`.
- **AC-3** End-to-end: `python packages/framework/benchmarks/GAIA/test_parallel_tracing_canonical_anthropic.py`
  (standard `anthropic` provider, `ANTHROPIC_API_KEY` from repo `.env`)
  reaches `result.success is True` with a non-empty final response and exits
  0 (the script asserts the outcome). The pre-fix `TypeError`/`message_id`
  crash does not occur.
- **AC-4** No regression to the OAuth path:
  `test_parallel_tracing_canonical.py` (`anthropic-oauth`) still reaches
  `Success: True`.
- **AC-5** The fix is confined to the adapter seam: only
  `packages/framework/src/marsys/models/adapters/anthropic.py` changes in
  production code. `ToolCallMsg`'s string check (`memory.py`) is NOT
  loosened; no `try`/`except` is added; no `_v2`/variant file; Root B is
  not bundled.
  - _[amended 2026-05-16 — user chose "fix the whole chain"] scope is TWO
    localized changes in that one file, not one line: (a) Root A —
    `harmonize_response` `json.dumps(block.get("input", {}))`; (b) Root C —
    the `format_request_payload` plain-message `else` branch rebuilt to emit
    only `{role, content}` (was `msg.copy()`, leaked OpenAI `name` →
    Anthropic 400). Both mirror the proven oauth twin. Root C was
    discovered mid-flight (exposed by the Root A fix), root-caused from the
    real Anthropic 400 body, and surfaced to the user before continuing._
- **AC-7** _[added 2026-05-16]_ `format_request_payload`, given a plain
  assistant/user message carrying a message-level `name` (as
  `memory.to_llm_dict` produces), emits Anthropic messages whose keys are a
  subset of `{role, content}` (no `name`/OpenAI-only keys); the
  tool / assistant-with-tool_calls branches still emit wire-legal shapes.
- **AC-6** Touched-module regression stays green (adapter/model + agent
  serialize suites that exercise harmonized tool calls).
