# Session 09 — Frozen acceptance criteria (B1; B2 separate)

Frozen 2026-05-16 after reproduce + investigate + independent
commit-before-exposure review. Scope here is **B1** (the exception-kwarg
masker). **B2** (async API 4xx body swallowed) is separable — the
independent review found the investigator's B2 fix *mechanism* wrong
("pass a parsed dict as `response`" is discarded by
`from_provider_response`'s `hasattr(response,'json')` gate); B2 is sent
back for re-spec and is NOT in this acceptance.

- **AC-1** Constructing the base `MessageError` with any of
  `message_id` / `invalid_data` / `expected_format` / `validation_path`
  returns a usable `MessageError` (no `TypeError`); the values are present
  in `error.context` (size-bounded, like the sibling subclasses).
- **AC-2** The realistic path —
  `Message(role="assistant", content="x", tool_calls=["bad"])` and the
  other `memory.py` validation guards — raises a usable `MessageError`
  (was: `TypeError: AgentFrameworkError.__init__() got an unexpected
  keyword argument 'message_id'`). The ~14 `memory.py` raise sites are
  UNCHANGED (fix is in the base class only).
- **AC-3** No sibling regression: `MessageFormatError` /
  `MessageContentError` / `SchemaValidationError` etc. still expose their
  own diagnostic attributes (e.g. `MessageFormatError(...).expected_format`
  is the supplied value, NOT clobbered to `None`). The base class
  deliberately does NOT set these as attributes (would clobber siblings
  that set them before `super().__init__`).
- **AC-4** `AgentFrameworkError.to_dict()` / `.context` shape is not
  broken — diagnostics are added as `context` keys (enrichment), no key
  removed, no structure changed. `AgentFrameworkError`, `memory.py`,
  the adapters are NOT modified for B1.
- **AC-5** The `pytest.raises((MessageError, TypeError))` tuples that
  encoded the bug (7 sites in `tests/memory/test_memory.py` +
  `tests/agents/test_memory.py`) are tightened to `pytest.raises(MessageError)`
  so the regression cannot silently return; the `(MessageError, KeyError,
  TypeError)` site at `agents/test_memory.py` drops only the bug arm
  (`TypeError`), keeping the legitimate `KeyError`. No skip/xfail.
- **AC-6** Touched-module regression green: `tests/agents/test_exceptions.py`,
  `tests/memory/`, `tests/agents/test_memory.py` — modulo the
  **pre-existing, unrelated** `test_memory_manual.py::test_message_with_images`
  failure (`FileNotFoundError` on a missing `tmp/screenshots/` dir; fails
  identically on clean HEAD with this change stashed; NOT caused by Root B
  and deliberately NOT fixed here — surfaced separately to avoid smuggled
  scope).
- **AC-7** No standard-anthropic e2e regression: B1 is on the exception
  path that runs in every workflow; `test_parallel_tracing_canonical_anthropic.py`
  still reaches `Success: True`.
