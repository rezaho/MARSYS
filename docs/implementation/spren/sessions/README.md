# Sessions — How Implementation Works

Implementation of Spren is sliced into **sessions**. A session is a self-contained brief that an implementer agent (in a fresh session, with no memory of any prior conversation) can execute end-to-end and produce a working, testable artifact.

## Why sessions

- **Scope discipline** — each session ships a small, complete thing rather than a half-finished feature
- **Context efficiency** — implementer agents don't need to know the whole project; they need to know *this* session
- **Reviewability** — one session = one PR that a reviewer can fully comprehend
- **Resilience** — if a session goes wrong, revert it; the rest of the system stands

## Contract every session must satisfy

1. **Self-contained brief.** An implementer with no prior context can read the session file + linked authoritative sources and execute the work without back-and-forth ambiguity.
2. **Working artifact at the end.** Sessions never end mid-feature. If a feature is too big to fit one session, split it into multiple sessions where each one ships something coherent.
3. **All three test types** (where applicable): unit, integration, end-to-end. No deferring tests. See spren design principle SP-007.
4. **Hard rules enforced.** No mocks of in-codebase features (SP-007). No backward-compatibility code paths (SP-006). No TRUNK-CRITICAL framework changes without explicit approval (SP-001 + `/CLAUDE.md` § 4).
5. **Implementer is responsible.** The implementer is a peer, not an order-taker. They must catch mistakes in the plan, verify against current repo state, do their own online research to validate "best practices" claims, and ask the user when they hit strategic or preference questions. See "Implementer responsibilities" in [`_template.md`](./_template.md).

## File layout

```
sessions/
├── README.md          # this file
├── _template.md       # the canonical session template — copy + fill
└── v<release>/        # one subdir per Spren release the sessions ship in
    ├── NN-<slug>.md           # individual session brief
    └── NN-<slug>/             # sibling artifact dir for that session (acceptance.md, etc.)
        └── acceptance.md
```

Concretely today:

```
sessions/
├── README.md
├── _template.md
└── v0.3.0/
    ├── 01-foundation.md
    ├── 02-workflow-crud-types.md
    └── 02-workflow-crud-types/
        └── acceptance.md
```

Sessions live under the Spren release version they target. Sessions are numbered in execution order within a release; session N+1 may depend on session N being shipped. Frozen acceptance criteria for each session live in a sibling artifact directory (`NN-<slug>/acceptance.md`), which the test-coverage auditor reads as its only context.

## How to write a new session

1. Identify the Spren release the session targets (e.g., `v0.3.0`); create `sessions/<version>/` if it doesn't exist
2. Copy `_template.md` to `sessions/<version>/NN-<slug>.md`
3. Fill out every section
4. Verify acceptance criteria are testable
5. Verify "Open questions for the user" is empty (or you've genuinely escalated)
6. Have one of: peer review, planner agent review — before any agent runs the session

## How to execute a session

Hand the session file path to an implementer agent. The agent:

1. Reads the session file from top to bottom
2. Reads every linked authoritative source
3. Verifies the plan against current repo state (`git log`, `grep`, read files); if anything has drifted, surfaces it before writing code
4. Validates every "best practice" claim with current online research; pushes back if outdated
5. Asks the user any escalation questions BEFORE writing code
6. Implements the smallest version that satisfies acceptance criteria, then iterates if there's slack
7. Writes all required tests (unit + integration + E2E where applicable)
8. Updates the session file with a "What was actually built" delta and "Lessons / Surprises" section
9. Updates `docs/implementation/spren/v<N>-*.md` checkboxes to reflect what shipped

## Implementer evaluation gate

At session completion, the implementer's work must satisfy:

- [ ] All acceptance criteria checked (with evidence in test logs)
- [ ] All required test types written and passing
- [ ] No mocks of in-codebase features
- [ ] No backward-compatibility code
- [ ] No TRUNK-CRITICAL framework changes (unless explicitly authorized in the session)
- [ ] No additions of features beyond session scope
- [ ] Session file updated with what was actually built

If any of these fails, the session is not done. Reopen the brief and continue.
