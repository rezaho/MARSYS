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
5. **Implementer is responsible.** The implementer is a peer, not an order-taker. They must catch mistakes in the plan, verify against current repo state, do their own online research to validate "best practices" claims, and ask the user when they hit strategic or preference questions. See "Implementer responsibilities" in [`_session-template.md`](./_session-template.md).

## File layout

Sessions are grouped into **bundles** — coherent feature slices that together produce a demo-able outcome. A bundle holds one or more sessions, plus bundle-level testing artifacts and (optional) shared assets.

```
docs/implementation/spren/
├── README.md
├── SESSIONS.md                                 # this file
├── _session-template.md                        # the canonical session template — copy + fill
├── v0.3-mvp.md                                 # release-level summary
├── v0.4-extensions.md
├── v0.5-future.md
└── <version>/                                  # e.g. v0.3.0/ — release version dir
    └── <NN>-<bundle-slug>/                     # e.g. 01-visual-builder/ — a bundle = demo-able slice
        ├── sessions/
        │   ├── <NN>-<slug>.md                  # individual session brief
        │   └── <NN>-<slug>/                    # per-session artifact dir
        │       └── acceptance.md               # frozen acceptance criteria (checked in)
        ├── testing/
        │   ├── test-scenarios.md               # user-facing: features + scenarios to test
        │   └── test-session.md                 # agent-facing: testing-session brief
        └── assets/                             # (optional) bundle-level shared media
```

Concretely today:

```
v0.3.0/
├── 01-visual-builder/                          # Bundle: Sessions 01+02+03 produce the visual workflow builder demo
│   ├── sessions/
│   │   ├── 01-foundation.md
│   │   ├── 02-workflow-crud-types.md
│   │   ├── 02-workflow-crud-types/
│   │   │   └── acceptance.md
│   │   ├── 03-visual-builder.md
│   │   └── 03-visual-builder/
│   │       └── acceptance.md
│   ├── testing/
│   │   └── test-scenarios.md
│   └── assets/
│       ├── spren-inspiration.png
│       ├── palette-preview.html
│       └── ...
└── 02-run-execution-and-inspection/            # Bundle: Sessions 04+05 produce the runnable-workflow demo
    ├── sessions/
    │   └── 04-run-execution.md
    └── testing/
```

Sessions are numbered globally within a release (Sessions 01 through 10 across all bundles in v0.3.0); session N+1 may depend on session N being shipped. Frozen acceptance criteria for each session live in a sibling artifact directory inside the bundle's `sessions/` (`<NN>-<slug>/acceptance.md`), which the test-coverage auditor reads as its only context. Bundle-level `testing/` carries cross-session integration test scenarios — the joins between sessions that per-session manual-verify checklists don't cover.

## How to write a new session

1. Identify the Spren release the session targets (e.g., `v0.3.0`); create `<version>/` if it doesn't exist
2. Identify the **bundle** the session belongs to. A bundle is a coherent demo-able slice. If unsure, ask the user. If first session of a new bundle, create `<version>/<NN>-<bundle-slug>/` with `sessions/`, `testing/`, and (if needed) `assets/` subdirs.
3. Copy `_session-template.md` to `<version>/<NN>-<bundle>/sessions/<NN>-<slug>.md`
4. Fill out every section
5. Verify acceptance criteria are testable
6. Verify "Open questions for the user" is empty (or you've genuinely escalated)
7. Have one of: peer review, planner agent review — before any agent runs the session

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
9. Updates `docs/implementation/spren/v<N>-*.md` (the release-level summary) checkboxes to reflect what shipped, plus the bundle's `testing/test-scenarios.md` if new scenarios surface

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
