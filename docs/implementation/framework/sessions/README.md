# Framework Sessions — How They Work

Each session is a single framework PR. Self-contained brief; implementer picks it up; it ships.

## Why sessions

- **Scope discipline** — each session = one coherent feature in the framework
- **Multi-consumer justifiable** — every feature must serve more than just Spren (LangSmith / Phoenix / Cloud / Studio / third-party — at minimum, the architecture must allow other consumers to use the same surface)
- **Framework purity** — no Spren-specific code paths in the framework (SP-018 from Spren's perspective; same rule from the framework's perspective: don't special-case any one consumer)
- **Reviewability** — one session = one PR a framework reviewer can fully comprehend

## Contract every session must satisfy

1. **Self-contained brief.** A framework implementer with no Spren context can read the brief + linked sources and execute the work.
2. **Lands as a framework PR.** Branch on the framework, PR to framework's main, review by framework folks.
3. **Working artifact at the end.** No half-finished features.
4. **Framework test discipline:** the framework's own test conventions apply (pytest, framework regression suite stays green, new feature has its own tests).
5. **Multi-consumer justification.** Brief states explicitly who else can use this beyond Spren. If only Spren can use it, the design is wrong — surface to architectural review.
6. **No TRUNK-CRITICAL changes** without an ADR and explicit framework-team approval. The TRUNK-CRITICAL list is in the framework's `CLAUDE.md`: `Orchestra`, `Orchestrator`, `RealRuntime`, `ValidationProcessor`, `TopologyGraph`. New code in `coordination/tracing/`, `coordination/`, etc. is fine.

## File layout

```
sessions/
├── README.md                          # this file
├── _template.md                       # canonical session template
├── 01-ndjson-streaming-tracing-writer.md
├── 02-telemetry-sink-protocol.md
├── 03-pause-resume-completion.md
├── 04-workflow-serializer.md
└── 05-semantic-linter.md
```

Sessions are numbered in execution order. Each session is independent within the framework (they can land in any order); the number expresses the order in which Spren needs them.

## How to write a new session

1. Copy `_template.md` to `NN-<title>.md`
2. Fill out every section
3. Verify acceptance criteria are testable in the framework's CI
4. Verify "Multi-consumer justification" lists at least one consumer beyond Spren
5. Verify "Open questions for the framework team" is empty (or you've genuinely escalated)

## How to execute a session

A framework implementer (could be the user, could be a delegated agent):

1. Reads the session file
2. Reads every linked authoritative source (framework architecture docs, framework `CLAUDE.md`, the affected modules' code)
3. Verifies the plan against current framework state
4. Validates "best practice" claims with current online research
5. Asks the framework team any escalation questions BEFORE writing code
6. Implements within scope
7. Writes framework-style tests (unit + integration; framework regression suite stays green)
8. Updates the session file with "What was actually built" + "Lessons / Surprises"
9. Submits the PR against the framework's main branch
10. Reports back to Spren planning when the PR merges + ships, so Spren-side dependencies update

## Implementer evaluation gate

At session completion:

- [ ] Acceptance criteria checked (with evidence in test logs / PR description)
- [ ] Framework regression suite green
- [ ] New tests written and passing
- [ ] Multi-consumer surface verified (no Spren-only paths)
- [ ] No TRUNK-CRITICAL changes outside what the brief explicitly authorizes
- [ ] Documentation updated (framework's own docs, where applicable)
- [ ] Session file updated with what was actually built

If any of these fails, the session is not done.
