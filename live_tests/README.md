# live_tests/

Real-world end-to-end test scripts that exercise the marsys framework against live dependencies — actual LLMs, OAuth, network, disk. Not run by pytest; run manually or by an automation harness (an agent like Claude Code, a periodic CI job, or a contributor verifying a change).

## Why this exists, distinct from `tests/`

`packages/framework/tests/` holds offline unit + integration tests run by pytest in CI. They mock LLMs, mock disks, mock everything external. They prove the code is correct in isolation.

`live_tests/` proves the framework behaves correctly under real conditions: an actual model dispatches, a real branch fans out, an actual NDJSON file lands on disk with the expected shape. Mocks can lie; live runs can't.

The two layers complement each other. Unit tests are fast, hermetic, and run on every commit. Live tests are slow, expensive, and run when you want to be sure the integration holds.

## Conventions every script must follow

These exist so an agent can run any script in this directory without per-script ramp-up.

1. **Self-documenting docstring at the top of every script.** Sections in this order:
   - `PURPOSE` — one or two sentences on what the script verifies.
   - `EXERCISES` — bullet list of framework modules / classes covered.
   - `TOPOLOGY` — ascii sketch if the test runs an Orchestra topology.
   - `RUN` — the exact CLI invocation.
   - `KEY ARGS` — the args that change behaviour, especially `--task`-style knobs that should stay stable across runs to keep traces comparable.
   - `OUTPUTS` — what files land in `<output-dir>` and what the stdout summary looks like.
   - `REQUIREMENTS` — OAuth profile, env vars, network access, etc.

2. **`--output-dir DIR` flag, every artifact under that dir.** Logs, traces, metrics — all of it. Predictable paths so downstream automation knows where to look without parsing stdout.

3. **Single-line JSON summary as the last stdout line.** Always parseable. Always includes `output_dir`, `log_file`, `trace_file` (where relevant), `all_checks_passed: bool`, and a `checks: {name: bool}` dict. Anything else is bonus.

4. **Exit code 0 / 1.** 0 = all checks passed, 1 = at least one failed. The exit code is authoritative; the JSON summary is the diagnostic detail.

5. **No `test_` prefix on filenames.** Pytest skips files without that prefix; we want that. These run via `python <path>`, never via `pytest`.

6. **No reliance on cwd magic.** Use `Path(__file__).resolve().parent` for any relative-path needs. Scripts must run from any working dir as long as the venv is active.

## Layout

```
live_tests/
├── README.md       this file
└── <area>/         one subdir per framework area being exercised
    └── <name>.py   one script per scenario
```

Current areas:

- **tracing/** — the execution-trace pipeline (collector → sink → file/network). Includes `parallel.py` (parallel-fan-out + convergence + redaction).

Planned future areas (not yet implemented; placeholders for direction):

- `orchestration/` — Orchestra.run lifecycle, retry semantics, barrier behaviour
- `agents/` — Agent self-invocation, tool routing, memory retention modes
- `topology/` — declarative topology validation, det-node lifecycle, gating

When adding a new area, create the subdir, drop a script there following the conventions above, and add a one-line entry under "Current areas" here.

## How to run a script (human or agent)

```bash
cd packages/framework
source ../../.venv/bin/activate
python live_tests/<area>/<name>.py --output-dir /tmp/marsys_runs/<id>
```

Each script's `--help` lists its full argument set. The first lines of every script's docstring also work as a quick reference — read them before changing args.

## How an agent (Claude Code) decides whether to investigate

Decision flow after a run completes:

- Exit 0 + `all_checks_passed: true` → done; report success.
- Exit 1 → read the run log at `<output-dir>/run.log` and the trace file at the summary's `trace_file` path. The `checks` dict tells you which check failed and where to look. Common failure modes per script live in that script's own docstring.
- Exit non-0 with no JSON summary on stdout → the script crashed before it could verify. Read the bottom of stdout/stderr for the traceback.

## Adding new scripts

Use `live_tests/tracing/parallel.py` as the template:

1. Copy it to the right area subdir under a new name.
2. Replace the docstring with the new scenario's purpose / args / outputs.
3. Replace the topology and agents.
4. Adjust `verify_run()` to assert the scenario-specific invariants.
5. Keep the JSON summary contract — same keys, same exit behaviour.
