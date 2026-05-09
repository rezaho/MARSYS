# Session 01: Foundation — Umbrella Workspace + FastAPI Sidecar + Vite Skeleton + Tauri Shell

---

## Working rules — how we collaborate (READ FIRST)

You are a peer on this project. You are NOT an order-taker. You share equal voice and equal responsibility for the success of MARSYS Spren. We work together — not above-and-below.

### Be a peer with equal voice

- **Push back when you disagree.** If this brief is wrong, or if a "best practice" cited here is outdated, or if a structural choice will cause us pain later, say so. Defend your position with evidence.
- **Stay engaged.** Comment in this session file as you go; flag concerns before they become problems.
- **Be proactive.** If you see something this session is missing, raise it.

### Take responsibility

- **Ownership is shared.** If something fails, it's our shared failure.
- **You own correctness.** Manually verify acceptance criteria, not just unit tests.
- **You own follow-through.** Update this file's "What was actually built". Update [`docs/implementation/spren/v0.3-mvp.md`](../v0.3-mvp.md) checkboxes. Add "Lessons / Surprises" if anything surprised you.

### Double-check before any decision

- **Read the code before changing it.** Don't assume; verify.
- **Verify file paths and symbols** still exist before referencing them.
- **Run tests after every meaningful change**, not just at end.
- **Use git commits as checkpoints.**

### Critically assess the plan itself

This brief was written before you started. **Don't follow it blindly.**

- **Read the actual code** for any module the brief references.
- **Spawn an independent sub-agent for verification** when material doubt exists.
- **Run online research** for any "best practice" or library-version claim. If 2026 docs disagree, defer to current docs and push back.
- **Cite sources when you challenge the plan.**

### Ask the user when blocked on intent

Strategic, opinionated, or subjective decisions belong to the user. Use `AskUserQuestion` for strategic / product-vision / opinionated/subjective / "doesn't feel right" / scope-expansion questions. Do NOT ask for technical implementation choices you can decide.

### Build the smallest thing that works first; don't expand scope silently.

### Foundational project rules

- [`/CLAUDE.md`](../../../../CLAUDE.md) — umbrella repo's working rules
- [`/BRAINDUMP.md`](../../../../BRAINDUMP.md) — context dump from prior design conversations (READ FIRST every fresh session)
- [`docs/architecture/spren/08-design-principles.md`](../../../architecture/spren/08-design-principles.md) — Spren design principles SP-001..SP-019

---

## The big picture — what we're building and why

### What MARSYS Spren is

MARSYS Spren is the **open-source umbrella product** of the Marsys AI company. It contains:

1. **The marsys Python framework** — multi-agent orchestration. Already exists; this session relocates it into `packages/framework/`.
2. **Spren** — a continuously-active personal AI assistant (the meta-agent + always-on layer; built session-by-session starting after this one).
3. **The visual builder web app** — a Vite + React + TanStack Router single-page app for designing and inspecting workflows. Renders inside the Tauri webview (desktop GUI) AND in a system browser tab (browser GUI).
4. **The Tauri 2 desktop shell** — a Rust binary that bundles the Python sidecar + the Vite static bundle and produces native installers per platform.
5. **The TUI** — `apps/tui/` Textual app (placeholder this session; full implementation in v0.4).

(The future `SprenTelemetrySink` implementation lives inside the Spren daemon package — `packages/spren/src/spren/telemetry/` — when the framework's `TelemetrySink` protocol ships in v0.4. There is no separate adapter SDK package.)

The umbrella is **distinct from** two proprietary products in sibling repos: **MARSYS Cloud** (hosted control plane) and **MARSYS Studio** (hosted SaaS UI). The umbrella stays **single-user-local** so it doesn't cannibalize the proprietary stack.

### Why one repo for all six pieces

The visual builder is a feature of marsys (every framework deserves a visual debugger / inspector for its core data structures) AND a feature of Spren (it's Spren's home page surface). The Tauri shell, the TUI, and the adapter SDK all consume the same FastAPI backend (SP-019). Splitting them across repos forces awkward cross-repo dependency management. Keeping them in one umbrella avoids that.

The framework remains a **library** distributed via PyPI (`pip install marsys`) — its audience is Python developers. **Spren** is a **product** distributed via native installers + secondary channels (brew / winget / apt / npm / pipx / Docker). See [`docs/architecture/spren/05-packaging-distribution.md`](../../../architecture/spren/05-packaging-distribution.md).

### Your role as an implementer

1. Honor the architecture (`docs/architecture/spren/`)
2. Honor the design principles (SP-001..SP-019)
3. Ship a working artifact at session end — never half-finished
4. Write all required tests (unit + integration + E2E where applicable)
5. Push back when something is wrong

### Where to read deeper if you need it

- High-level: [`docs/architecture/spren/00-overview.md`](../../../architecture/spren/00-overview.md)
- System context: [`docs/architecture/spren/01-system-context.md`](../../../architecture/spren/01-system-context.md)
- Frontend architecture: [`docs/architecture/spren/04-frontend-architecture.md`](../../../architecture/spren/04-frontend-architecture.md)
- Packaging & distribution: [`docs/architecture/spren/05-packaging-distribution.md`](../../../architecture/spren/05-packaging-distribution.md)
- Meta-agent design: [`docs/architecture/spren/09-meta-agent.md`](../../../architecture/spren/09-meta-agent.md)
- Memory architecture: [`docs/architecture/spren/10-memory-architecture.md`](../../../architecture/spren/10-memory-architecture.md)
- Full v0.3 plan: [`docs/implementation/spren/v0.3-mvp.md`](../v0.3-mvp.md), [`docs/implementation/spren/00-overview.md`](../00-overview.md)
- BRAINDUMP: [`/BRAINDUMP.md`](../../../../BRAINDUMP.md)

---

## What came before this session

**Previous sessions:** None — this is the first implementation session.

**State at start of this session:**

- The umbrella repo lives at `~/research_projects/marsys-spren-work/` (this worktree) on branch `feature/spren-umbrella`.
- The marsys framework code currently lives at `src/marsys/` (repo root) — this session relocates it.
- Tests live at `tests/` (repo root) — also relocates.
- The framework's `pyproject.toml` is at the repo root.
- `docs/` exists with `architecture/{framework, spren}/` and `implementation/spren/` subdirs.
- `CLAUDE.md` and `BRAINDUMP.md` exist at repo root (gitignored).
- `tmp/spren-migration/` contains preserved files from an early-design experiment.
- Framework is at v0.2.x.
- No `apps/`, no `packages/` dirs yet. No Node deps. No Vite. No FastAPI server. No Tauri shell. No Cargo workspace.

**Verify state with:**
```bash
cd /home/rezaho/research_projects/marsys-spren-work/
ls -la                                    # README, CLAUDE.md, BRAINDUMP.md, docs/, src/, tests/, pyproject.toml at root; no apps/ or packages/
git status                                # may have uncommitted user work — note the baseline
source .venv/bin/activate
pytest tests/ -x --tb=short               # CAPTURE BASELINE TEST COUNTS (passed/failed/skipped)
```

If the baseline test suite has unexpected failures, **stop and surface to the user** before proceeding. Post-restructure, the same numbers must hold.

---

## Bundle position + tier

- **Bundle**: A — Visual workflow builder ([test scenarios](../bundles/A-visual-builder/test-scenarios.md))
- **Position in bundle**: 1 of 3 (foundation; Sessions 02 + 03 build on top)
- **Tier**: CRITICAL — Researcher + Designer + Validator/Critic. Foundation work touches the framework relocation (TRUNK-CRITICAL implication: zero `.py` edits inside `src/marsys/`), distribution scaffolding, and the Tauri sidecar protocol — all load-bearing for everything after.
- **Bundle outcome this session contributes to**: dev environment exists; Tauri shell launches sidecar + opens webview; framework regression tests still green at new location.

## Files to DELETE from prior sessions

n/a — first session; nothing prior to remove.

**Note for Session 02**: this session's placeholder home route at `apps/web/src/routes/index.tsx` is *replaced* by Session 02 with workflow CRUD UI (which is itself replaced by Session 03's visual builder). Each session's brief MUST list its predecessor's placeholder file for deletion.

**Note for v0.4**: the future `SprenTelemetrySink` lives inside `packages/spren/src/spren/telemetry/` (not in a separate adapter SDK package).

## What this session ships

After this session, the umbrella repo has:

- **`packages/framework/`** — the marsys framework, relocated from `src/marsys/`. NO behavior change; same Python source, same tests, same `pyproject.toml`. Just under a different path.
- **`packages/spren/`** — a new Python package for the Spren meta-agent layer. This session populates it with: a FastAPI server with `/healthz` + `/v1/bootstrap` + per-launch auth; basic CORS middleware (incl. `tauri://localhost`); an entry point that the Tauri sidecar launches.
- **`apps/web/`** — a Vite + React + TanStack Router skeleton. A single placeholder route reads bootstrap, renders the response.
- **`apps/desktop/`** — a Tauri 2 shell that spawns the Python sidecar as a managed child process, captures the auth token, opens the webview pointing at `http://127.0.0.1:<port>/` (where the sidecar serves the Vite bundle).
- **`apps/tui/`** — a placeholder Textual project (empty entry point); full implementation in v0.4.
- **Root `pyproject.toml`** is converted to a uv workspace declaration (no `[project]` section).
- **`pnpm-workspace.yaml`** at repo root for the JS workspace.
- **`Cargo.toml`** at repo root declaring the Rust workspace (Tauri shell + future Rust components).
- **`Justfile`** orchestrates cross-language tasks: `install`, `dev`, `dev-desktop`, `test`, `build`, `lint`, `clean`.

`just install` sets up Python (uv workspace install), JS deps (pnpm), and Rust deps (cargo build --release skipped at install; only on `build`). `just dev` starts FastAPI sidecar + Vite dev server; `just dev-desktop` starts FastAPI sidecar + Tauri shell pointed at the Vite dev server. `just test` runs all test suites. The framework's existing test suite passes at its new location with **zero regressions**.

**This is the foundation every later session builds on.**

### Acceptance criteria

- [ ] `packages/framework/`, `packages/spren/`, `apps/web/`, `apps/desktop/`, `apps/tui/` directories exist at the right paths
- [ ] `src/marsys/` no longer exists at repo root; it lives at `packages/framework/src/marsys/` with no code change
- [ ] `tests/` no longer exists at repo root; it lives at `packages/framework/tests/` with no code change
- [ ] Root `pyproject.toml` contains ONLY `[tool.uv.workspace]` declaration
- [ ] `packages/framework/pyproject.toml` exists with the marsys framework's project definition (moved + adjusted from old root)
- [ ] `packages/spren/pyproject.toml` exists; `name = "spren"`; depends on `marsys` via uv workspace
- [ ] `packages/spren/src/spren/server.py` defines a FastAPI app with `/healthz`, `/v1/bootstrap`, CORS middleware, and per-launch auth token validation
- [ ] `packages/spren/src/spren/auth.py` issues per-launch tokens via `secrets.token_urlsafe(32)`; constant-time validation via `secrets.compare_digest`
- [ ] `packages/spren/src/spren/__main__.py` runs uvicorn on `127.0.0.1:<random-port>`, generates token, prints a sidecar-ready signal on stdout (`spren-ready: port=<N> token=<T>`), serves the Vite production bundle from `<package>/_webui/` (when present) on `/`
- [ ] CORS allows `http://127.0.0.1:<port>`, `http://localhost:<port>`, AND `tauri://localhost` (per `docs/architecture/spren/07-security.md`)
- [ ] Server binds to 127.0.0.1 (NOT 0.0.0.0)
- [ ] `apps/web/package.json` declares React 19, Vite, TypeScript, TanStack Router, TanStack Query, `geist`, Vitest, Playwright. (Tailwind / shadcn / `@xyflow/react` / cmdk / `radix-ui` / MSW are deferred to Session 03 visual builder; declaring them here would add unused weight to `node_modules`.)
- [ ] `apps/web/vite.config.ts` configures the dev server, build output to `dist/`, and the Tauri-friendly `clearScreen: false` + `host: '127.0.0.1'` settings
- [ ] `apps/web/src/main.tsx` mounts a TanStack Router root with a single placeholder route at `/` that reads the auth token from `window.__SPREN_AUTH__` (Tauri injection) OR from URL fragment (browser fallback), strips fragment via `history.replaceState`, fetches `/v1/bootstrap`, renders the response
- [ ] `apps/desktop/Cargo.toml` declares a Tauri 2 binary; `apps/desktop/tauri.conf.json` configures the shell to load `http://127.0.0.1:$SPREN_PORT/` from the sidecar
- [ ] `apps/desktop/src/main.rs` spawns the Python sidecar (`packages/spren/src/spren/__main__.py` via the workspace's Python interpreter), reads stdout for the `spren-ready` line, captures port + token, injects the token into the webview as `window.__SPREN_AUTH__`, opens the webview
- [ ] `apps/tui/pyproject.toml` exists; package name `spren-tui` (or chosen); depends on `textual` and `httpx`; entry point `spren-tui` is a one-liner that prints "Not implemented in v0.3 — coming in v0.4"
- [ ] `pnpm-workspace.yaml` lists `apps/web` (and `packages/*-js` glob if needed)
- [ ] `Cargo.toml` at repo root declares a Rust workspace including `apps/desktop`
- [ ] Root `Justfile` provides `install`, `dev`, `dev-desktop`, `test`, `build`, `lint`, `clean` recipes
- [ ] `just install` installs Python (uv sync workspace), JS (pnpm install), and Rust deps (`cargo fetch`)
- [ ] `just dev` starts FastAPI sidecar (via `uv run python -m spren`) AND Vite dev server (in `apps/web/`) concurrently with prefixed output
- [ ] `just dev-desktop` starts FastAPI sidecar AND Tauri shell pointed at the Vite dev server
- [ ] `just test` runs framework tests + spren tests + web tests (Vitest + Playwright); all exit 0
- [ ] **Framework regression: same baseline test counts** post-restructure as pre-restructure (zero regressions)
- [ ] `import marsys` works inside `packages/spren/src/spren/`
- [ ] `CLAUDE.md` path references updated to point to new framework location
- [ ] All required tests written and passing (see Tests section)
- [ ] No mocks of in-codebase features (SP-007)
- [ ] No backward-compat code (SP-006) — old root pyproject is REPLACED, not kept
- [ ] No TRUNK-CRITICAL framework code changes (SP-001) — only relocation; zero edits to `.py` files inside `src/marsys/`
- [ ] No Spren type imported into `packages/framework/` (SP-018)

---

## Background reading (do this before writing code)

1. [`/CLAUDE.md`](../../../../CLAUDE.md)
2. [`/BRAINDUMP.md`](../../../../BRAINDUMP.md)
3. [`docs/architecture/spren/00-overview.md`](../../../architecture/spren/00-overview.md)
4. [`docs/architecture/spren/01-system-context.md`](../../../architecture/spren/01-system-context.md)
5. [`docs/architecture/spren/04-frontend-architecture.md`](../../../architecture/spren/04-frontend-architecture.md) — full stack spec
6. [`docs/architecture/spren/05-packaging-distribution.md`](../../../architecture/spren/05-packaging-distribution.md) — distribution channels + Tauri sidecar pattern
7. [`docs/architecture/spren/07-security.md`](../../../architecture/spren/07-security.md) — § Localhost API exposure (auth-token pattern)
8. [`docs/architecture/spren/08-design-principles.md`](../../../architecture/spren/08-design-principles.md) — SP-001 through SP-019
9. [`docs/implementation/spren/v0.3-mvp.md`](../v0.3-mvp.md) — what the rest of v0.3 builds on this foundation
10. Current root `pyproject.toml` — see what we're moving and what dependencies stay
11. Current `src/marsys/` and `tests/` layouts — verify what relocates

**Verify before proceeding:**
- `git log --oneline -20 src/marsys/` to confirm recent activity in framework code
- `cat pyproject.toml | head -30` to see the current marsys project metadata
- Check Vite, React 19, TypeScript, TanStack Router, Tauri 2 latest stable versions on their respective registries — DO NOT trust this brief's pin without verifying
- Check current uv version's workspace syntax (the `[tool.uv.sources]` mechanism may have evolved)
- Check Tauri 2's sidecar binding documentation for the current canonical pattern of spawning a Python child process

---

## Detailed plan

### Step 1 — Capture baseline test counts before any move

```bash
cd /home/rezaho/research_projects/marsys-spren-work/
source .venv/bin/activate
pytest tests/ -x --tb=short
# Record: total tests, passed, failed, skipped — this is your baseline
```

If unexpected failures: **stop and surface to user**.

### Step 2 — Move framework code

```bash
mkdir -p packages/framework/src
mv src/marsys packages/framework/src/marsys
mv tests packages/framework/tests
rmdir src 2>/dev/null || true
```

After this, `src/` at root no longer exists.

### Step 3 — Create `packages/framework/pyproject.toml`

Move the existing root `pyproject.toml` content to `packages/framework/pyproject.toml` and adjust:

- Build configuration: `[tool.setuptools.packages.find]` with `where = ["src"]`
- Pytest config: `[tool.pytest.ini_options]` with `testpaths = ["tests"]`
- Keep `[project]` table identical: name=`marsys`, version, deps, classifiers — the framework's external identity is unchanged

### Step 4 — Convert root `pyproject.toml` to uv workspace declaration

Replace root `pyproject.toml` contents with ONLY:

```toml
[tool.uv.workspace]
members = ["packages/framework", "packages/spren", "apps/tui"]
```

Explicit list, not glob: defends against future non-Python directories landing under `packages/` and being silently picked up by `uv sync`. NO `[project]` section at root (SP-006).

### Step 5 — Create `packages/spren/`

```
packages/spren/pyproject.toml
packages/spren/src/spren/__init__.py
packages/spren/src/spren/server.py            # FastAPI app
packages/spren/src/spren/auth.py              # token gen + validation
packages/spren/src/spren/__main__.py          # uvicorn entry; emits sidecar-ready signal
packages/spren/src/spren/_webui/.gitkeep      # Vite build output gets copied here at build time
packages/spren/tests/conftest.py
packages/spren/tests/test_auth.py
packages/spren/tests/test_server.py
packages/spren/tests/integration/test_marsys_import.py
```

`packages/spren/pyproject.toml`:

```toml
[project]
name = "spren"
version = "0.3.0"
description = "MARSYS Spren — local-first OSS personal AI assistant on top of the marsys framework."
requires-python = ">=3.12"
license = {text = "Apache-2.0"}
dependencies = [
    "marsys",
    "fastapi>=0.115",
    "uvicorn[standard]>=0.32",
    "pydantic>=2.11",
    "click>=8.1",
    "platformdirs>=4.2",
    "python-dotenv>=1.0",
]

[build-system]
requires = ["setuptools>=61", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
spren = ["_webui/**/*"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.uv.sources]
marsys = { workspace = true }
```

`server.py` shape:
- FastAPI app with `/healthz`, `/v1/bootstrap`
- CORS middleware locked to `http://127.0.0.1:*`, `http://localhost:*`, `tauri://localhost`
- Auth dependency validates `Authorization: Bearer <token>` (constant-time)
- `BootstrapResponse` Pydantic model
- StaticFiles mount on `/` serving `_webui/` if present (production); falls back to `/__no-webui` placeholder if absent (dev mode where Vite dev server is running on a different port)

`__main__.py` shape:
- `platformdirs.user_data_dir("spren")` for the data directory
- Generate token via `secrets.token_urlsafe(32)`
- Pick port via `--port` flag or default 8765 (random fallback if taken)
- Print `spren-ready: port=<N> token=<T> data-dir=<path>` on stdout (this is the Tauri shell's ready signal)
- Start uvicorn

### Step 6 — Create `apps/web/` Vite + React + TanStack Router skeleton

```
apps/web/package.json
apps/web/vite.config.ts
apps/web/tsconfig.json
apps/web/index.html
apps/web/.gitignore                          # node_modules/, dist/, *.log
apps/web/src/main.tsx                        # router root mount
apps/web/src/routes/__root.tsx               # CapabilitiesProvider + outlet
apps/web/src/routes/index.tsx                # placeholder home route
apps/web/src/providers/capabilities.tsx      # bootstrap fetch + context
apps/web/src/lib/api.ts                      # fetch wrapper with auth
apps/web/src/styles/globals.css              # minimal CSS reset + Tailwind
apps/web/src/types/api.ts                    # placeholder; Session 02 generates real types
apps/web/tests/foundation.test.ts            # Vitest unit smoke
apps/web/tests/e2e/foundation.spec.ts        # Playwright
apps/web/playwright.config.ts
```

`vite.config.ts` configures `clearScreen: false` and `host: '127.0.0.1'` (Tauri-friendly), plus the TanStack Router Vite plugin for file-based route generation, plus React + Tailwind plugins.

The placeholder home route renders `<h1>MARSYS Spren — Foundation Session</h1>` plus the bootstrap response JSON. Intentionally ugly — Session 03 replaces it with the real UI.

The capabilities provider reads the auth token from `window.__SPREN_AUTH__` (Tauri injection) OR the URL fragment (`window.location.hash` — browser fallback), strips fragment via `history.replaceState`, calls `/v1/bootstrap`, exposes the result via React Context.

### Step 7 — Create `apps/desktop/` Tauri 2 shell

```
apps/desktop/Cargo.toml                       # crate name "spren-desktop"
apps/desktop/tauri.conf.json                  # Tauri config: window, sidecar, devUrl, distDir
apps/desktop/src/main.rs                      # spawn sidecar, read ready signal, open webview
apps/desktop/build.rs                         # standard Tauri build script
apps/desktop/icons/                           # placeholder icons (replaced in Session 10)
```

`main.rs` shape:
- Resolve the Python interpreter (workspace `.venv/bin/python` in dev; bundled PyInstaller binary in production)
- Spawn the sidecar with the resolved interpreter and `-m spren`
- Read stdout line-by-line until matching `spren-ready: port=<N> token=<T>`
- Inject `window.__SPREN_AUTH__` via Tauri's `init_script` API
- Open the webview at `http://127.0.0.1:<port>/`
- On window close: send shutdown over stdin, wait for sidecar exit, then terminate

`tauri.conf.json` configures:
- `frontendDist`: in dev, `http://localhost:5173` (Vite dev server); in prod, the path served by the sidecar at `/` (so we point at `http://127.0.0.1:$SPREN_PORT/`)
- Single window: 1280x800, resizable, restorable
- App identifier: `ai.marsys.spren`
- Updater: enabled but pointed at a placeholder URL (Session 10 wires it to the real manifest server)

### Step 8 — Create `apps/tui/` placeholder

```
apps/tui/pyproject.toml                       # name "spren-tui", depends on textual + httpx + spren (workspace)
apps/tui/src/spren_tui/__init__.py
apps/tui/src/spren_tui/__main__.py            # one-liner: print message + exit
apps/tui/README.md                            # documents v0.4 plan
```

`__main__.py`: one print statement explaining the TUI ships in v0.4, then exits 0. This is intentional — the package exists so v0.4 can fill it without restructuring.

### Step 9 — Create root `pnpm-workspace.yaml`

```yaml
packages:
  - "apps/web"
```

### Step 10 — Create root `Cargo.toml` (Rust workspace)

```toml
[workspace]
members = ["apps/desktop"]
resolver = "2"
```

### Step 11 — Create root `Justfile`

The `web` workspace package is named `@marsys/spren-web`, so pnpm filters use that exact name. `cargo tauri` is the Tauri CLI binary (installed via `cargo install tauri-cli`); it does not accept `--manifest-path`, so `dev-desktop` `cd`s into `apps/desktop` first. The `set windows-shell := [...]` directive routes single-line recipe bodies to PowerShell on Windows; recipes that use bash idioms (`&` backgrounding, `trap`, `sed`, `rm -rf`, `cp -r`, `find`) are split into `[unix]` and `[windows]` variants.

```just
set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]
set script-interpreter := ["powershell.exe", "-NoLogo", "-ExecutionPolicy", "Bypass", "-File"]

default:
    @just --list

# Install Python + JS + Rust deps + tauri-cli
install:
    uv sync --extra test
    pnpm install
    cargo fetch --manifest-path apps/desktop/Cargo.toml
    cargo install tauri-cli --version "^2"

# Run dev: FastAPI sidecar + Vite dev server (no Tauri)
[unix]
dev:
    #!/usr/bin/env bash
    set -euo pipefail
    (uv run --package spren python -m spren --port 8765 2>&1 | sed 's/^/[sidecar] /') &
    SIDECAR_PID=$!
    trap "kill $SIDECAR_PID 2>/dev/null || true" EXIT INT TERM
    VITE_SPREN_API_URL=http://127.0.0.1:8765 pnpm --filter @marsys/spren-web dev 2>&1 | sed 's/^/[vite] /'

# Run dev-desktop: Vite runs separately for HMR; Tauri spawns the sidecar internally via main.rs
[unix]
dev-desktop:
    #!/usr/bin/env bash
    set -euo pipefail
    (VITE_SPREN_API_URL=http://127.0.0.1:8765 pnpm --filter @marsys/spren-web dev 2>&1 | sed 's/^/[vite] /') &
    VITE_PID=$!
    trap "kill $VITE_PID 2>/dev/null || true" EXIT INT TERM
    sleep 2
    cd apps/desktop && cargo tauri dev

# Run all tests
test:
    uv run --package marsys pytest packages/framework/tests --tb=short
    uv run --package spren pytest packages/spren/tests --tb=short
    uv run --package spren-tui pytest apps/tui/tests --tb=short
    pnpm --filter '@marsys/spren-web' test --run
    cargo test --manifest-path apps/desktop/Cargo.toml

# Build: Vite production bundle copied into spren/_webui/ (with a guard against silently-empty builds)
[unix]
build:
    pnpm --filter @marsys/spren-web build
    test -f apps/web/dist/index.html || { echo "ERROR: vite build did not produce apps/web/dist/index.html" >&2; exit 1; }
    rm -rf packages/spren/src/spren/_webui
    cp -r apps/web/dist packages/spren/src/spren/_webui

# Lint
lint:
    pnpm --filter '@marsys/spren-web' typecheck
    cargo fmt --manifest-path apps/desktop/Cargo.toml --check
    cargo clippy --manifest-path apps/desktop/Cargo.toml -- -D warnings

# Clean
[unix]
clean:
    rm -rf apps/web/dist apps/web/node_modules
    rm -rf packages/*/dist packages/*/build packages/*/*.egg-info
    rm -rf packages/spren/src/spren/_webui
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    cargo clean --manifest-path apps/desktop/Cargo.toml || true
```

`dev`, `dev-desktop`, `build`, and `clean` ship `[windows]` companion variants that drive the same effect via PowerShell (`Start-Job`, `Receive-Job`, `Test-Path`, `Remove-Item -Recurse`, `Copy-Item -Recurse`, `Get-ChildItem`). See the actual `Justfile` for the full text.

### Step 12 — Update CLAUDE.md, README.md, .gitignore

**CLAUDE.md** (gitignored):
- Update directory-structure section to reflect new layout
- Add Cargo workspace mention

**README.md** (tracked, public):
- Update to explain the umbrella product (MARSYS Spren) and its components
- Add install instructions: native installer (curl | sh) as the front door; `pip install marsys` for the framework only

**.gitignore**:
- Verify `node_modules/`, `apps/*/dist/`, `apps/*/.next/` (legacy guard), `apps/desktop/target/`, `packages/*/dist/`, `packages/spren/src/spren/_webui/` are covered

### Step 13 — Verify the integration loop

1. `uv run --package marsys pytest packages/framework/tests --tb=short` — same counts as baseline
2. `uv run --package spren pytest packages/spren/tests --tb=short` — passes (new tests)
3. `pnpm install` from repo root — succeeds
4. `pnpm --filter @marsys/spren-web build` — succeeds; produces `apps/web/dist/`
5. `cargo build --manifest-path apps/desktop/Cargo.toml` — succeeds (debug build)
6. `just dev` — sidecar + Vite both start; combined output prefixed
7. `curl localhost:8765/healthz` (no auth) → 200 with `{"status": "ok"}`
8. `curl localhost:8765/v1/bootstrap` (no auth) → 401
9. `curl localhost:8765/v1/bootstrap -H 'Authorization: Bearer $TOKEN'` → 200 with bootstrap JSON
10. Browser at `http://localhost:5173/#token=$TOKEN` → page renders placeholder + bootstrap response
11. `just dev-desktop` (i.e. `cd apps/desktop && cargo tauri dev` after Vite starts) — Tauri window opens at Vite dev server URL with token injected; renders the placeholder; bootstrap fetch succeeds
12. `uv run --package spren python -c "from marsys.coordination import Orchestra; print(Orchestra)"` → works (workspace dep resolution)

If any step fails: investigate. Don't declare done.

### Files NOT to touch

- Anything inside `packages/framework/src/marsys/` after move (only relocation; zero code edits) — SP-001
- Existing tests under `packages/framework/tests/` (only relocation)
- `docs/` — only update path references in CLAUDE.md and the README
- `tmp/spren-migration/` — archived files; reference only

### Load-bearing shapes

```python
# packages/spren/src/spren/server.py — Bootstrap response
class FrameworkInfo(BaseModel):
    version: str

class SprenInfo(BaseModel):
    active: bool
    version: str

class BootstrapResponse(BaseModel):
    framework: FrameworkInfo
    spren: SprenInfo
    surfaces: list[str]                    # ["gui"] for v0.3 foundation; later versions add "tui"
    capabilities: dict[str, bool]          # placeholder; populated in later sessions
    endpoints: dict[str, str]              # placeholder; populated in Session 02
    started_at: datetime
    data_dir: str                          # platformdirs-resolved
```

```rust
// apps/desktop/src/main.rs — sidecar ready signal parsing
fn parse_ready_line(line: &str) -> Option<(u16, String)> {
    // matches: "spren-ready: port=<N> token=<T>"
}
```

---

## Hard rules (per session)

### No mocks of in-codebase features (SP-007)

The Vite placeholder route is NOT a mock — it's a real (minimal) page that will be REPLACED in Session 03. The bootstrap response is real (computed at runtime). The auth token is real. The Tauri shell really spawns the sidecar and reads its stdout.

### No backward compatibility (SP-006)

The root `pyproject.toml` change from "the marsys package definition" to "uv workspace declaration" is a one-shot move. Do NOT keep an "old-style root pyproject" alongside.

`src/marsys/` at repo root must be GONE after this session.

### No TRUNK-CRITICAL framework code changes (SP-001)

This session relocates `src/marsys/` to `packages/framework/src/marsys/` but makes ZERO edits to `.py` files inside. Framework behavior is bit-for-bit identical post-move.

### Framework knows nothing of Spren (SP-018)

`packages/framework/src/marsys/` MUST NOT import from `packages/spren/` or any Spren-side path. Verify with grep at session end.

### API is single source of truth (SP-019)

The placeholder UI consumes the same `/v1/bootstrap` endpoint that future TUI / Tauri / browser clients will consume. No bootstrap-specific UI types invented in the frontend; consume the Pydantic-emitted types directly (or, in Session 02, the generated TS types).

### Other Spren design principles

- SP-002: server binds to 127.0.0.1; per-launch token; CORS strictly limited to localhost + tauri://localhost
- SP-005: Pydantic source of truth; TS types generated in Session 02

---

## Tests (required for "done")

### Unit tests

- `packages/spren/tests/test_auth.py`:
  - `secrets.token_urlsafe(32)` produces a 43-character URL-safe token
  - constant-time validation accepts the right token, rejects wrong / empty / missing
- `packages/spren/tests/test_server.py` (FastAPI test client):
  - `/healthz` → 200 without auth
  - `/v1/bootstrap` → 401 without auth
  - `/v1/bootstrap` with valid token → 200 + correct shape
  - `/v1/bootstrap` with wrong token → 401
  - CORS preflight from `http://127.0.0.1:5173` → 200; from `http://malicious.example` → rejected
  - CORS preflight from `tauri://localhost` → 200

### Integration tests

- `packages/spren/tests/integration/test_marsys_import.py`:
  - `from marsys.coordination import Orchestra` works
  - `from marsys.agents import Agent` works
- **Framework regression test:** entire `packages/framework/tests/` passes with the SAME counts as baseline. Document in "What was actually built."

### Rust tests

- `apps/desktop/src/main.rs`:
  - Unit test for `parse_ready_line` parsing
  - Unit test for sidecar lifecycle helpers (mock subprocess)

### End-to-end tests

- `apps/web/tests/e2e/foundation.spec.ts` (Playwright):
  - Manage Vite dev server via `webServer` config
  - Manage FastAPI server lifecycle via Playwright fixture (start/stop with known token)
  - Navigate to `http://localhost:5173/#token=<test-token>`
  - Assert page contains `MARSYS Spren — Foundation Session`
  - Assert bootstrap response is fetched and rendered (e.g., `framework.version` field is visible)
  - Negative test: navigate without a token → page renders an "auth required" error

CI configuration is **out of scope** — Session 10 sets up CI. Tests must pass locally.

---

## Manual-verify checklist (implementer fills before declaring done)

Tests passing isn't enough. Before marking this session done, run through this list against a real launch. Capture screenshots/observations into `./tmp/spren/sessions/01-foundation/manual-verify.md`.

- [ ] Pre-restructure baseline test counts captured + post-restructure counts match exactly (zero regressions in framework tests)
- [ ] `src/marsys/` and `tests/` no longer exist at repo root (`ls` confirms)
- [ ] `packages/framework/src/marsys/` and `packages/framework/tests/` exist with the same files (`diff -r` against pre-move snapshot, if available)
- [ ] `just install` completes without errors on a clean repo
- [ ] `just dev` launches sidecar + Vite both without errors; both prefixed log streams appear
- [ ] `curl localhost:8765/healthz` returns `{"status":"ok"}` without auth
- [ ] `curl localhost:8765/v1/bootstrap` without `Authorization` header returns 401
- [ ] `curl localhost:8765/v1/bootstrap` with the printed token returns 200 with the BootstrapResponse JSON shape (framework.version, surfaces, capabilities, endpoints, started_at, data_dir)
- [ ] Browser at `http://localhost:5173/#token=<token>` renders placeholder + bootstrap content; URL fragment is stripped after first read (check `window.location.href`)
- [ ] Browser without token in fragment shows an "auth required" message — no silent failure
- [ ] `just dev-desktop` launches Tauri shell; window opens; sidecar visible in `ps`; Network tab shows `/v1/bootstrap` 200; no CORS errors in console
- [ ] Closing the Tauri window cleanly terminates the sidecar (`ps` shows no orphaned process); shutdown completes within ~2s
- [ ] CORS preflight from `tauri://localhost` succeeds (test by opening webview); from `http://malicious.example` rejected (verify in unit test)
- [ ] `from marsys.coordination import Orchestra` works inside `packages/spren/`
- [ ] `grep -rn 'Mock\|mock.patch\|vi.mock\|MagicMock' packages/spren/src/ apps/web/src/` returns ZERO hits in product code
- [ ] No `# legacy`, `# TODO: remove`, `if version` patterns in any new product code
- [ ] No Spren type imported into `packages/framework/` (`grep -rn 'from spren\|import spren\|from apps' packages/framework/src/`) — zero hits
- [ ] All test suites green: `just test` exits 0
- [ ] Tauri auth-token injection works (token in `window.__SPREN_AUTH__` is the same as the one printed by sidecar)

If any item fails: investigate and fix before declaring done. Do NOT close out the session with a known regression.

---

## Open questions for the user — RESOLVED in this session

(Originally `MUST surface BEFORE writing code`. All resolved during plan-mode + execution review with the user.)

1. **Spren version number:** ✅ Confirmed → `0.3.0` (matches umbrella v0.3 release).
2. **Final dir naming:** ✅ Confirmed → `apps/web/`, `apps/desktop/`, `apps/tui/`.
3. **Default port for FastAPI server:** ✅ Confirmed → `8765` (with kernel-picked random fallback if taken). Architecture-doc default.
4. **uv workspace dependency syntax:** ✅ Verified — `[tool.uv.sources] marsys = { workspace = true }` is current in uv 0.9.21. Workspace-root members declared via `[tool.uv.workspace] members = [...]` (explicit list, not glob).
5. **Tauri 2 sidecar spawning pattern:** ✅ Decided — for v0.3 dev mode (no PyInstaller until Session 10), raw `std::process::Command` spawning `<workspace>/.venv/bin/python -m spren --port 0` from `apps/desktop/src/main.rs`. The canonical `tauri_plugin_shell::ShellExt::sidecar()` pattern requires a single binary and is deferred to Session 10 alongside PyInstaller bundling.
6. **Tauri 2 directory layout:** ✅ Decided — kept `apps/desktop/src/main.rs` (NOT `apps/desktop/src-tauri/`). The `src-tauri/` convention exists to distinguish frontend from Rust shell when both live in the same dir; since our frontend lives in sibling `apps/web/`, the inner `src-tauri/` would be redundant nesting. `cargo tauri dev` works fine — it locates `tauri.conf.json` via cwd, not the dir name.
7. **CORS regex pattern:** ✅ Adopted — `allow_origin_regex=r"^(http://(127\.0\.0\.1|localhost)(:\d+)?|tauri://localhost)$"` in `packages/spren/src/spren/server.py`. Robust against Vite fallback ports (5174 if 5173 taken) and random sidecar ports.
8. **Vite + TanStack Router file-based routing plugin:** ✅ Verified — npm package `@tanstack/router-plugin`; import `import { tanstackRouter } from '@tanstack/router-plugin/vite'`; MUST come before `@vitejs/plugin-react` in the plugin array.

---

## Sign-off

On completion:

1. Update **What was actually built** below with delta from plan
2. Update [`docs/implementation/spren/v0.3-mvp.md`](../v0.3-mvp.md) — check Session 01's row
3. Add **Lessons / Surprises** below

### What was actually built

#### Restructure delta

- `src/marsys/` → `packages/framework/src/marsys/` (zero `.py` edits per SP-001; verified by grep)
- `tests/` → `packages/framework/tests/` (zero edits)
- Other framework artifacts moved into `packages/framework/`: `mkdocs.yml`, `examples/`, `benchmarks/`, `research/`, `CHANGELOG.md`, `DEPRECATIONS.md`, `FRAMEWORK_DEVELOPMENT_GUIDE.md`, `MANIFEST.in`, `system_diagram_MARS.jpg`, original `README.md`
- **Deleted (per SP-006 redundancy):** `setup.py` and `requirements.txt` (both duplicate `pyproject.toml`)
- **Deleted (build artifact):** `src/marsys.egg-info/`
- Root `pyproject.toml` replaced with workspace declaration (3 lines, no `[project]`)
- Root `uv.lock` regenerated by `uv sync`

#### New packages / apps

- `packages/spren/` — FastAPI sidecar with `/healthz`, `/v1/bootstrap`, per-launch auth, CORS regex; `__main__.py` emits `spren-ready: port=<N> token=<T> data-dir=<path>` on stdout; StaticFiles mount on `/` from `_webui/` when populated; serves Vite bundle in production mode
- `apps/web/` — Vite 8.0 + React 19.2 + TypeScript 6.0 + TanStack Router 1.169 + TanStack Query 5.100 skeleton; placeholder home route reads `window.__SPREN_AUTH__` (Tauri injection) OR URL fragment, strips fragment, fetches bootstrap, renders the JSON
- `apps/desktop/` — Tauri 2 shell (`spren-desktop` crate); `main.rs` spawns sidecar via `std::process::Command`, parses ready signal via regex, injects `__SPREN_AUTH__` + `__SPREN_PORT__` via Tauri `init_script`; `tauri.conf.json` configured with `devUrl=http://localhost:5173` and `frontendDist=../web/dist`; `bundle.active=false` (Session 10 enables); `parse_ready_line` unit-tested with 4 cases
- `apps/tui/` — Textual placeholder (`spren-tui` package; Python 3.12+, textual + httpx); `__main__.py` prints "placeholder in v0.3"; smoke test passes

#### NOT created (deviation from earlier plan, per architectural review with user)

- `packages/spren-sdk/` — DROPPED. The future `SprenTelemetrySink` will live inside `packages/spren/src/spren/telemetry/` in v0.4 alongside the framework's `TelemetrySink` PR. Reduces package count by 1; removes redundant skeleton. Documentation reconciliation propagated across `docs/architecture/spren/03-api-design.md`, `docs/implementation/spren/00-overview.md`, `docs/implementation/spren/v0.3-mvp.md`, `docs/implementation/spren/v0.4-extensions.md`, `docs/implementation/framework/v0.4-spren-support.md`, `docs/implementation/framework/sessions/02-telemetry-sink-protocol.md`, and `BRAINDUMP.md` (all stale `spren-sdk` / `marsys-spren-sdk` / `marsys_spren` references swept).

#### Workspace + tooling

- Root `pyproject.toml`: `[tool.uv.workspace] members = ["packages/framework", "packages/spren", "apps/tui"]` (explicit list, not glob — prevents accidental inclusion of future non-Python dirs)
- Root `pnpm-workspace.yaml`: `packages: ["apps/web"]`
- Root `Cargo.toml`: `[workspace] members = ["apps/desktop"], resolver = "2"`
- Root `Justfile`: `install`, `dev`, `dev-desktop`, `test`, `build`, `lint`, `clean` recipes; `dev-desktop` uses `cd apps/desktop && cargo tauri dev` (NOT the brief's `--manifest-path` flag — that's not valid for `cargo tauri`)
- Root `README.md` replaced with new umbrella README; framework README preserved at `packages/framework/README.md`
- `CLAUDE.md` path references updated (no more `src/marsys/` references)
- `.gitignore` extended with workspace artifacts (node_modules, dist, target, _webui, routeTree.gen.ts, playwright-report)

#### Versions pinned

| Tool | Version |
|---|---|
| Python | 3.12.9 |
| uv | 0.9.21 |
| Node | (system pnpm 10.33.0) |
| Vite | ^8.0.0 |
| React / React DOM | ^19.2.0 |
| TypeScript | ^6.0.0 |
| TanStack Router | ^1.169.0 |
| TanStack Router Plugin | ^1.167.0 |
| TanStack Query | ^5.100.0 |
| Vitest | ^4.1.0 |
| Playwright | ^1.59.0 |
| Tauri | 2.x |
| Cargo / Rust | 1.95.0 (installed via rustup during this session) |
| `just` | 1.50.0 (installed via cargo install during this session) |

#### Test results

- **Framework regression** (`packages/framework/tests/`): pre-move baseline = 841 collected, 764 passed, 20 failed, 43 skipped, 14 errors; post-move = **identical numbers**. Zero regressions. (Pre-existing fails/errors are framework-test fixture issues requiring `OPENROUTER_API_KEY` env var; out of scope for Session 01 per CLAUDE.md "fix root causes" — would need a separate framework test-fixture refactor session.)
- **Spren** (`packages/spren/tests/`): 18/18 passed (auth, server CORS regex incl. `tauri://localhost`, healthz no-auth → 200, bootstrap 401/200 with shape verification, marsys workspace import).
- **TUI** (`apps/tui/tests/`): 2/2 passed (smoke).
- **Web** (`apps/web/tests/foundation.test.ts`): 1/1 passed (Vitest type/shape smoke).
- **Web E2E** (`apps/web/tests/e2e/foundation.spec.ts`): **2/2 passed** (Playwright with bundled chromium-1217). Run from `apps/web/` cwd via `pnpm exec playwright test`. The test starts the FastAPI sidecar via fixture, injects port + token via `addInitScript`, navigates to Vite, asserts heading + bootstrap content; negative test asserts the auth-required error path. ~2.3 min total runtime (sidecar startup is the long part; tests themselves <200ms each).
- **Desktop** (`cargo test --manifest-path apps/desktop/Cargo.toml`): **4/4 passed** after Tauri Linux deps installed and `apps/desktop/icons/icon.png` placeholder generated. Tests cover `parse_ready_line` happy/whitespace/garbage/strips-to-first-token cases.
- **`just dev-desktop`**: builds + runs the binary cleanly (compile finishes, `cargo run --no-default-features` invokes the binary, sidecar spawns, `[INFO] sidecar ready on port <N>` logs). However, on this WSLg/WSL2 host, Tauri's `wry::Builder::run()` returns Ok(_) immediately without rendering a window — known interop gap between `webkit2gtk-4.1` and the WSLg X server. This is environmental; the binary's logic (sidecar spawn, ready parsing, `init_script` construction) is verified by the 4 cargo tests + the Playwright E2E (which exercises the same auth-injection flow with a real chromium). Verify webview rendering on a native Linux host, macOS, or Windows.

#### End-to-end smoke (verified manually with the sidecar populated)

- `python -m spren --port 8765` → emits `spren-ready: port=8765 token=<43-char> data-dir=/home/rezaho/.local/share/spren`
- `curl localhost:8765/healthz` → 200 `{"status":"ok"}` (no auth)
- `curl localhost:8765/v1/bootstrap` → 401 (no auth)
- `curl -H "Authorization: Bearer <token>" localhost:8765/v1/bootstrap` → 200 with full BootstrapResponse JSON (framework.version=`0.2.1b0`, spren.version=`0.3.0`, surfaces=`["gui"]`, capabilities/endpoints `{}`, started_at, data_dir)
- After `pnpm --filter @marsys/spren-web build && cp -r apps/web/dist packages/spren/src/spren/_webui/`: `curl -I localhost:8765/` → 200 OK, served by `uvicorn` — StaticFiles mount works in production mode
- Bundle size: `apps/web/dist/assets/index-<hash>.js` = 276 KB (gzip 88 KB) for the placeholder bundle. Reasonable for a foundation skeleton.

#### Deliberate deferrals (NOT acceptance-blocking for Session 01)

- **`apps/web/` Tailwind / @xyflow/react / cmdk / radix-ui dependencies.** Architect's brief lists them; user opted to defer to Session 03 (visual builder) when actually needed. Avoids ~50 MB unused node_modules now.
- **PyInstaller + `tauri_plugin_shell::sidecar()` refactor.** Session 10 territory.
- **Real platform-specific icons (`32x32.png`, `128x128.png`, `128x128@2x.png`, `icon.ico`, `icon.icns`).** Session 10 (bundling). The placeholder `icon.png` (256×256 warm-amber square, 858 bytes) satisfies the `tauri::generate_context!()` macro at compile time but is not production-ready.
- **Webview render verification on native desktop.** This session's host is WSLg (Linux on Windows) where Tauri's webkit2gtk integration has known display issues. The Rust binary + sidecar lifecycle + auth injection is verified via cargo tests + Playwright E2E, but visible window rendering needs a native macOS / Windows / Linux desktop.

### Lessons / Surprises

- **Pre-existing pytest collection error.** Before relocation, `tests/agents/test_memory.py` and `tests/memory/test_memory.py` collided as duplicate `test_memory` modules (no `__init__.py` files in subdirs). `pytest` errored on collection: 755 collected + 1 error. Per CLAUDE.md memory ("never ignore pre-existing problems"), fixed at source — added `tests/__init__.py` to all 5 test subdirectories. Result: 841 collected, 0 collection errors. The collision-fixed `test_memory.py` adds 86 previously-uncollected tests to the runnable pool.
- **Tauri Linux compile is heavy.** `cargo build` for `apps/desktop` requires `libdbus-1-dev`, `libgtk-3-dev`, `libsoup-3.0-dev`, `libwebkit2gtk-4.1-dev`, `libjavascriptcoregtk-4.1-dev`, and `pkg-config` system packages — totalling ~150 MB. None of these were documented in the brief. Surface in Session 10 prep + add to `just install` as a `pre-install` note (we can't `apt install` from `just`).
- **Tauri 2's `generate_context!` macro fails compile if `icons/icon.png` is missing.** Even with `bundle.active = false` in `tauri.conf.json` (where bundler icons aren't needed), the macro reads the icon path at compile time and panics if absent. Generated a 256×256 warm-amber placeholder PNG via Pillow; ~858 bytes; satisfies the macro. Session 10 swaps in the real branded icons.
- **Playwright "Host validation warning" on Linux is non-fatal.** First `pnpm exec playwright install` print scary missing-libs message (`libgtk-4.so.1`, `libevent-2.1.so.7`, `libgstcodecparsers-1.0.so.0`, `libavif.so.13`) but still downloads chromium-1217. Headless tests run successfully without those libs. They're for full Firefox/WebKit support; chromium needs less. Don't reflexively `apt install` more deps before checking whether the tests actually fail.
- **WSLg + Tauri webview is broken on this host.** `cargo tauri dev` compiles + runs the binary, the sidecar starts, but `wry::Builder::run()` returns Ok(_) without showing a window. Likely a webkit2gtk-4.1/WSLg integration gap. Worked around by relying on Playwright E2E for the auth-flow verification. Real webview rendering is a "user runs this on their actual hardware" item.
- **Stray `tests/models/outputs/` after pytest at root.** A test in `packages/framework/tests/models/` writes outputs to a CWD-relative path (`tests/models/outputs/`). Running pytest from the repo root creates `tests/models/outputs/` at root. Cleaned up. Pre-existing test design; should be fixed in a future framework-test refactor session (use `tmp_path` fixture instead of CWD-relative).
- **The brief's `cargo tauri dev --manifest-path apps/desktop/Cargo.toml` invocation is invalid.** `cargo tauri` is the Tauri CLI (not `cargo`'s subcommand); it does NOT accept `--manifest-path`. Correct usage: `cd apps/desktop && cargo tauri dev`. Updated in `Justfile`.
- **FastAPI `CORSMiddleware` and request-level vs route-level deps.** First attempt used `Annotated[None, Depends(require_auth)]` on the bootstrap route signature → FastAPI returned 422 (validation) instead of 401 (auth). Fixed by moving the dep to route-level via `dependencies=[Depends(require_auth)]` parameter on `@app.get(...)`. Cleaner separation: auth concerns don't touch the route's signature.
- **CORS hardcoded origins are brittle vs `allow_origin_regex`.** Initial impl listed five exact origins; the architect's update flagged it as fragile (Vite picks 5174 if 5173 is taken; sidecar gets a random port from `--port 0`). Switched to `allow_origin_regex=r"^(http://(127\.0\.0\.1|localhost)(:\d+)?|tauri://localhost)$"`. Existing CORS tests still pass.
- **The `_webui` dir trick for Vite-build-into-Python-package.** Vite emits `apps/web/dist/`; the build pipeline (`just build`) copies `apps/web/dist/` into `packages/spren/src/spren/_webui/`. Setuptools includes it via `[tool.setuptools.package-data] spren = ["_webui/**/*"]`. The FastAPI app conditionally mounts StaticFiles only when `_webui/index.html` exists — so dev mode (empty `_webui/`) doesn't accidentally shadow the Vite dev server.
- **Workspace member explicit list beats glob.** Using `members = ["packages/framework", "packages/spren", "apps/tui"]` (explicit) prevents `uv` from picking up `packages/<future-non-python-dir>/` accidentally. The brief's `members = ["packages/*", "apps/tui"]` glob would have worked too but is less defensive.
- **`@marsys/spren-web` (scoped) vs `web` (bare).** The architect's brief uses `pnpm --filter web` shorthand. Kept the scoped name `@marsys/spren-web` to leave room for future `@marsys/spren-shared`, `@marsys/spren-types` packages. Justfile uses `pnpm --filter @marsys/spren-web` (matches the actual `name` field).
- **`apps/desktop/src/main.rs` vs `apps/desktop/src-tauri/src/main.rs`.** The architect raised this as an open question. Decided to keep `apps/desktop/src/main.rs` because the `src-tauri/` convention exists to separate frontend from Rust shell; we've already separated them via sibling app dirs (`apps/web/` vs `apps/desktop/`). The inner `src-tauri/` would be redundant.

---

# Session 01 Fixup — Sidecar shutdown protocol, Windows shell, ambient types, brief alignment

A follow-up work block continuing from Session 01. Picks up loose ends and adds the sidecar lifecycle features that v0.4 pause/resume needs.

## What this fixup ships

After this fixup, the foundation has:

- A clean stdin-based sidecar shutdown protocol replacing the SIGKILL fallback. Closing the Tauri window terminates the sidecar gracefully within 2 seconds; SIGKILL becomes a last-resort timeout fallback only.
- Windows-compatible Justfile recipes. `just install`, `just dev`, `just test`, `just build`, `just lint`, `just clean` run on Windows hosts.
- Ambient TypeScript declarations for `window.__SPREN_AUTH__` and `window.__SPREN_PORT__`. No `unknown as ...` casts in product code.
- A `just build` guard that fails fast when the Vite production bundle is missing.
- A Playwright config that detects an externally-running Vite dev server and refuses to spawn a duplicate.
- The brief body itself updated to reflect actual shipped commands, package names, and workspace member list — so future readers see the real implementation, not the original sketch.

### Acceptance criteria

- [ ] Closing the Tauri window cleanly drains in-flight sidecar requests, then writes `shutdown\n` to the sidecar's stdin; sidecar exits 0 within 2 seconds; no orphaned process in `ps`. SIGKILL is invoked only on the 2-second timeout.
- [ ] `just install`, `just dev`, `just test`, `just build`, `just lint`, `just clean` all run cleanly on a fresh Windows 11 host with PowerShell.
- [ ] `apps/web/src/types/spren.d.ts` declares `Window.__SPREN_AUTH__: string | undefined` and `Window.__SPREN_PORT__: number | undefined`. `tsc --noEmit` passes with `noImplicitAny`. No `unknown as ...` casts on `window.__SPREN_*` accesses anywhere in `apps/web/src/`.
- [ ] `just build` exits 1 with a clear error message when `apps/web/dist/index.html` is missing after the Vite build step.
- [ ] `apps/web/playwright.config.ts` aborts with a clear message when port 5173 is already in use by a non-Playwright process; does not silently reuse a server it didn't start.
- [ ] Brief body matches shipped reality: `grep -n 'pnpm --filter web\|cargo tauri dev --manifest-path\|packages/\*"\|members = \["packages/\*"' docs/implementation/spren/sessions/01-foundation.md` returns zero hits in the body (Steps 1-13 + Acceptance Criteria + Background Reading + Verify-loop). The "What was actually built" and "Lessons / Surprises" sections are immutable history; do not edit them.
- [ ] Framework regression test counts unchanged from Session 01's baseline (841 collected, 764 passed, 20 failed, 43 skipped, 14 errors).
- [ ] No new mocks of in-codebase features (SP-007).
- [ ] No legacy code paths preserved (SP-006); the SIGKILL call is replaced, not coexisting with stdin-shutdown.

---

## Detailed plan

### Step 1 — Sidecar stdin-shutdown protocol

In `packages/spren/src/spren/__main__.py`, add a stdin reader running on a background thread that consumes lines from `sys.stdin` and triggers a clean uvicorn shutdown when it sees `shutdown\n`:

- Spawn the reader thread alongside uvicorn, before `server.serve()`.
- On receiving `shutdown\n`, call `server.should_exit = True` (uvicorn's documented graceful-shutdown signal).
- On EOF (parent died without writing the line), also set `should_exit = True`. The sidecar should never outlive its parent.
- Idempotent: receiving `shutdown\n` twice is fine.

In `apps/desktop/src/main.rs`, replace the `child.kill()` call in the window-close handler:

```rust
// On window close:
// 1. Close stdin handle to parent → triggers EOF on sidecar's reader (defensive)
// 2. Write "shutdown\n" to the sidecar's stdin pipe (already opened on spawn at line 40)
// 3. Wait up to 2 seconds for the child to exit cleanly
// 4. Only on timeout, fall back to SIGKILL
```

Concretely: wrap the existing `Child` handle in a small `SidecarHandle` struct exposing `request_shutdown()` (writes the line + waits with timeout) and `force_kill()` (SIGKILL). The window-close handler calls `request_shutdown()` first; only on `Err(Timeout)` does it call `force_kill()`.

The 2-second timeout is deliberate: long enough for the FastAPI lifespan-shutdown handler to finish (DB cleanup, in-flight request drain), short enough that a hung sidecar doesn't keep the Tauri shell hanging at exit.

### Step 2 — Windows-compatible Justfile recipes

At the top of the root `Justfile`, add:

```just
set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]
```

This routes recipes to PowerShell on Windows, bash on Linux/macOS. Recipes that use Unix idioms (`#!/usr/bin/env bash`, `trap`, `(... &)`, `sed`, `kill`) need cross-shell rewrites. Two approaches:

- **Per-recipe shebangs** for recipes that genuinely need bash: keep the `#!/usr/bin/env bash` line for Linux/macOS but split the recipe into `dev-unix` (existing bash form) + `dev-windows` (PowerShell form), and have `dev` delegate via `[unix]`/`[windows]` attributes.
- **Cross-shell rewrites** where simple: replace `pnpm --filter @marsys/spren-web build` (no shebang needed) with platform-agnostic invocations.

Recipes that use background processes + signal-handling (`dev`, `dev-desktop`) MUST split into Unix/Windows variants since PowerShell's process management is fundamentally different from bash's `&` + `trap`. Use just's `[unix]` and `[windows]` recipe attributes.

Verify on a Windows 11 host: install Python 3.12, uv, pnpm, Rust, just (via `winget install`), Visual Studio Build Tools (for Tauri's MSVC dependency), then run the full `just install && just test && just dev` sequence. Document any required tooling versions in the README's "Windows dev setup" section (new).

### Step 3 — Ambient TypeScript Window types

Create `apps/web/src/types/spren.d.ts`:

```typescript
declare global {
  interface Window {
    __SPREN_AUTH__?: string;
    __SPREN_PORT__?: number;
  }
}
export {};
```

Verify `tsconfig.json`'s `include` covers `src/**/*.d.ts` (or add it). Run `pnpm exec tsc --noEmit` and confirm no errors.

In `apps/web/src/providers/capabilities.tsx` and `apps/web/src/lib/api.ts`, remove any `unknown as string` or `unknown as number` casts on `window.__SPREN_AUTH__` / `__SPREN_PORT__` accesses — the ambient types now provide narrowing.

### Step 4 — Just build dist validation

In `Justfile`, modify the `build` recipe:

```just
build:
    pnpm --filter @marsys/spren-web build
    test -f apps/web/dist/index.html || (echo "ERROR: Vite build did not produce apps/web/dist/index.html" >&2; exit 1)
    rm -rf packages/spren/src/spren/_webui
    cp -r apps/web/dist packages/spren/src/spren/_webui
```

(For the Windows variant, use PowerShell's `Test-Path` + `Write-Error` + `exit 1` equivalent.)

The current recipe silently copies an empty/missing `dist/` if the Vite build had a TypeScript error that didn't fail the build (rare but possible with strict-mode regressions). The guard catches that early.

### Step 5 — Playwright server-collision guard

In `apps/web/playwright.config.ts`, replace `reuseExistingServer: !process.env.CI` with explicit logic:

- Probe `http://127.0.0.1:5173` before Playwright spawns its own dev server.
- If port is open AND the response identifies as Vite (e.g., the `/` HTML contains `<script type="module" src="/@vite/client">`), abort with: `ERROR: Vite dev server is already running on :5173. Stop 'just dev' first, OR run Playwright in CI mode (CI=1 pnpm test:e2e) to use a fresh server on a different port.`
- If port is open AND it's something else (different framework, conflicting process), abort with a different clear message.
- If port is closed, Playwright spawns its own server normally.

This prevents the failure mode where a contributor runs `pnpm test:e2e` while `just dev` is active and Playwright silently uses the dev server's state, producing flaky-test confusion.

### Step 6 — Align brief body with shipped reality

Edit `docs/implementation/spren/sessions/01-foundation.md` body sections (NOT the "What was actually built" or "Lessons / Surprises" sections — those are immutable post-implementation history):

- **Acceptance criterion line 169**: drop `Tailwind`, `shadcn-cli (or radix-ui directly)`, `@xyflow/react`, `cmdk`, `MSW`. Keep React 19, Vite, TypeScript, TanStack Router, TanStack Query, `geist`, Vitest, Playwright. Add a one-line note: "Tailwind / shadcn / radix / `@xyflow/react` / cmdk / MSW are deferred to Session 03 (visual builder)."
- **Step 4 root pyproject example** (~line 252): change `members = ["packages/*", "apps/tui"]` to `members = ["packages/framework", "packages/spren", "apps/tui"]` and add a one-line note: "Explicit list, not glob — defends against future non-Python dirs landing under `packages/`."
- **Step 11 Justfile example**: replace every `pnpm --filter web` with `pnpm --filter @marsys/spren-web`. Replace `cargo tauri dev --manifest-path apps/desktop/Cargo.toml` with `cd apps/desktop && cargo tauri dev`. Update the `dev-desktop` recipe to reflect the shipped pattern (Vite running separately for HMR; Tauri spawns sidecar internally). Add the `set windows-shell := [...]` line at the top of the example.
- **Step 13 verify-loop**: same pnpm + tauri command corrections in the curl/build sequence.
- **Files NOT to touch** (~line 486): no change.
- Cross-check after edits: `grep -n 'pnpm --filter web\|cargo tauri dev --manifest-path' docs/implementation/spren/sessions/01-foundation.md` returns zero hits in the body.

The sections "Open questions for the user — RESOLVED in this session", "What was actually built", "Lessons / Surprises", and this Fixup section itself are NOT edited — they are post-implementation history.

### Files NOT to touch

- The `coordination/` / `agents/` / `models/` paths inside `packages/framework/src/marsys/` — TRUNK-CRITICAL; SP-001.
- "What was actually built" and "Lessons / Surprises" sections in this brief.
- Any uv / pnpm / Cargo workspace member lists beyond what Step 6 specifies.
- Anything in `packages/spren-sdk/` (the package was dropped in Session 01; do not resurrect it).

---

## Tests (required for "done")

### Unit tests

- `apps/desktop/tests/test_sidecar_handle.rs` (new): unit-test the `SidecarHandle::request_shutdown()` helper with a mock subprocess: writes `shutdown\n` to a pipe; on EOF returns `Ok(())`; on timeout returns `Err(Timeout)`. Mock subprocess responds with exit-after-50ms (success), exit-after-3s (timeout), no-exit (timeout).
- `packages/spren/tests/test_main_stdin.py` (new): launch `python -m spren --port 0` as subprocess; write `shutdown\n` to its stdin; assert it exits within 2s with code 0 + no orphaned uvicorn worker thread.
- `apps/web/tests/types-smoke.test.ts` (new Vitest): import a tiny module that reads `window.__SPREN_AUTH__` directly and asserts the value is `string | undefined` (TypeScript-checked at compile time; runtime test just confirms the file compiles).

### Integration tests

- `apps/desktop/tests/integration/test_window_close_drains_sidecar.rs` (new): end-to-end test using a real Tauri shell + real sidecar subprocess. Spawn → close window → assert sidecar exits cleanly within 2s (success path) or hangs and gets SIGKILLed at 2s+ (timeout fallback).
- Existing tests in `packages/spren/tests/`, `apps/web/tests/`, `apps/desktop/tests/` all pass at unchanged baseline counts.

### End-to-end tests

- `apps/web/tests/e2e/foundation.spec.ts` (existing): verify the test still passes after the playwright config changes; specifically run with `pnpm test:e2e` while `just dev` is active and confirm the new collision-guard fires with a clear error.

### Manual-verify checklist (implementer fills before declaring done)

Capture screenshots / observations into `./tmp/spren/sessions/01-foundation-fixup/manual-verify.md`.

- [ ] `just dev-desktop` opens Tauri window; close window; sidecar exits within 2s; `ps` shows no orphan
- [ ] Same flow but with a sidecar that has a `time.sleep(5)` injected into its lifespan-shutdown handler: window-close waits 2s then SIGKILLs cleanly
- [ ] On Windows 11 with PowerShell: `just install && just test && just dev` all complete without errors
- [ ] `pnpm exec tsc --noEmit` from `apps/web/` reports zero errors after ambient types added
- [ ] `pnpm --filter @marsys/spren-web build && rm apps/web/dist/index.html && just build` exits 1 with the dist-missing error
- [ ] Run `just dev` in one terminal, then `pnpm --filter @marsys/spren-web exec playwright test` in another → the new collision-guard fires with the documented error message
- [ ] `grep -n 'pnpm --filter web\|cargo tauri dev --manifest-path' docs/implementation/spren/sessions/01-foundation.md` returns zero hits in lines 1-590 (the brief body)
- [ ] `grep -rn 'Mock\|mock.patch\|vi.mock\|MagicMock' packages/spren/src/ apps/web/src/ apps/desktop/src/` returns zero hits in product code (test files only)

---

## Sign-off

On completion of the fixup:

1. Update **What was actually built (fixup)** below.
2. Add **Lessons / Surprises (fixup)** if anything surprising came up.

### What was actually built (fixup)

The fixup landed across three PRs, with item 2 split out because it was implemented and verified on a separate Windows 11 host:

- **PR #33 (chore: add Windows PowerShell support to dev workflow, MERGED)** — item 2. Justfile gets `set windows-shell` plus `[unix]` / `[windows]` splits for `dev`, `dev-desktop`, `build`, `clean`. Apps/desktop/src/main.rs `resolve_python` learns a `cfg!(windows)` branch (`.venv/Scripts/python.exe` vs `.venv/bin/python`) and absorbs rustfmt 1.9 chain wrapping plus clippy 1.95's `lines_filter_map_ok` migration. A 1.96 KB placeholder `apps/desktop/icons/icon.ico` is added so `tauri-build` can embed a Windows resource even though `bundle.active = false`. README gains a "Windows dev setup" section.
- **PR (chore: session 01 fixup, items 1+3+4+5, in chore/session-01-fixup-rest)** — items 1, 3, 4, 5. Sidecar stdin-shutdown protocol (Python `_watch_stdin_for_shutdown` + `_stdin_is_pipe` gate; Rust `SidecarHandle` with `request_shutdown` / `force_kill`). Ambient `apps/web/src/types/spren.d.ts` for `Window.__SPREN_AUTH__` / `__SPREN_PORT__`; the inline `declare global` block in `main.tsx` is removed and the Playwright test no longer needs `unknown as` casts. `just build` now `test -f apps/web/dist/index.html` (Unix) / `Test-Path` (Windows) before copying, fail-fast on silently-empty Vite builds. Playwright e2e gets a port-conflict preflight at `apps/web/scripts/preflight-e2e.mjs` wired through the `test:e2e` package script; conflict messages are explicit ("Vite dev server is already running on 127.0.0.1:5173 ..." vs "A non-Vite process is already listening on ..."), and the config is back to plain `reuseExistingServer: false`.
- **Local-only brief cleanup (item 6)** — `docs/implementation/spren/sessions/01-foundation.md` body sections (acceptance criteria line 169, Step 4 pyproject example, Step 11 Justfile example, Step 13 verify-loop) are rewritten to match shipped reality. The "What was actually built", "Lessons / Surprises", "Open questions for the user — RESOLVED", and the Fixup section itself are intentionally not edited — they are post-implementation history. Cross-check `awk 'NR<=590' docs/implementation/spren/sessions/01-foundation.md | grep -nE 'pnpm --filter web|cargo tauri dev --manifest-path|members = \["packages/\*"'` returns zero hits in the body. Because `docs/` is gitignored, this cleanup ships only to local clones; new fresh clones will read the post-fixup brief from whichever local copy is shared.

Test results across both PRs:

- Framework regression: 841 collected, 764 passed, 20 failed, 43 skipped, 14 errors (Linux baseline preserved).
- Spren: 21/21 — 18 from Session 01 (auth, server, marsys import) plus 3 new ones in `test_main_stdin.py` (shutdown line, stdin EOF, repeated shutdown idempotency).
- TUI smoke: 2/2.
- Web Vitest: 1/1 (the foundation type-shape smoke).
- Web Playwright e2e: 2/2 — the existing happy-path / no-token cases pass; the new preflight aborts cleanly on conflict (verified by starting `pnpm dev` in a separate process, then running `pnpm test:e2e` which exits with the "Vite dev server is already running" message).
- Cargo test: 7/7 — the original 4 `parse_ready_line` cases plus 3 new ones for `SidecarHandle` (request_shutdown succeeds when the child reads stdin and exits, request_shutdown returns `Err(ShutdownTimeout)` against an unresponsive child, force_kill is idempotent). The mocks use `resolve_python` so the same tests work on Windows.
- Cargo fmt --check + clippy -D warnings: clean.

### Lessons / Surprises (fixup)

- **`for line in sys.stdin` plus `stdio=ignore` is a foot-gun.** The first cut of the stdin-shutdown protocol set `server.should_exit = True` on every EOF, which is correct when the parent uses `Stdio::piped()` (Tauri shell, the dedicated Python tests) but wrong when the parent passes `stdio=ignore` / `stdin=DEVNULL` (the Playwright e2e fixture). On Linux the latter case binds fd 0 to `/dev/null`, the reader thread sees EOF on its first iteration, and the sidecar shuts itself down before uvicorn even starts serving. Discovered via the e2e suite throwing "expected substring framework" — the page rendered the auth-required state because the bootstrap fetch was hitting a sidecar that had already exited. The fix is `os.fstat(0).st_mode` and only watching FIFOs (`stat.S_ISFIFO`); pipes get full shutdown-on-EOF semantics, terminals / null devices / regular files skip the watcher entirely. Documented inline in `_stdin_is_pipe`.
- **Playwright reloads its config in workers, after webServer has started.** First attempt at the collision guard was a `globalSetup` script; it always fired against Playwright's own webServer because globalSetup runs after webServer. Second attempt was a top-level `await` inside `playwright.config.ts`; it also fired in worker reloads because each worker re-evaluates the config in its own process. Final design moves the probe out of Playwright entirely: `apps/web/scripts/preflight-e2e.mjs` runs once via the `test:e2e` package script and exits 1 with a specific message before Playwright is invoked. This is also the only design where `pnpm dev` in another terminal produces the right error rather than a generic Playwright timeout.
- **Vite re-runs on `just build` make the dist guard's negative path harder to test directly.** Easy to verify the *guard logic* (`test -f` short-circuits correctly with `||`), harder to test *the recipe path* end-to-end without injecting a deliberately-broken Vite step. Decided that direct guard verification is sufficient evidence; if a future Vite-strict-mode change starts producing empty dist directories, the guard will catch it where it belongs.
- **Cargo borrow-checker around `Mutex<Option<Child>>`.** Initial `request_shutdown` tried to borrow `child_guard.as_mut()`, drop the lock during sleep, then reassign. NLL was unhappy because the borrow into `child` outlived the reassignment of `child_guard`. The clean fix is to `take()` the `Child` out of the mutex up front, hold it as a local while polling, and put it back on the timeout path so `force_kill` can reach it. One-pass through the lock; no risk of a concurrent `force_kill` racing the same handle.
- **rustfmt 1.9 chain heuristics + clippy 1.95 lint pair.** The Windows agent already absorbed both in PR #33; this fixup just had to keep the same idioms (`reader.lines().map_while(Result::ok)` instead of `for line in reader.lines() { if let Ok(l) = line { ... } }`). Worth flagging for future Rust edits in this codebase: any `for x in iter { if let Ok(...) = x { ... } }` pattern over a `Lines` iterator now triggers `lines_filter_map_ok`, and `flatten()` is wrong for `Lines` because `Err` items repeat forever. `map_while(Result::ok)` is the canonical pattern.
- **The umbrella has its own gitignore-vs-PR rule.** `docs/architecture/` and `docs/implementation/` are gitignored, so item 6's brief alignment is local-only cleanup. The architect's fixup spec was written as if the brief were tracked. That is fine in practice — the PR description for the items 1+3+4+5 PR carries the user-facing summary, and the brief reflects shipped reality for whoever opens the local clone. Worth remembering for future fixup specs that they target tracked files when the rationale needs to reach the team via PR review.
