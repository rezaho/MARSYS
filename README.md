# MARSYS Spren

The open-source umbrella product of [Marsys AI](https://marsys.ai). Three things in one monorepo:

- **`packages/framework/`** — the [marsys](https://pypi.org/project/marsys/) Python framework for orchestrating multi-agent systems. `pip install marsys`.
- **`packages/spren/`** — Spren, a continuously-active personal AI assistant ("the meta-agent that runs your other agents") layered on top of the framework. Native installer per platform; secondary channels via Homebrew, winget, apt, npm, pipx, Docker.
- **`apps/web/` + `apps/desktop/` + `apps/tui/`** — the visual builder + run inspector (Vite + React + TanStack Router), Tauri 2 desktop shell, and Textual TUI client. All consume the same FastAPI surface served by `packages/spren/`.

The MARSYS Spren umbrella is **open-source (Apache-2.0) and local-first**. Hosted multi-tenant execution is the proprietary [MARSYS Cloud](https://marsys.ai); the hosted SaaS UI is [MARSYS Studio](https://marsys.ai). Both are sibling products in separate repos and out of scope here.

## Status

**v0.3 in development.** Foundation session (Session 01) ships the workspace structure and the FastAPI sidecar entry point. Subsequent sessions add workflow CRUD, the visual builder, run execution, the meta-agent runtime, and native packaging.

## Quick start

### Just the framework

```bash
pip install marsys
```

See `packages/framework/README.md` for framework usage.

### The full umbrella (developer / contributor)

Prerequisites: Python 3.12+, Node 22+ with pnpm, Rust toolchain (`rustup`), and [`just`](https://github.com/casey/just).

```bash
git clone https://github.com/rezaho/marsys.git marsys-spren
cd marsys-spren
just install
just dev          # starts FastAPI sidecar + Vite dev server
just dev-desktop  # adds the Tauri shell
just test         # runs all tests across Python, JS, Rust
```

### Spren as an end-user product

Native installers, brew tap, winget manifest, apt repo, npm wrapper, pipx, and Docker arrive in v0.3 release (Session 10).

## Documentation

- `docs/architecture/spren/` — architecture (overview, system context, API design, frontend, packaging, security, design principles, meta-agent, memory)
- `docs/implementation/spren/` — version roadmap (v0.3 MVP, v0.4 extensions, v0.5 future) and per-session briefs

## License

Apache-2.0. See [`LICENSE`](./LICENSE).
