# MARSYS Spren

The open-source umbrella product of [Marsys AI](https://marsys.ai). Three things in one monorepo:

- **`packages/framework/`** — the [marsys](https://pypi.org/project/marsys/) Python framework for orchestrating multi-agent systems. `pip install marsys`.
- **`packages/spren/`** — Spren, a continuously-active personal AI assistant ("the meta-agent that runs your other agents") layered on top of the framework. Native installer per platform; secondary channels via Homebrew, winget, apt, npm, pipx, Docker.
- **`apps/web/` + `apps/desktop/` + `apps/tui/`** — the visual builder + run inspector (Vite + React + TanStack Router), Tauri 2 desktop shell, and Textual TUI client. All consume the same FastAPI surface served by `packages/spren/`.

The MARSYS Spren umbrella is open-source and local-first. See [`LICENSE`](./LICENSE) for the per-package licensing breakdown.

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

### Windows dev setup

The Justfile recipes work on Windows 11 PowerShell. Recipes that use bash-only idioms (`dev`, `dev-desktop`, `build`, `clean`) are split into `[unix]` and `[windows]` variants; the rest run unchanged across shells.

Install the toolchain via winget:

```powershell
winget install astral-sh.uv
winget install OpenJS.NodeJS.LTS
winget install pnpm.pnpm
winget install Rustlang.Rustup
winget install Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
winget install Casey.Just
winget install GitHub.cli
winget install Git.Git
cargo install tauri-cli --version "^2"
```

Notes:

- The Visual Studio Build Tools install must include the **Desktop development with C++** workload. The `--override` flag above selects it; if you install through the GUI, pick the workload manually.
- WebView2 ships with Windows 11. If missing, get the Evergreen Bootstrapper from Microsoft.
- Clone into a path on a local NTFS drive (e.g. `C:\dev\marsys-spren`). Working from a WSL share (`\\wsl$\...` or a drive mapped to `\\wsl.localhost\...`) breaks `uv` venv creation because the venv layout uses POSIX symlinks the share cannot represent.
- If your repo path is deep, enable long path support: `git config --global core.longpaths true`.
- `cargo install tauri-cli` takes ~5 min on first build.

Verified versions used during the Windows fixup: Python 3.12.12 (managed by uv), Node 22+, pnpm 10.33+, Rust 1.95 / cargo 1.95, just 1.50, GitHub CLI 2.92.

Then the same flow as Linux works:

```powershell
git clone https://github.com/rezaho/marsys.git marsys-spren
cd marsys-spren
just install
just dev
just dev-desktop
just test
```

### Spren as an end-user product

Native installers, brew tap, winget manifest, apt repo, npm wrapper, pipx, and Docker arrive in v0.3 release (Session 10).

## Documentation

- `docs/architecture/spren/` — architecture (overview, system context, API design, frontend, packaging, security, design principles, meta-agent, memory)
- `docs/implementation/spren/` — version roadmap (v0.3 MVP, v0.4 extensions, v0.5 future) and per-session briefs

## License

This monorepo distributes code under two distinct licenses:

- `packages/framework/` is licensed under the Apache License, Version 2.0 (`Apache-2.0`). Permissive, with no restrictions on use, modification, or commercial distribution beyond attribution.
- `packages/spren/`, `apps/web/`, `apps/desktop/`, and `apps/tui/` are licensed under the Functional Source License, Version 1.1, ALv2 Future License (`FSL-1.1-ALv2`). Source-available with a 2-year non-compete that automatically converts to Apache 2.0. Free for individual use, internal company use, education, research, modification, and redistribution with attribution preserved. Forbids offering as a competing commercial managed or hosted service.

See [`LICENSE`](./LICENSE) for the full umbrella explainer and the per-package `LICENSE` files for the canonical license texts.
