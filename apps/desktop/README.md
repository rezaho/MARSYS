# spren-desktop

Tauri 2 shell for MARSYS Spren. v0.3 foundation: spawns the Python sidecar
(`<workspace>/.venv/bin/python -m spren --port 0`) via raw `std::process::Command`,
reads the ready signal from stdout, opens the webview pointed at the Vite dev
server (in dev) with `__SPREN_AUTH__` and `__SPREN_PORT__` injected as window
variables.

PyInstaller-bundled binary + canonical `tauri_plugin_shell::sidecar()` pattern
arrive in Session 10 alongside native installer packaging.

## Prerequisites

- Rust toolchain (`rustup`) and Cargo
- `cargo install tauri-cli --version "^2"` (run by `just install`)
- A working Python `.venv` at the workspace root with the `spren` package installed

## Run

```bash
cd apps/desktop
cargo tauri dev
```

(Use `just dev-desktop` from the repo root for the orchestrated start that also runs the Vite dev server.)

## Test

```bash
cargo test
```

Tests cover the `parse_ready_line` parser. Sidecar lifecycle integration is
exercised via the Playwright E2E suite in `apps/web/`.
