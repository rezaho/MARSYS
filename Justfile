default:
    @just --list

# Install Python + JS + Rust deps + tauri-cli
install:
    uv sync --extra test
    pnpm install
    cargo fetch --manifest-path apps/desktop/Cargo.toml
    cargo install tauri-cli --version "^2"

# Run dev: FastAPI sidecar + Vite dev server (no Tauri)
dev:
    #!/usr/bin/env bash
    set -euo pipefail
    (uv run --package spren python -m spren --port 8765 2>&1 | sed 's/^/[sidecar] /') &
    SIDECAR_PID=$!
    trap "kill $SIDECAR_PID 2>/dev/null || true" EXIT INT TERM
    VITE_SPREN_API_URL=http://127.0.0.1:8765 pnpm --filter @marsys/spren-web dev 2>&1 | sed 's/^/[vite] /'

# Run dev-desktop: Tauri shell launches sidecar internally; Vite still runs separately for HMR
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
    pnpm --filter @marsys/spren-web test --run
    cargo test --manifest-path apps/desktop/Cargo.toml

# Build: Vite production bundle copied into spren/_webui/
build:
    pnpm --filter @marsys/spren-web build
    rm -rf packages/spren/src/spren/_webui
    cp -r apps/web/dist packages/spren/src/spren/_webui

# Lint
lint:
    pnpm --filter @marsys/spren-web typecheck
    cargo fmt --manifest-path apps/desktop/Cargo.toml --check
    cargo clippy --manifest-path apps/desktop/Cargo.toml -- -D warnings

# Clean
clean:
    rm -rf apps/web/dist apps/web/node_modules
    rm -rf packages/*/dist packages/*/build packages/*/*.egg-info
    rm -rf packages/spren/src/spren/_webui
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    cargo clean --manifest-path apps/desktop/Cargo.toml || true
