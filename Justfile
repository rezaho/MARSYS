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

# Run dev: FastAPI sidecar + Vite dev server (no Tauri)
[windows]
[script]
dev:
    $ErrorActionPreference = 'Stop'
    $sidecarJob = Start-Job -Name spren-sidecar -ScriptBlock {
        Set-Location $using:PWD
        cmd.exe /c "uv run --package spren python -m spren --port 8765 2>&1"
    }
    $viteJob = Start-Job -Name spren-vite -ScriptBlock {
        Set-Location $using:PWD
        $env:VITE_SPREN_API_URL = 'http://127.0.0.1:8765'
        cmd.exe /c "pnpm --filter @marsys/spren-web dev 2>&1"
    }
    try {
        while (($sidecarJob.State -eq 'Running') -and ($viteJob.State -eq 'Running')) {
            Receive-Job $sidecarJob | ForEach-Object { Write-Host "[sidecar] $_" }
            Receive-Job $viteJob    | ForEach-Object { Write-Host "[vite] $_" }
            Start-Sleep -Milliseconds 200
        }
    } finally {
        Stop-Job    $sidecarJob, $viteJob -ErrorAction SilentlyContinue
        Receive-Job $sidecarJob, $viteJob -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "[shutdown] $_" }
        Remove-Job  $sidecarJob, $viteJob -ErrorAction SilentlyContinue
    }

# Run dev-desktop: Tauri shell launches sidecar internally; Vite still runs separately for HMR
[unix]
dev-desktop:
    #!/usr/bin/env bash
    set -euo pipefail
    (VITE_SPREN_API_URL=http://127.0.0.1:8765 pnpm --filter @marsys/spren-web dev 2>&1 | sed 's/^/[vite] /') &
    VITE_PID=$!
    trap "kill $VITE_PID 2>/dev/null || true" EXIT INT TERM
    sleep 2
    cd apps/desktop && cargo tauri dev

# Run dev-desktop: Tauri shell launches sidecar internally; Vite still runs separately for HMR
[windows]
[script]
dev-desktop:
    $ErrorActionPreference = 'Stop'
    $viteJob = Start-Job -Name spren-vite -ScriptBlock {
        Set-Location $using:PWD
        $env:VITE_SPREN_API_URL = 'http://127.0.0.1:8765'
        cmd.exe /c "pnpm --filter @marsys/spren-web dev 2>&1"
    }
    try {
        $deadline = (Get-Date).AddSeconds(2)
        while ((Get-Date) -lt $deadline) {
            Receive-Job $viteJob | ForEach-Object { Write-Host "[vite] $_" }
            Start-Sleep -Milliseconds 200
        }
        Push-Location apps/desktop
        try {
            cargo tauri dev
        } finally {
            Pop-Location
        }
    } finally {
        Stop-Job    $viteJob -ErrorAction SilentlyContinue
        Receive-Job $viteJob -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "[vite] $_" }
        Remove-Job  $viteJob -ErrorAction SilentlyContinue
    }

# Run all tests
test:
    uv run --package marsys pytest packages/framework/tests --tb=short
    uv run --package spren pytest packages/spren/tests --tb=short
    uv run --package spren-tui pytest apps/tui/tests --tb=short
    pnpm --filter '@marsys/spren-web' test --run
    cargo test --manifest-path apps/desktop/Cargo.toml

# Build: Vite production bundle copied into spren/_webui/
[unix]
build:
    pnpm --filter @marsys/spren-web build
    rm -rf packages/spren/src/spren/_webui
    cp -r apps/web/dist packages/spren/src/spren/_webui

# Build: Vite production bundle copied into spren/_webui/
[windows]
[script]
build:
    $ErrorActionPreference = 'Stop'
    pnpm --filter '@marsys/spren-web' build
    $dst = 'packages/spren/src/spren/_webui'
    if (Test-Path $dst) { Remove-Item -Recurse -Force $dst }
    Copy-Item -Recurse 'apps/web/dist' $dst

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

# Clean
[windows]
[script]
clean:
    $ErrorActionPreference = 'Continue'
    foreach ($p in @('apps/web/dist','apps/web/node_modules','packages/spren/src/spren/_webui')) {
        if (Test-Path $p) { Remove-Item -Recurse -Force $p -ErrorAction SilentlyContinue }
    }
    Get-ChildItem packages -Directory -ErrorAction SilentlyContinue | ForEach-Object {
        foreach ($sub in @('dist','build')) {
            $p = Join-Path $_.FullName $sub
            if (Test-Path $p) { Remove-Item -Recurse -Force $p -ErrorAction SilentlyContinue }
        }
        Get-ChildItem $_.FullName -Filter '*.egg-info' -Directory -ErrorAction SilentlyContinue |
            ForEach-Object { Remove-Item -Recurse -Force $_.FullName -ErrorAction SilentlyContinue }
    }
    Get-ChildItem . -Directory -Recurse -Filter __pycache__ -ErrorAction SilentlyContinue |
        ForEach-Object { Remove-Item -Recurse -Force $_.FullName -ErrorAction SilentlyContinue }
    cargo clean --manifest-path apps/desktop/Cargo.toml
