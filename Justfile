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

# Run dev: FastAPI sidecar + Vite dev server (no Tauri).
# Parses the sidecar's `spren-ready: port=N token=T` line and prints a
# click-able URL (with the token in the URL fragment, never the query
# string — the fragment is never sent over the wire) so the dev tab can
# open straight to an authenticated state.
[unix]
dev:
    #!/usr/bin/env bash
    set -euo pipefail
    UI_HOST="${SPREN_DEV_UI_HOST:-localhost}"
    UI_PORT="${SPREN_DEV_UI_PORT:-5173}"
    (uv run --package spren python -m spren --port 8765 2>&1 \
        | awk -v host="$UI_HOST" -v port="$UI_PORT" '
            /^spren-ready:/ {
                tok = ""
                for (i = 1; i <= NF; i++) if ($i ~ /^token=/) { tok = substr($i, 7); break }
                printf "\n──────────────────────────────────────────────────────────────────────\n"
                printf " Spren ready. Open in your browser:\n\n"
                printf "   http://%s:%s/#token=%s\n\n", host, port, tok
                printf " (the token lives in the URL fragment so it never crosses the wire)\n"
                printf "──────────────────────────────────────────────────────────────────────\n\n"
                fflush()
            }
            { print "[sidecar] " $0; fflush() }
        ') &
    SIDECAR_PID=$!
    trap "kill $SIDECAR_PID 2>/dev/null || true" EXIT INT TERM
    VITE_SPREN_API_URL=http://127.0.0.1:8765 pnpm --filter @marsys/spren-web dev 2>&1 | sed 's/^/[vite] /'

# Run dev: FastAPI sidecar + Vite dev server (no Tauri).
# Parses the sidecar's `spren-ready: port=N token=T` line and prints a
# click-able URL (with the token in the URL fragment, never the query
# string — the fragment is never sent over the wire) so the dev tab
# can open straight to an authenticated state.
[windows]
[script]
dev:
    $ErrorActionPreference = 'Stop'
    $UiHost = if ($env:SPREN_DEV_UI_HOST) { $env:SPREN_DEV_UI_HOST } else { 'localhost' }
    $UiPort = if ($env:SPREN_DEV_UI_PORT) { $env:SPREN_DEV_UI_PORT } else { '5173' }
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
            Receive-Job $sidecarJob | ForEach-Object {
                $line = $_
                if ($line -match '^spren-ready:.*token=(\S+)') {
                    $tok = $Matches[1]
                    Write-Host ""
                    Write-Host "──────────────────────────────────────────────────────────────────────"
                    Write-Host " Spren ready. Open in your browser:"
                    Write-Host ""
                    Write-Host "   http://${UiHost}:${UiPort}/#token=$tok"
                    Write-Host ""
                    Write-Host " (the token lives in the URL fragment so it never crosses the wire)"
                    Write-Host "──────────────────────────────────────────────────────────────────────"
                    Write-Host ""
                }
                Write-Host "[sidecar] $line"
            }
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

# Run cheap-tier framework tests (paid model, ~$0.05/run). Used by the CI
# smoke job. Requires ANTHROPIC_API_KEY (Claude Haiku 4.5 by default; set
# MARSYS_CHEAP_MODEL / MARSYS_CHEAP_PROVIDER to swap).
test-cheap:
    uv run --package marsys pytest packages/framework/tests -m cheap --tb=short

# Build: Vite production bundle copied into spren/_webui/
[unix]
build:
    pnpm --filter @marsys/spren-web build
    test -f apps/web/dist/index.html || { echo "ERROR: vite build did not produce apps/web/dist/index.html" >&2; exit 1; }
    rm -rf packages/spren/src/spren/_webui
    cp -r apps/web/dist packages/spren/src/spren/_webui

# Build: Vite production bundle copied into spren/_webui/
[windows]
[script]
build:
    $ErrorActionPreference = 'Stop'
    pnpm --filter '@marsys/spren-web' build
    if (-not (Test-Path 'apps/web/dist/index.html')) {
        Write-Error 'ERROR: vite build did not produce apps/web/dist/index.html'
        exit 1
    }
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
