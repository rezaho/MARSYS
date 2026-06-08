set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

default:
    @just --list

# Install dependencies (test extra: pytest + hypothesis)
install:
    uv sync --extra test

# Run the test suite
test:
    uv run pytest

# Lint
lint:
    uv run flake8 src tests

# Format
format:
    uv run black src tests

# Serve the docs locally with live reload
docs:
    uv run mkdocs serve

# Build the static docs site
docs-build:
    uv run mkdocs build

# Build the wheel + sdist
build:
    uv build
