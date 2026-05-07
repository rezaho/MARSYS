"""End-to-end tests for the sidecar's stdin-shutdown protocol.

Spawns `python -m spren --port 0` as a real subprocess, exercises the two
graceful-shutdown paths (explicit `shutdown\\n` line, parent stdin close ->
EOF), and asserts the process exits within a bounded time.
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

READY_LINE_RE = re.compile(r"^spren-ready: port=(\d+) token=(\S+) data-dir=(.+)$")


def _spawn_sidecar(tmp_path: Path) -> subprocess.Popen[str]:
    """Launch `python -m spren --port 0 --data-dir <tmp>` and wait for ready."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "spren", "--port", "0", "--data-dir", str(tmp_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    ready_line = proc.stdout.readline()
    if not ready_line or READY_LINE_RE.match(ready_line.strip()) is None:
        # Drain whatever the process has emitted so far for diagnostics.
        proc.kill()
        out, err = proc.communicate(timeout=5)
        pytest.fail(
            f"sidecar did not emit a ready line. stdout={ready_line!r}{out!r}, stderr={err!r}"
        )
    return proc


def test_shutdown_line_terminates_sidecar(tmp_path: Path) -> None:
    proc = _spawn_sidecar(tmp_path)
    assert proc.stdin is not None
    proc.stdin.write("shutdown\n")
    proc.stdin.flush()

    start = time.monotonic()
    rc = proc.wait(timeout=5)
    elapsed = time.monotonic() - start

    assert rc == 0, f"expected clean exit, got {rc}"
    assert elapsed < 3, f"shutdown took {elapsed:.2f}s, expected <3s"


def test_stdin_eof_terminates_sidecar(tmp_path: Path) -> None:
    proc = _spawn_sidecar(tmp_path)
    assert proc.stdin is not None
    proc.stdin.close()  # closing the pipe surfaces EOF on the sidecar's reader

    rc = proc.wait(timeout=5)
    assert rc == 0, f"expected clean exit on stdin EOF, got {rc}"


def test_repeated_shutdown_is_idempotent(tmp_path: Path) -> None:
    proc = _spawn_sidecar(tmp_path)
    assert proc.stdin is not None
    proc.stdin.write("shutdown\n")
    proc.stdin.write("shutdown\n")
    proc.stdin.flush()

    rc = proc.wait(timeout=5)
    assert rc == 0
