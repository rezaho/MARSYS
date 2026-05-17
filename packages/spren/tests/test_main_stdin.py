"""End-to-end tests for the sidecar's stdin-shutdown protocol.

Spawns `python -m spren --port 0` as a real subprocess and asserts the
contract: an explicit `shutdown\\n` line shuts it down; a closed / EOF /
DEVNULL stdin must NOT (a launcher that gives the child no usable stdin —
`just dev`'s PowerShell `Start-Job` + `cmd /c`, CI's DEVNULL — surfaces an
immediate EOF that previously killed the sidecar right after startup).
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

READY_LINE_RE = re.compile(r"^spren-ready: port=(\d+) token=(\S+) data-dir=(.+)$")


def _spawn_sidecar(
    tmp_path: Path, *, stdin=subprocess.PIPE
) -> subprocess.Popen[str]:
    """Launch `python -m spren --port 0 --data-dir <tmp>` and wait for ready."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "spren", "--port", "0", "--data-dir", str(tmp_path)],
        stdin=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    ready_line = proc.stdout.readline()
    if not ready_line or READY_LINE_RE.match(ready_line.strip()) is None:
        proc.kill()
        out, err = proc.communicate(timeout=5)
        pytest.fail(
            f"sidecar did not emit a ready line. stdout={ready_line!r}{out!r}, stderr={err!r}"
        )
    return proc


def _terminate(proc: subprocess.Popen[str]) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


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


def test_stdin_devnull_does_not_terminate_sidecar(tmp_path: Path) -> None:
    """Regression (just dev / CI): a launcher giving the child DEVNULL
    stdin surfaces immediate EOF. The sidecar must STAY UP — only an
    explicit `shutdown` line shuts it down. (The old contract killed it
    here right after `spren-ready`, collapsing `just dev`.)"""
    proc = _spawn_sidecar(tmp_path, stdin=subprocess.DEVNULL)
    try:
        # Old bug: exited within ~1s of startup. It must still be running.
        with pytest.raises(subprocess.TimeoutExpired):
            proc.wait(timeout=4)
        assert proc.poll() is None, "sidecar exited on DEVNULL stdin EOF"
    finally:
        _terminate(proc)


def test_stdin_close_does_not_terminate_sidecar(tmp_path: Path) -> None:
    """A managing parent that opens then closes stdin (EOF) without
    sending `shutdown` must NOT take the sidecar down."""
    proc = _spawn_sidecar(tmp_path)
    assert proc.stdin is not None
    proc.stdin.close()  # surfaces EOF on the sidecar's reader
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            proc.wait(timeout=4)
        assert proc.poll() is None, "sidecar exited on stdin close (EOF)"
    finally:
        _terminate(proc)


def test_repeated_shutdown_is_idempotent(tmp_path: Path) -> None:
    proc = _spawn_sidecar(tmp_path)
    assert proc.stdin is not None
    proc.stdin.write("shutdown\n")
    proc.stdin.write("shutdown\n")
    proc.stdin.flush()

    rc = proc.wait(timeout=5)
    assert rc == 0
