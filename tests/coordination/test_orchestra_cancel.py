"""Regression: Orchestra.execute() must propagate asyncio.CancelledError
cleanly when the body is interrupted by cancellation.

Background
----------
Before the fix at orchestra.py:993 (pre-bind ``result``) + the explicit
``except asyncio.CancelledError: raise`` clause, the following sequence
silently corrupted the cancel contract:

1. ``Orchestra.execute(...)`` body raises ``asyncio.CancelledError`` —
   typically because the awaiting outer task was cancelled and the
   exception lands on the ``await orchestrator.run(...)`` at line ~1117.
2. The broad ``except Exception as e`` at line ~1181 does NOT catch it
   (``CancelledError`` is ``BaseException`` in Python 3.8+).
3. Neither the success-path ``result = OrchestraResult(...)`` (~line
   1141) nor the except-path assignment (~line 1184) runs, so the
   local ``result`` is unbound.
4. The ``finally`` block at ~line 1193 still runs (Python semantics).
   At line ~1222 ``if result is not None:`` reads the unbound local
   and raises ``UnboundLocalError``.
5. The ``UnboundLocalError`` MASKS the original ``CancelledError`` —
   callers that have ``except asyncio.CancelledError`` (notably a
   downstream daemon's run lifecycle) are silently routed around their
   handler and see the bug string instead.

A downstream consumer's own lifecycle test that uses a stub orchestra
with no ``try/finally`` cannot catch this class of regression. This test
fills that gap at the framework level.

Design
------
The bug is independent of *where* in the try-body the
``CancelledError`` arises — any path that exits the try without
binding ``result`` hits it. The simplest reproduction patches
``Orchestra._ensure_topology`` (the very first call inside the try
at line ~1000) to raise ``CancelledError`` synchronously. This
short-circuits the topology + agent + runtime setup machinery while
still driving the real ``execute()`` body, the real ``except Exception``,
and the real ``finally``. If the bug regresses, this test fails with
``UnboundLocalError`` instead of ``CancelledError``.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from marsys.agents.registry import AgentRegistry
from marsys.coordination.orchestra import Orchestra


class _NullAgentRegistry:
    """Minimal registry that satisfies ``Orchestra.__init__``. We never
    reach the dispatch loop in these tests, so ``get`` is never called.
    """

    @staticmethod
    def get(name: str) -> Any:
        return None

    @staticmethod
    def clear() -> None:
        pass


class _StubTraceCollector:
    """Minimal trace_collector that exercises the bug-prone branch.

    The bug at ``orchestra.py:1222`` (``if result is not None:``) is
    gated on ``self.trace_collector`` being truthy (line ~1201). A
    fresh ``Orchestra()`` has ``trace_collector = None``, which skips
    the entire trace block and incidentally bypasses the unbound-
    ``result`` read. To make the test actually exercise the
    regression, we install this stub.

    ``finalize`` and ``close`` are no-op coroutines so the finally's
    legitimate awaits succeed and the test isolates the
    ``UnboundLocalError`` from ``result``.
    """

    active_traces: dict = {}

    async def finalize(self, session_id):  # noqa: ARG002
        return None

    async def close(self):
        return None


@pytest.fixture(autouse=True)
def clean_registry():
    AgentRegistry.clear()
    yield
    AgentRegistry.clear()


@pytest.mark.asyncio
async def test_execute_propagates_cancelled_error_through_finally(monkeypatch):
    """A ``CancelledError`` raised inside ``Orchestra.execute()``'s
    try-body must propagate to the caller — not be replaced by an
    ``UnboundLocalError`` from the finally's ``result`` read.

    Patches ``_ensure_topology`` to raise ``asyncio.CancelledError``
    synchronously: this is the very first call inside the try at
    line ~1000 of orchestra.py, before either ``result =`` site runs.
    The finally still executes and must hit the
    ``if result is not None`` branch with ``result`` properly bound
    (to ``None``) — without the pre-bind it raises ``UnboundLocalError``
    which then MASKS the original ``CancelledError``.
    """
    orchestra = Orchestra(agent_registry=_NullAgentRegistry)
    # The bug-prone branch is gated on ``self.trace_collector`` being
    # truthy (orchestra.py:~1201). Install a stub so the finally
    # actually reaches ``if result is not None`` at line ~1222.
    orchestra.trace_collector = _StubTraceCollector()  # type: ignore[assignment]

    def _cancel_during_setup(self, topology):  # noqa: ARG001
        raise asyncio.CancelledError()

    monkeypatch.setattr(
        Orchestra, "_ensure_topology", _cancel_during_setup, raising=True,
    )

    # Contract: CancelledError propagates, not UnboundLocalError.
    # The pre-fix behaviour was: finally tries to read ``result`` ->
    # NameError(UnboundLocalError) -> replaces CancelledError as the
    # in-flight exception -> caller sees UnboundLocalError.
    with pytest.raises(asyncio.CancelledError):
        await orchestra.execute(
            task="anything",
            topology={"agents": [], "flows": []},
            context={"session_id": "test-cancel-1"},
        )


@pytest.mark.asyncio
async def test_execute_finally_runs_when_cancelled(monkeypatch):
    """The ``finally`` at line ~1193 must run cleanly on the cancel
    path. ``_active_orchestrators`` is populated at line ~1077 (before
    the patched setup raises), so a working finally pops it back out;
    a broken finally leaves the entry behind or raises a different
    exception.

    Different invariant than the first test: that one asserts the
    surfaced exception type; this one asserts the cleanup behaviour
    of the finally itself.
    """
    orchestra = Orchestra(agent_registry=_NullAgentRegistry)
    orchestra.trace_collector = _StubTraceCollector()  # type: ignore[assignment]
    session_id = "test-cancel-finally"

    # Pre-seed the active-orchestrators map so we can observe the
    # finally's pop unambiguously even though our patched setup
    # short-circuits before line ~1077 would have done it organically.
    orchestra._active_orchestrators[session_id] = object()

    def _cancel_during_setup(self, topology):  # noqa: ARG001
        raise asyncio.CancelledError()

    monkeypatch.setattr(
        Orchestra, "_ensure_topology", _cancel_during_setup, raising=True,
    )

    with pytest.raises(asyncio.CancelledError):
        await orchestra.execute(
            task="anything",
            topology={"agents": [], "flows": []},
            context={"session_id": session_id},
        )

    assert session_id not in orchestra._active_orchestrators, (
        "finally did not pop session_id from _active_orchestrators — "
        "either skipped or raised inside the finally"
    )
