"""Coordination test fixtures.

When the env var MARSYS_TEST_NEW_ORCHESTRATOR=1 is set, every
ExecutionConfig instance created during the test session has
`use_new_orchestrator` defaulted to True. This lets us run the full
integration suite under the unified-barrier orchestrator without
parametrizing every test file.

Used to verify Phase 3 step 14 / step 15 acceptance:
  MARSYS_TEST_NEW_ORCHESTRATOR=1 pytest tests/coordination/
"""
from __future__ import annotations

import os

import pytest

from marsys.coordination.config import ExecutionConfig


@pytest.fixture(scope="session", autouse=True)
def _force_new_orchestrator_if_env_set():
    """Session-level patch: flip the new-orchestrator flag default."""
    if os.environ.get("MARSYS_TEST_NEW_ORCHESTRATOR") != "1":
        yield
        return

    original_init = ExecutionConfig.__init__

    def patched_init(self, *args, **kwargs):
        kwargs.setdefault("use_new_orchestrator", True)
        original_init(self, *args, **kwargs)

    ExecutionConfig.__init__ = patched_init  # type: ignore[method-assign]
    try:
        yield
    finally:
        ExecutionConfig.__init__ = original_init  # type: ignore[method-assign]
