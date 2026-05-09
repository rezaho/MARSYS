"""Top-level pytest fixtures shared across the framework test suite.

Conventions:
- Tests that hit a real cheap paid model are marked ``@pytest.mark.cheap``
  and depend on the :func:`cheap_model` fixture below. They run in the
  CI smoke job (``just test-cheap``) which is gated on
  ``ANTHROPIC_API_KEY``.
- Tests that don't need a real model use ``MagicMock`` or
  ``mock-model``-style ``ModelConfig``; nothing in here should be
  injected into them.

Why "cheap" and not "free":
- Free tiers (Gemini direct, OpenRouter ``:free``) are too rate-limit
  flaky for CI to depend on. Paying ~$0.05/PR for Claude Haiku 4.5 is
  cheap enough to ignore and reliable enough for a green-or-red signal.
- See ``docs/api/configuration.md`` (ErrorHandlingConfig) and the Phase 2
  section of the streaming-tracing follow-up plan.
"""

from __future__ import annotations

import os

import pytest

from marsys.models.models import ModelConfig


@pytest.fixture
def cheap_model() -> ModelConfig:
    """A small paid model suitable for CI smoke tests.

    Default: Claude Haiku 4.5 via Anthropic direct (cheapest Claude, strong
    tool-calling, no OpenRouter markup). Skips the test if
    ``ANTHROPIC_API_KEY`` is not configured.

    Override with ``MARSYS_CHEAP_MODEL`` /``MARSYS_CHEAP_PROVIDER`` env vars
    to swap providers without touching code (e.g. for local debugging
    against a different cheap model).
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set; skipping cheap-tier test")

    return ModelConfig(
        type="api",
        name=os.environ.get("MARSYS_CHEAP_MODEL", "claude-haiku-4-5-20251001"),
        provider=os.environ.get("MARSYS_CHEAP_PROVIDER", "anthropic"),
        max_tokens=1024,
        temperature=0.0,
    )
