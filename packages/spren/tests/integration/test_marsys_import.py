"""Workspace integration: the marsys framework is importable from the spren package."""
from __future__ import annotations


def test_orchestra_importable() -> None:
    from marsys.coordination import Orchestra  # noqa: F401


def test_agent_importable() -> None:
    from marsys.agents import Agent  # noqa: F401


def test_marsys_version_available() -> None:
    import marsys

    assert hasattr(marsys, "__version__") or hasattr(marsys, "__file__")
