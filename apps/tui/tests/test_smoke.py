"""Smoke tests for the placeholder TUI package."""
from __future__ import annotations


def test_package_importable() -> None:
    import spren_tui

    assert spren_tui.__version__ == "0.3.0"


def test_main_callable() -> None:
    from spren_tui.__main__ import main

    main()
