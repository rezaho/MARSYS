"""Doc-drift check: the generated aggui-custom-events.md matches the checked-in file."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("ag_ui")

REPO_ROOT = Path(__file__).resolve().parents[5]
DOC_PATH = REPO_ROOT / "docs" / "architecture" / "framework" / "aggui-custom-events.md"
SCRIPT_PATH = REPO_ROOT / "packages" / "framework" / "scripts" / "generate_aggui_custom_events_doc.py"


def test_doc_matches_pydantic_models():
    """Re-run the generator in-process and compare against the checked-in markdown.

    If this test fails, regenerate the doc:
        python packages/framework/scripts/generate_aggui_custom_events_doc.py
    """
    # Load the script's `render()` function directly so we don't shell out.
    sys.path.insert(0, str(SCRIPT_PATH.parent))
    try:
        import importlib

        # The script's module-level `sys.path.insert` is harmless to re-run.
        spec = importlib.util.spec_from_file_location(
            "_aggui_doc_gen", SCRIPT_PATH
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        generated = mod.render()
    finally:
        sys.path.pop(0)

    assert DOC_PATH.exists(), (
        f"{DOC_PATH} is missing — run the generator:\n"
        f"  python {SCRIPT_PATH}"
    )
    on_disk = DOC_PATH.read_text(encoding="utf-8")
    assert on_disk == generated, (
        "Generated aggui-custom-events.md differs from the checked-in version. "
        "Regenerate:\n"
        f"  python {SCRIPT_PATH}"
    )
