"""Generate ``docs/architecture/framework/aggui-custom-events.md`` from the
Pydantic models in ``coordination.aggui.custom_events``.

Single source of truth: ``CUSTOM_EVENT_REGISTRY``. The generated markdown is
checked into the repo and validated against re-generation in CI by
``tests/coordination/aggui/test_doc_generation.py``.

Run from the repo root:

.. code-block:: bash

    python packages/framework/scripts/generate_aggui_custom_events_doc.py

Or directly with the package on PYTHONPATH:

.. code-block:: bash

    python -m scripts.generate_aggui_custom_events_doc
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DOC_PATH = REPO_ROOT / "docs" / "architecture" / "framework" / "aggui-custom-events.md"

# Ensure package is importable when run as a script
sys.path.insert(0, str(REPO_ROOT / "packages" / "framework" / "src"))


def render() -> str:
    from marsys.coordination.aggui.custom_events import CUSTOM_EVENT_REGISTRY

    lines: list[str] = []
    lines.append("# AG-UI `Custom` Events — `marsys.*`")
    lines.append("")
    lines.append(
        "**AUTO-GENERATED — DO NOT EDIT BY HAND.** "
        "Regenerate via "
        "`python packages/framework/scripts/generate_aggui_custom_events_doc.py`. "
        "Source: `packages/framework/src/marsys/coordination/aggui/custom_events.py`."
    )
    lines.append("")
    lines.append(
        "AG-UI's `Custom` event is the documented escape hatch for protocol-internal "
        "events that don't fit the standard lifecycle. MARSYS uses it for "
        "framework-specific lifecycle signals (branch / parallel-group / convergence / "
        "error / resource-limit / user-interaction / memory-compaction) and for the "
        "stream-level handshake."
    )
    lines.append("")
    lines.append(
        "Every event below is validated at emission time against the Pydantic model "
        "in `coordination/aggui/custom_events.py`. Validation failure raises — "
        "catches schema drift at the source. Consumers that want lenient parsing "
        "wrap the iterator in `try/except` at their boundary."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    for name in sorted(CUSTOM_EVENT_REGISTRY.keys()):
        model = CUSTOM_EVENT_REGISTRY[name]
        schema = model.model_json_schema()
        lines.append(f"## `{name}`")
        lines.append("")
        if model.__doc__:
            doc = " ".join(line.strip() for line in model.__doc__.splitlines() if line.strip())
            lines.append(doc)
            lines.append("")
        lines.append("### JSON Schema")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(schema, indent=2, sort_keys=True))
        lines.append("```")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    content = render()
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text(content, encoding="utf-8")
    print(f"Wrote {DOC_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
