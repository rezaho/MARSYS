"""Rewrite stored workflow definitions to the post-ADR-008 NodeKind shape.

Forward-only, one-shot (SP-006). The framework canonical ``NodeSpec`` uses
``kind`` (``NodeKind`` = {agent,start,end,user}) and forbids the legacy
``node_type`` key (``extra="forbid"``); a stored ``system``/``tool`` value is
hard-rejected at load. This migration, per node:

  - renames ``node_type`` -> ``kind``
  - maps a legacy ``system``/``tool`` value -> ``agent`` (MANDATORY, not
    defensive: a DB that never ran ``04`` reaches here with ``system``/``tool``
    and the framework would otherwise refuse to load the definition)
  - leaves ``agent``/``start``/``end``/``user`` intact under ``kind``
  - drops no node and no edge (Start/End/User survive as ``kind`` nodes)

``definition_version`` is bumped for every rewritten row. Edges, agents and
execution_config are untouched. Reusing prefix ``04`` would be silently
skipped on any DB that ran the obsolete ``04`` (the runner records applied
ids), so this is ``05``.

Frozen-artifact discipline: imports NO ``spren.models`` symbol — a pure
``json`` dict transform, so a future model change cannot retroactively alter
what this migration did. The runner wraps this in a transaction; do NOT
manage one here.
"""
from __future__ import annotations

import json
import sqlite3

_KIND_REMAP = {"system": "agent", "tool": "agent"}


def _migrate_definition(raw: str) -> str | None:
    """Return the rewritten JSON string, or ``None`` if nothing changed."""
    doc = json.loads(raw)
    topology = doc.get("topology")
    if not isinstance(topology, dict):
        return None
    nodes = topology.get("nodes")
    if not isinstance(nodes, list):
        return None

    changed = False
    for node in nodes:
        if not isinstance(node, dict) or "node_type" not in node:
            continue
        value = node.pop("node_type")
        node["kind"] = _KIND_REMAP.get(value, value)
        changed = True

    return json.dumps(doc) if changed else None


def upgrade(conn: sqlite3.Connection) -> None:
    rows = conn.execute("SELECT id, definition FROM workflows").fetchall()
    for row_id, definition in rows:
        rewritten = _migrate_definition(definition)
        if rewritten is not None:
            conn.execute(
                "UPDATE workflows "
                "SET definition = ?, definition_version = definition_version + 1 "
                "WHERE id = ?",
                (rewritten, row_id),
            )
