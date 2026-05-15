"""Rewrite stored workflow definitions to the Session-07 node model.

Forward-only, one-shot (SP-006 + ``docs/architecture/spren/11-node-model.md``).
Pre-07 ``$.topology.nodes[*].node_type`` was ``{agent,user,system,tool}``;
post-07 the model is ``{agent,start,end,user}`` (``system``/``tool`` were
vestigial framework enum members with zero execution semantics — P3).

Mapping (per node, in place):
  agent | (absent) -> unchanged (absent means default 'agent')
  user             -> unchanged
  system | tool    -> 'agent'  (vestigial; collapse — no behaviour lost)

No old value maps to start/end (they did not exist pre-07; Start is seeded
by the canvas going forward — not synthesized here, out of this migration's
remit). Edges, agents, and execution_config are untouched.

Frozen-artifact discipline: this file imports NO ``spren.models`` symbol. A
future model change must not retroactively alter what this migration did.
Pure ``json`` dict transform. The runner wraps this in a transaction — do
NOT manage one here.
"""
from __future__ import annotations

import json
import sqlite3

_NODE_TYPE_REMAP = {"system": "agent", "tool": "agent"}


def _remap_definition(raw: str) -> str | None:
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
        if not isinstance(node, dict):
            continue
        nt = node.get("node_type")
        if nt in _NODE_TYPE_REMAP:
            node["node_type"] = _NODE_TYPE_REMAP[nt]
            changed = True

    return json.dumps(doc) if changed else None


def upgrade(conn: sqlite3.Connection) -> None:
    rows = conn.execute("SELECT id, definition FROM workflows").fetchall()
    for row_id, definition in rows:
        rewritten = _remap_definition(definition)
        if rewritten is not None:
            conn.execute(
                "UPDATE workflows SET definition = ? WHERE id = ?",
                (rewritten, row_id),
            )
