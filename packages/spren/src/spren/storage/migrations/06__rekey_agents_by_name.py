"""Re-key stored ``agents`` by ``AgentSpec.name`` and rewrite ``agent_ref``.

Forward-only, one-shot (SP-006). The framework's ``pydantic_to_topology``
binds a node's ``agent_ref`` against ``AgentSpec.name`` (serialize.py:503,509),
not against the ``agents`` dict key. The Spren canvas historically keyed
``agents`` by a random ``agent_<rand>`` id (== ``agent_ref``), so a stored
canvas workflow would hydrate with *unbound* agents post-ADR-008. This
migration, per definition:

  - builds an ``old_key -> final_name`` map from each agent spec's ``name``
    (falling back to the old key when ``name`` is missing/blank)
  - dedupes name collisions deterministically (iteration order; second
    ``Bot`` -> ``Bot_2``, etc.), and sets each spec's own ``name`` to its
    final (deduped) value so ``key == spec.name`` (the canonical invariant)
  - re-keys the ``agents`` dict by the final name
  - rewrites every node's ``agent_ref`` from the old id to the final name
  - drops no node, edge, or agent; bumps ``definition_version``

Idempotent by construction: a definition already keyed by name with no
collisions and ``agent_ref == name`` produces no change (skipped). Runs
AFTER ``05`` (numeric order) so nodes already use ``kind``. Frozen-artifact
discipline: imports NO ``spren.models`` symbol — pure ``json`` dict
transform. The runner wraps this in a transaction; do NOT manage one here.
"""
from __future__ import annotations

import json
import sqlite3


def _rekey_definition(raw: str) -> str | None:
    """Return the rewritten JSON string, or ``None`` if nothing changed."""
    doc = json.loads(raw)
    agents = doc.get("agents")
    topology = doc.get("topology")
    if not isinstance(agents, dict) or not isinstance(topology, dict):
        return None

    old_to_final: dict[str, str] = {}
    used: set[str] = set()
    new_agents: dict[str, object] = {}
    changed = False

    for old_key, spec in agents.items():
        base = old_key
        if isinstance(spec, dict):
            name = spec.get("name")
            if isinstance(name, str) and name.strip():
                base = name
        candidate = base
        suffix = 2
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        used.add(candidate)
        old_to_final[old_key] = candidate
        if candidate != old_key:
            changed = True
        if isinstance(spec, dict) and spec.get("name") != candidate:
            spec = {**spec, "name": candidate}
            changed = True
        new_agents[candidate] = spec

    nodes = topology.get("nodes")
    if isinstance(nodes, list):
        for node in nodes:
            if not isinstance(node, dict):
                continue
            ref = node.get("agent_ref")
            if isinstance(ref, str) and ref in old_to_final:
                final = old_to_final[ref]
                if final != ref:
                    node["agent_ref"] = final
                    changed = True

    if not changed:
        return None
    doc["agents"] = new_agents
    return json.dumps(doc)


def upgrade(conn: sqlite3.Connection) -> None:
    rows = conn.execute("SELECT id, definition FROM workflows").fetchall()
    for row_id, definition in rows:
        rewritten = _rekey_definition(definition)
        if rewritten is not None:
            conn.execute(
                "UPDATE workflows "
                "SET definition = ?, definition_version = definition_version + 1 "
                "WHERE id = ?",
                (rewritten, row_id),
            )
