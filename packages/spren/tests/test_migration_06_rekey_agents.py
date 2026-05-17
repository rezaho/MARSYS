"""Frozen-baseline test for migration 06 (re-key agents by name).

Seeds an id-keyed row with a name-collision, re-applies migration 06 via the
real MigrationsRunner, and asserts the migrated row (a) parses as the
framework canonical WorkflowDefinition and (b) binds EVERY agent node to a
live Agent via pydantic_to_topology — the actual end goal (AC-AGENTKEY-2).
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pytest

from marsys.agents.agents import Agent
from marsys.agents.registry import AgentRegistry
from marsys.coordination.topology.core import NodeKind
from marsys.coordination.topology.serialize import (
    WorkflowDefinition,
    pydantic_to_topology,
)
from marsys.environment.tools import AVAILABLE_TOOLS
from spren.storage import Database, MigrationsRunner

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _agent(name: str) -> dict:
    return {
        "agent_model": {"type": "api", "name": "gpt-4o", "provider": "openai"},
        "name": name,
        "goal": "g",
        "instruction": "i",
    }


# id-keyed canvas row, post-05 (kind) shape, with a NAME COLLISION (both "Bot").
_IDKEYED = {
    "topology": {
        "nodes": [
            {"name": "Start", "kind": "start"},
            {"name": "N1", "kind": "agent", "agent_ref": "agent_aaa"},
            {"name": "N2", "kind": "agent", "agent_ref": "agent_bbb"},
            {"name": "End", "kind": "end"},
        ],
        "edges": [
            {"source": "Start", "target": "N1"},
            {"source": "N1", "target": "N2"},
            {"source": "N2", "target": "End"},
        ],
        "rules": [],
    },
    "agents": {
        "agent_aaa": _agent("Bot"),
        "agent_bbb": _agent("Bot"),  # collision -> Bot_2
    },
    "execution_config": {},
}


def _seed(conn, wf_id: str, definition: dict) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO workflows (id, name, description, definition, "
        "definition_version, provenance, provenance_metadata, is_archived, "
        "created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (wf_id, wf_id, None, json.dumps(definition), 1, "api", None, 0, now, now),
    )


def test_migration_06_rekey_agents_binds(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    AgentRegistry.clear()
    db = Database(tmp_path)
    conn = db.connection
    MigrationsRunner(conn).run()

    _seed(conn, "wf_idkeyed", _IDKEYED)
    conn.commit()

    conn.execute("DELETE FROM _migrations WHERE id = '06'")
    conn.commit()
    assert MigrationsRunner(conn).run() == ["06"]

    row = conn.execute(
        "SELECT definition, definition_version FROM workflows WHERE id = ?",
        ("wf_idkeyed",),
    ).fetchone()
    doc = json.loads(row["definition"])

    # agents re-keyed by name; collision deduped deterministically.
    assert set(doc["agents"].keys()) == {"Bot", "Bot_2"}
    assert doc["agents"]["Bot"]["name"] == "Bot"
    assert doc["agents"]["Bot_2"]["name"] == "Bot_2"
    refs = {n["name"]: n.get("agent_ref") for n in doc["topology"]["nodes"]}
    assert refs["N1"] == "Bot"
    assert refs["N2"] == "Bot_2"
    # nothing dropped; version bumped.
    assert len(doc["topology"]["nodes"]) == 4
    assert len(doc["topology"]["edges"]) == 3
    assert row["definition_version"] == 2

    # Parses as the framework canonical type AND binds every agent node.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        wd = WorkflowDefinition.model_validate_json(json.dumps(doc))
    topo = pydantic_to_topology(wd, AVAILABLE_TOOLS, handler_registry={})
    agent_nodes = [n for n in topo.nodes if n.kind is NodeKind.AGENT]
    assert len(agent_nodes) == 2
    for n in agent_nodes:
        assert isinstance(n.agent_ref, Agent), f"{n.name} agent not bound"
    assert {n.agent_ref.name for n in agent_nodes} == {"Bot", "Bot_2"}
    AgentRegistry.clear()
