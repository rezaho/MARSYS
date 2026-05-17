"""Frozen-baseline test for migration 05 (node_type -> kind).

Seeds the three legacy row shapes, re-applies migration 05 via the real
MigrationsRunner, and asserts every row parses as the framework canonical
WorkflowDefinition with the correct `kind` and with NO node/edge dropped
(SP-017-spirit, AC-8/NF-1). The framework's permissive missing-Start
DeprecationWarning is expected for agent-only rows and is not under test.
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pytest

from marsys.coordination.topology.serialize import WorkflowDefinition
from spren.storage import Database, MigrationsRunner

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

_AGENT = {
    "agent_model": {"type": "api", "name": "gpt-4o", "provider": "openai"},
    "name": "R",
    "goal": "g",
    "instruction": "i",
}

# (a) pre-04 row: vestigial system/tool node_type values (never ran 04).
_PRE04 = {
    "topology": {
        "nodes": [
            {"name": "Sys", "node_type": "system"},
            {"name": "R", "node_type": "agent", "agent_ref": "R"},
            {"name": "T", "node_type": "tool"},
        ],
        "edges": [],
        "rules": [],
    },
    "agents": {"R": _AGENT},
    "execution_config": {},
}

# (b) Session-07 row: node_type in {agent,start,end,user}, with det-nodes.
_S07 = {
    "topology": {
        "nodes": [
            {"name": "Start", "node_type": "start"},
            {"name": "R", "node_type": "agent", "agent_ref": "R"},
            {"name": "End", "node_type": "end"},
        ],
        "edges": [
            {"source": "Start", "target": "R"},
            {"source": "R", "target": "End"},
        ],
        "rules": [],
    },
    "agents": {"R": _AGENT},
    "execution_config": {},
}

# (c) already-post-S08 row: uses `kind`, no `node_type` — 05 must skip it.
_POST = {
    "topology": {
        "nodes": [{"name": "R", "kind": "agent", "agent_ref": "R"}],
        "edges": [],
        "rules": [],
    },
    "agents": {"R": _AGENT},
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


def _fetch(conn, wf_id: str) -> tuple[dict, int]:
    row = conn.execute(
        "SELECT definition, definition_version FROM workflows WHERE id = ?",
        (wf_id,),
    ).fetchone()
    return json.loads(row["definition"]), row["definition_version"]


def test_migration_05_node_kind_frozen_baseline(tmp_path: Path):
    db = Database(tmp_path)
    conn = db.connection
    MigrationsRunner(conn).run()  # schema + all migrations (incl. 05)

    _seed(conn, "wf_pre04", _PRE04)
    _seed(conn, "wf_s07", _S07)
    _seed(conn, "wf_post", _POST)
    conn.commit()

    # Re-apply ONLY 05 against the seeded legacy rows via the real runner.
    conn.execute("DELETE FROM _migrations WHERE id = '05'")
    conn.commit()
    applied = MigrationsRunner(conn).run()
    assert applied == ["05"]

    # (a) pre-04: system/tool collapsed to agent; key renamed; nothing dropped.
    doc, ver = _fetch(conn, "wf_pre04")
    nodes = doc["topology"]["nodes"]
    assert len(nodes) == 3
    assert all("node_type" not in n for n in nodes)
    assert {n["name"]: n["kind"] for n in nodes} == {
        "Sys": "agent", "R": "agent", "T": "agent",
    }
    assert ver == 2  # bumped
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        wd = WorkflowDefinition.model_validate_json(json.dumps(doc))
    assert len(wd.topology.nodes) == 3

    # (b) Session-07: start/end/user survive as kind nodes; edges intact.
    doc, ver = _fetch(conn, "wf_s07")
    nodes = doc["topology"]["nodes"]
    assert {n["name"]: n["kind"] for n in nodes} == {
        "Start": "start", "R": "agent", "End": "end",
    }
    assert len(doc["topology"]["edges"]) == 2  # no edge dropped
    assert ver == 2
    wd = WorkflowDefinition.model_validate_json(json.dumps(doc))
    assert {n.name for n in wd.topology.nodes} == {"Start", "R", "End"}

    # (c) post-S08: untouched (no node_type), still parses, version unchanged.
    doc, ver = _fetch(conn, "wf_post")
    assert doc["topology"]["nodes"][0]["kind"] == "agent"
    assert ver == 1  # 05 skipped it
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        WorkflowDefinition.model_validate_json(json.dumps(doc))
