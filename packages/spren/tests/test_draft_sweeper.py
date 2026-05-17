"""Tests for the draft sweeper + ``include_drafts`` list filter.

Acceptance criteria AC-22 .. AC-30. Empty visual-builder drafts are hidden
from the default list, included with ``include_drafts=true``, and deleted by
the sweeper after 24 hours.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from spren.models import TaskInput
from spren.storage import Database
from spren.storage.runs import insert_run
from spren.storage.workflows import delete_empty_drafts_older_than
from spren.workers.draft_sweeper import (
    DRAFT_MAX_AGE,
    sweep_empty_drafts_once,
)


EMPTY_DEFINITION = {
    "topology": {"nodes": [], "edges": [], "rules": []},
    "agents": {},
    "execution_config": {},
}


def test_default_list_excludes_empty_visual_builder_drafts(client, auth_headers) -> None:
    response = client.post(
        "/v1/workflows",
        headers=auth_headers,
        json={
            "name": "Untitled workflow",
            "definition": EMPTY_DEFINITION,
            "provenance": "visual_builder",
        },
    )
    assert response.status_code == 201, response.text

    listed = client.get("/v1/workflows", headers=auth_headers).json()
    assert listed["items"] == []
    assert listed["has_more"] is False


def test_include_drafts_shows_empty_visual_builder_drafts(client, auth_headers) -> None:
    client.post(
        "/v1/workflows",
        headers=auth_headers,
        json={
            "name": "Untitled workflow",
            "definition": EMPTY_DEFINITION,
            "provenance": "visual_builder",
        },
    )
    response = client.get("/v1/workflows?include_drafts=true", headers=auth_headers)
    body = response.json()
    assert len(body["items"]) == 1
    assert body["items"][0]["name"] == "Untitled workflow"


def test_non_draft_workflow_visible_without_include_drafts(client, auth_headers, sample_definition) -> None:
    """A workflow with non-empty topology is never a draft, regardless of provenance."""
    client.post(
        "/v1/workflows",
        headers=auth_headers,
        json={
            "name": "Real workflow",
            "definition": sample_definition,
            "provenance": "visual_builder",
        },
    )
    body = client.get("/v1/workflows", headers=auth_headers).json()
    assert len(body["items"]) == 1
    assert body["items"][0]["name"] == "Real workflow"


def test_api_provenance_with_empty_topology_is_not_a_draft(client, auth_headers) -> None:
    """Only `visual_builder` provenance + empty nodes counts as draft."""
    client.post(
        "/v1/workflows",
        headers=auth_headers,
        json={
            "name": "API empty",
            "definition": EMPTY_DEFINITION,
            "provenance": "api",
        },
    )
    body = client.get("/v1/workflows", headers=auth_headers).json()
    assert len(body["items"]) == 1


def test_sweeper_deletes_old_empty_drafts_only(client, auth_headers, sample_definition, data_dir) -> None:
    # Three workflows: empty draft (24h+1m old), empty draft (just created),
    # populated visual_builder (24h+1m old).
    old_ts = (datetime.now(timezone.utc) - DRAFT_MAX_AGE - timedelta(minutes=1)).isoformat()

    for payload, name in [
        ({"name": "old-empty-draft", "definition": EMPTY_DEFINITION, "provenance": "visual_builder"}, "old-empty"),
        ({"name": "fresh-empty-draft", "definition": EMPTY_DEFINITION, "provenance": "visual_builder"}, "fresh-empty"),
        ({"name": "old-real", "definition": sample_definition, "provenance": "visual_builder"}, "old-real"),
    ]:
        client.post("/v1/workflows", headers=auth_headers, json=payload)

    # Backdate the old rows via direct SQL — no API surface mutates `updated_at`
    # except a real save.
    db = Database(data_dir)
    db.connection.execute(
        "UPDATE workflows SET updated_at = ? WHERE name IN ('old-empty-draft', 'old-real')",
        (old_ts,),
    )

    deleted = sweep_empty_drafts_once(lambda: db.connection)
    assert deleted == 1

    remaining = {
        row[0]
        for row in db.connection.execute("SELECT name FROM workflows").fetchall()
    }
    assert remaining == {"fresh-empty-draft", "old-real"}


def test_sweeper_preserves_non_visual_builder_drafts(data_dir) -> None:
    db = Database(data_dir)
    from spren.storage import MigrationsRunner

    MigrationsRunner(db.connection).run()

    old_ts = (datetime.now(timezone.utc) - DRAFT_MAX_AGE - timedelta(hours=1)).isoformat()
    db.connection.execute(
        """
        INSERT INTO workflows (
            id, name, description, definition, definition_version,
            provenance, provenance_metadata, is_archived, created_at, updated_at
        ) VALUES (?, ?, ?, ?, 1, ?, NULL, 0, ?, ?)
        """,
        ("01TESTSAPI", "api-empty", None, json.dumps(EMPTY_DEFINITION), "api", old_ts, old_ts),
    )
    deleted = delete_empty_drafts_older_than(db.connection, max_age_iso=datetime.now(timezone.utc).isoformat())
    assert deleted == 0


def test_sweeper_preserves_old_empty_draft_with_runs(
    client, auth_headers, data_dir
) -> None:
    """WF-BUG-SWEEPER-1: an old empty visual_builder draft that HAS a run
    must NOT be swept (the FK has no ON DELETE → deleting it raised
    IntegrityError every tick; SP-009 also makes keeping it correct). A
    sibling old empty draft WITHOUT runs is still swept (behaviour kept)."""
    with_run = client.post(
        "/v1/workflows",
        headers=auth_headers,
        json={"name": "old-empty-with-run", "definition": EMPTY_DEFINITION, "provenance": "visual_builder"},
    ).json()
    no_run = client.post(
        "/v1/workflows",
        headers=auth_headers,
        json={"name": "old-empty-no-run", "definition": EMPTY_DEFINITION, "provenance": "visual_builder"},
    ).json()

    db = Database(data_dir)
    insert_run(
        db.connection,
        run_id="01SWEEPRUN0000000000000001",
        workflow_id=with_run["id"],
        task_input=TaskInput(),
    )
    old_ts = (datetime.now(timezone.utc) - DRAFT_MAX_AGE - timedelta(minutes=1)).isoformat()
    db.connection.execute(
        "UPDATE workflows SET updated_at = ? WHERE name IN ('old-empty-with-run', 'old-empty-no-run')",
        (old_ts,),
    )
    db.connection.commit()

    # Must not raise (the original bug caught + logged IntegrityError here).
    deleted = sweep_empty_drafts_once(lambda: db.connection)
    assert deleted == 1

    remaining = {
        row[0] for row in db.connection.execute("SELECT name FROM workflows").fetchall()
    }
    assert remaining == {"old-empty-with-run"}


def test_list_surfaces_empty_draft_with_runs(client, auth_headers, data_dir) -> None:
    """An empty visual_builder draft with run history is not an abandoned
    draft (SP-009) — it must appear in the default list, not be hidden.
    Keeps the list filter and the sweeper predicate in sync."""
    created = client.post(
        "/v1/workflows",
        headers=auth_headers,
        json={"name": "draft-with-history", "definition": EMPTY_DEFINITION, "provenance": "visual_builder"},
    ).json()

    db = Database(data_dir)
    insert_run(
        db.connection,
        run_id="01SWEEPRUN0000000000000002",
        workflow_id=created["id"],
        task_input=TaskInput(),
    )
    db.connection.commit()

    body = client.get("/v1/workflows", headers=auth_headers).json()
    names = [w["name"] for w in body["items"]]
    assert "draft-with-history" in names
