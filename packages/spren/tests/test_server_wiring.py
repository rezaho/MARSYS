"""Smoke tests for the Session 03 server wiring.

Covers AC-25 (sweeper task registered via FastAPI lifespan) and AC-29
(first PUT advances `updated_at` such that the row leaves the empty-
draft predicate and shows on the default list).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from spren.auth import generate_token
from spren.server import create_app


@pytest.fixture
def app_with_sweeper(tmp_path: Path):
    """An app instance where the draft sweeper is enabled.

    The sweeper sleeps `SWEEP_INTERVAL` (4 hours) between passes — far
    longer than the test runs — so its presence is observable but
    benign. ``TestClient`` exercises the lifespan handler synchronously
    via context-manager enter/exit.
    """
    token = generate_token()
    app = create_app(
        token=token,
        port=8765,
        data_dir=tmp_path,
        started_at=datetime(2026, 5, 12, tzinfo=timezone.utc),
        enable_draft_sweeper=True,
    )
    app.state._spren_test_token = token  # surface for the test
    return app


def test_sweeper_task_runs_on_lifespan(app_with_sweeper) -> None:
    """AC-25 — entering the lifespan handler schedules the sweeper task.

    We can't time-skip 4 hours, but we can observe that the lifespan
    handler completes cleanly with the sweeper enabled (the task gets
    cancelled on shutdown and the lifespan exits with no exception).
    """
    token = app_with_sweeper.state._spren_test_token
    with TestClient(app_with_sweeper) as client:
        # The sidecar must respond on healthz immediately after the
        # lifespan startup completes — proves the task didn't deadlock
        # the lifespan or raise during scheduling.
        response = client.get("/healthz")
        assert response.status_code == 200
        # And the new endpoints from Session 03 are reachable.
        response = client.get(
            "/v1/tools",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200


def test_empty_draft_save_advances_past_predicate(client, auth_headers) -> None:
    """AC-29 — a first PUT on an empty draft makes it surface in the default list.

    Create a `visual_builder` workflow with empty topology (the row is a
    draft and hidden from the default list). Save it via PUT with a
    populated topology. The same default list now includes it.
    """
    empty = {
        "topology": {"nodes": [], "edges": [], "rules": []},
        "agents": {},
        "execution_config": {},
    }
    populated = {
        "topology": {
            "nodes": [{"name": "Researcher", "node_type": "agent", "agent_ref": "a1"}],
            "edges": [],
            "rules": [],
        },
        "agents": {
            "a1": {
                "agent_model": {"type": "api", "name": "claude-opus-4-7", "provider": "anthropic"},
                "name": "Researcher",
                "goal": "do research",
                "instruction": "Find sources",
                "tools": [],
                "memory_retention": "session",
                "allowed_peers": [],
            },
        },
        "execution_config": {},
    }
    create = client.post(
        "/v1/workflows",
        headers=auth_headers,
        json={"name": "Untitled workflow", "definition": empty, "provenance": "visual_builder"},
    )
    workflow_id = create.json()["id"]

    # Before save: default list excludes the row.
    listing = client.get("/v1/workflows", headers=auth_headers).json()
    assert listing["items"] == []

    # Save via PUT.
    saved = client.put(
        f"/v1/workflows/{workflow_id}",
        headers=auth_headers,
        json={
            "name": "research-pipeline",
            "definition": populated,
            "provenance": "visual_builder",
        },
    )
    assert saved.status_code == 200

    # After save: the row appears in the default list.
    listing = client.get("/v1/workflows", headers=auth_headers).json()
    assert len(listing["items"]) == 1
    assert listing["items"][0]["id"] == workflow_id
    assert listing["items"][0]["name"] == "research-pipeline"


def test_app_endpoints_includes_new_session_03_routes(client, auth_headers) -> None:
    """The bootstrap surface advertises the new endpoints."""
    response = client.get("/v1/bootstrap", headers=auth_headers)
    assert response.status_code == 200
    body = response.json()
    assert body["endpoints"]["tools"] == "/v1/tools"
    assert body["endpoints"]["lint"] == "/v1/workflows/{id}/lint"
