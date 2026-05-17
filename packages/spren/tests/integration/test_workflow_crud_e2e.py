"""End-to-end workflow CRUD against a real SQLite file."""
from __future__ import annotations

import io
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from spren.auth import generate_token
from spren.server import create_app


FIXTURES = Path(__file__).parent.parent / "fixtures" / "python_workflows"


@pytest.fixture
def real_app(tmp_path: Path):
    token = generate_token()
    app = create_app(
        token=token,
        port=8765,
        data_dir=tmp_path,
        started_at=datetime(2026, 5, 4, tzinfo=timezone.utc),
    )
    with TestClient(app) as client:
        yield {"client": client, "token": token, "data_dir": tmp_path}


def _complex_definition() -> dict:
    return {
        "topology": {
            "nodes": [
                {"name": "Researcher", "kind": "agent", "agent_ref": "agent_1", "is_convergence_point": False, "metadata": {}},
                {"name": "Writer", "kind": "agent", "agent_ref": "agent_2", "is_convergence_point": True, "metadata": {}},
            ],
            "edges": [
                {"source": "Researcher", "target": "Writer", "edge_type": "invoke", "bidirectional": False, "pattern": None, "metadata": {}},
            ],
            "rules": ["never_invoke_self"],
        },
        "agents": {
            "agent_1": {
                "agent_model": {"type": "api", "name": "gpt-4o", "provider": "openai", "max_tokens": 4096, "temperature": 0.5},
                "name": "Researcher",
                "goal": "research",
                "instruction": "find sources",
                "tools": ["search_web"],
                "memory_retention": "session",
                "allowed_peers": ["Writer"],
            },
            "agent_2": {
                "agent_model": {"type": "api", "name": "claude-opus-4-7", "provider": "anthropic", "max_tokens": 8192},
                "name": "Writer",
                "goal": "write",
                "instruction": "polish output",
                "tools": ["write_doc"],
                "memory_retention": "session",
                "allowed_peers": [],
            },
        },
        "execution_config": {
            "convergence_timeout": 60.0,
            "response_format": "json",
            "user_interaction": "none",
        },
    }


def test_round_trip_complex_workflow(real_app):
    client, token = real_app["client"], real_app["token"]
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"name": "complex", "definition": _complex_definition(), "provenance": "api"}

    r = client.post("/v1/workflows", json=payload, headers=headers)
    assert r.status_code == 201, r.text
    wf_id = r.json()["id"]

    r = client.get(f"/v1/workflows/{wf_id}", headers=headers)
    assert r.status_code == 200
    fetched = r.json()
    assert fetched["definition"]["topology"]["rules"] == ["never_invoke_self"]
    assert fetched["definition"]["agents"]["agent_1"]["agent_model"]["temperature"] == 0.5
    assert fetched["definition"]["agents"]["agent_2"]["agent_model"]["provider"] == "anthropic"


def test_round_trip_python_file_import(real_app):
    client, token = real_app["client"], real_app["token"]
    headers = {"Authorization": f"Bearer {token}"}

    src = (FIXTURES / "valid_minimal.py").read_bytes()
    files = {"file": ("valid_minimal.py", io.BytesIO(src), "text/x-python")}
    r = client.post("/v1/workflows/import-python", files=files, headers=headers)
    assert r.status_code == 201, r.text
    envelope = r.json()
    assert envelope["warnings"] == []
    imported = envelope["workflow"]

    # GET it back via the read endpoint
    r2 = client.get(f"/v1/workflows/{imported['id']}", headers=headers)
    fetched = r2.json()
    assert fetched["provenance"] == "code_import"
    assert set(fetched["definition"]["agents"].keys()) == {"Researcher", "Writer"}
    edges = fetched["definition"]["topology"]["edges"]
    assert len(edges) == 1
    assert edges[0]["source"] == "Researcher"
    assert edges[0]["target"] == "Writer"


def test_openapi_schema_reachable(real_app):
    client = real_app["client"]
    r = client.get("/openapi.json")
    assert r.status_code == 200
    spec = r.json()
    paths = spec["paths"]
    assert "/v1/workflows" in paths
    assert "/v1/workflows/{workflow_id}" in paths
    assert "/v1/workflows/import-python" in paths
    schemas = spec["components"]["schemas"]
    for required in ["Workflow", "WorkflowDefinition", "AgentSpec", "ModelConfigSpec", "ErrorEnvelope"]:
        assert required in schemas


def test_idempotency_cache_survives_daemon_restart(tmp_path: Path):
    """a same-key replay still returns the cached body after the app
    is torn down and re-created against the same on-disk SQLite."""
    token = generate_token()
    headers = {"Authorization": f"Bearer {token}", "Idempotency-Key": "across-restart"}
    payload = {"name": "persisted", "definition": _complex_definition(), "provenance": "api"}

    app1 = create_app(token=token, port=0, data_dir=tmp_path, started_at=datetime(2026, 5, 4, tzinfo=timezone.utc))
    with TestClient(app1) as client:
        r1 = client.post("/v1/workflows", json=payload, headers=headers)
        assert r1.status_code == 201
        first_id = r1.json()["id"]

    app2 = create_app(token=token, port=0, data_dir=tmp_path, started_at=datetime(2026, 5, 4, tzinfo=timezone.utc))
    with TestClient(app2) as client:
        r2 = client.post("/v1/workflows", json=payload, headers=headers)
        assert r2.status_code == 201
        # Same cached body replayed → same id, no second row created.
        assert r2.json()["id"] == first_id
        listed = client.get("/v1/workflows", headers={"Authorization": f"Bearer {token}"}).json()
        assert len(listed["items"]) == 1


def test_persistence_across_app_lifecycle(tmp_path: Path):
    """Create with one TestClient, read with another — both backed by the same on-disk SQLite."""
    token = generate_token()

    app1 = create_app(
        token=token,
        port=8765,
        data_dir=tmp_path,
        started_at=datetime(2026, 5, 4, tzinfo=timezone.utc),
    )
    with TestClient(app1) as client:
        r = client.post(
            "/v1/workflows",
            json={
                "name": "persisted",
                "definition": _complex_definition(),
                "provenance": "api",
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        wf_id = r.json()["id"]

    # Verify the row reached SQLite directly.
    db_path = tmp_path / "data" / "spren.db"
    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT name, provenance FROM workflows WHERE id = ?", (wf_id,)).fetchone()
    conn.close()
    assert row == ("persisted", "api")

    # Bring up a second TestClient against the same data dir; the workflow is still there.
    app2 = create_app(
        token=token,
        port=8765,
        data_dir=tmp_path,
        started_at=datetime(2026, 5, 4, tzinfo=timezone.utc),
    )
    with TestClient(app2) as client:
        r = client.get(f"/v1/workflows/{wf_id}", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        assert r.json()["name"] == "persisted"
