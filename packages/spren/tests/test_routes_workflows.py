"""Workflow CRUD endpoint tests."""
from __future__ import annotations

import json

import pytest


def _create(client, headers, sample_definition, *, name="wf", provenance="api"):
    return client.post(
        "/v1/workflows",
        json={"name": name, "definition": sample_definition, "provenance": provenance},
        headers=headers,
    )


def test_post_creates_with_generated_id(client, auth_headers, sample_definition):
    r = _create(client, auth_headers, sample_definition)
    assert r.status_code == 201, r.text
    body = r.json()
    assert len(body["id"]) == 26  # ULID
    assert body["name"] == "wf"
    assert body["provenance"] == "api"
    assert body["created_at"] == body["updated_at"]
    assert body["is_archived"] is False


def test_post_default_provenance_is_api(client, auth_headers, sample_definition):
    r = client.post(
        "/v1/workflows",
        json={"name": "default", "definition": sample_definition},
        headers=auth_headers,
    )
    assert r.status_code == 201, r.text
    assert r.json()["provenance"] == "api"


def test_get_list_returns_created(client, auth_headers, sample_definition):
    _create(client, auth_headers, sample_definition)
    r = client.get("/v1/workflows", headers=auth_headers)
    assert r.status_code == 200
    body = r.json()
    assert len(body["items"]) == 1
    assert body["next_cursor"] is None
    assert body["has_more"] is False


def test_get_list_filters_by_provenance(client, auth_headers, sample_definition):
    _create(client, auth_headers, sample_definition, provenance="api")
    _create(client, auth_headers, sample_definition, provenance="meta_agent")
    r = client.get("/v1/workflows?provenance=meta_agent", headers=auth_headers)
    assert r.status_code == 200
    items = r.json()["items"]
    assert len(items) == 1
    assert items[0]["provenance"] == "meta_agent"


def test_get_by_id(client, auth_headers, sample_definition):
    wf_id = _create(client, auth_headers, sample_definition).json()["id"]
    r = client.get(f"/v1/workflows/{wf_id}", headers=auth_headers)
    assert r.status_code == 200
    assert r.json()["id"] == wf_id


def test_get_by_unknown_id_returns_404(client, auth_headers):
    r = client.get("/v1/workflows/01ABC", headers=auth_headers)
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "WORKFLOW_NOT_FOUND"


def test_put_replaces_workflow(client, auth_headers, sample_definition):
    created = _create(client, auth_headers, sample_definition).json()
    wf_id = created["id"]

    r = client.put(
        f"/v1/workflows/{wf_id}",
        json={"name": "renamed", "definition": sample_definition, "provenance": "api"},
        headers=auth_headers,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["name"] == "renamed"
    assert body["created_at"] == created["created_at"]
    assert body["updated_at"] != created["updated_at"]


def test_patch_archive(client, auth_headers, sample_definition):
    wf_id = _create(client, auth_headers, sample_definition).json()["id"]
    r = client.patch(f"/v1/workflows/{wf_id}", json={"is_archived": True}, headers=auth_headers)
    assert r.status_code == 200, r.text
    assert r.json()["is_archived"] is True

    # archived workflows are filtered out by default
    listed = client.get("/v1/workflows", headers=auth_headers).json()
    assert listed["items"] == []
    listed_archived = client.get("/v1/workflows?archived=true", headers=auth_headers).json()
    assert len(listed_archived["items"]) == 1
    assert listed_archived["items"][0]["is_archived"] is True


def test_delete_removes(client, auth_headers, sample_definition):
    wf_id = _create(client, auth_headers, sample_definition).json()["id"]
    r = client.delete(f"/v1/workflows/{wf_id}", headers=auth_headers)
    assert r.status_code == 204
    r = client.get(f"/v1/workflows/{wf_id}", headers=auth_headers)
    assert r.status_code == 404


def test_delete_returns_409_when_runs_reference(client, auth_headers, sample_definition, app_with_token):
    """DELETE returns 409 + WORKFLOW_HAS_RUNS when a run references the workflow."""
    wf_id = _create(client, auth_headers, sample_definition).json()["id"]

    # Insert a real run row referencing the workflow. The Session 04 migration
    # created the real `runs` table; we use the DAL so the row is valid.
    from spren.models import TaskInput
    from spren.storage import Database  # local import to avoid circular at module load
    from spren.storage.runs import insert_run

    # Reach into the same on-disk DB the app uses (discovered via /v1/bootstrap).
    boot = client.get("/v1/bootstrap", headers=auth_headers).json()
    db = Database(__import__("pathlib").Path(boot["data_dir"]))
    insert_run(
        db.connection,
        run_id="01J9X4ABCDEFGHJKMPRUN1",
        workflow_id=wf_id,
        task_input=TaskInput(),
    )
    db.connection.commit()
    db.close()

    r = client.delete(f"/v1/workflows/{wf_id}", headers=auth_headers)
    assert r.status_code == 409, r.text
    body = r.json()
    assert body["error"]["code"] == "WORKFLOW_HAS_RUNS"

    # Workflow row still present
    r2 = client.get(f"/v1/workflows/{wf_id}", headers=auth_headers)
    assert r2.status_code == 200


def test_pagination_25_with_limit_10(client, auth_headers, sample_definition):
    for i in range(25):
        _create(client, auth_headers, sample_definition, name=f"p{i:02d}")

    r = client.get("/v1/workflows?limit=10", headers=auth_headers)
    body = r.json()
    assert len(body["items"]) == 10
    assert body["has_more"] is True
    assert body["next_cursor"] is not None

    cursor = body["next_cursor"]
    r2 = client.get(f"/v1/workflows?cursor={cursor}&limit=10", headers=auth_headers)
    body2 = r2.json()
    assert len(body2["items"]) == 10
    assert body2["has_more"] is True

    r3 = client.get(f"/v1/workflows?cursor={body2['next_cursor']}&limit=10", headers=auth_headers)
    body3 = r3.json()
    assert len(body3["items"]) == 5
    assert body3["has_more"] is False
    assert body3["next_cursor"] is None

    # rows across pages do not overlap
    page_ids = [w["id"] for page in (body, body2, body3) for w in page["items"]]
    assert len(page_ids) == 25
    assert len(set(page_ids)) == 25  # no duplicates


def test_pagination_default_limit_is_20(client, auth_headers, sample_definition):
    """default limit is 20 (when ?limit= omitted)."""
    for i in range(30):
        _create(client, auth_headers, sample_definition, name=f"d{i:02d}")
    r = client.get("/v1/workflows", headers=auth_headers)
    body = r.json()
    assert len(body["items"]) == 20
    assert body["has_more"] is True


def test_cursor_is_bare_ulid(client, auth_headers, sample_definition):
    """cursor is the last returned row's ULID string (NOT base64 / HMAC)."""
    for i in range(3):
        _create(client, auth_headers, sample_definition, name=f"c{i}")
    body = client.get("/v1/workflows?limit=2", headers=auth_headers).json()
    assert body["next_cursor"] == body["items"][-1]["id"]
    # ULID format: 26 chars Crockford base32
    assert len(body["next_cursor"]) == 26


def test_pagination_limit_clamps(client, auth_headers, sample_definition):
    r = client.get("/v1/workflows?limit=0", headers=auth_headers)
    assert r.status_code == 422
    r = client.get("/v1/workflows?limit=101", headers=auth_headers)
    assert r.status_code == 422


# --- auth ---


def test_missing_auth_returns_401(client):
    r = client.get("/v1/workflows")
    assert r.status_code == 401


def test_invalid_auth_returns_401(client):
    r = client.get("/v1/workflows", headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401


@pytest.mark.parametrize(
    "method,path",
    [
        ("GET", "/v1/workflows"),
        ("POST", "/v1/workflows"),
        ("GET", "/v1/workflows/01"),
        ("PUT", "/v1/workflows/01"),
        ("PATCH", "/v1/workflows/01"),
        ("DELETE", "/v1/workflows/01"),
        ("POST", "/v1/workflows/import-python"),
    ],
)
def test_every_endpoint_requires_auth(client, method, path):
    """every /v1/workflows* endpoint returns 401 without auth."""
    r = client.request(method, path, json={} if method in {"POST", "PUT", "PATCH"} else None)
    assert r.status_code == 401, f"{method} {path} → {r.status_code}"


def test_healthz_unauth(client):
    """/healthz returns 200 without auth."""
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_bootstrap_includes_workflows_endpoint(client, auth_headers):
    """/v1/bootstrap.endpoints['workflows'] = '/v1/workflows'."""
    r = client.get("/v1/bootstrap", headers=auth_headers)
    assert r.status_code == 200
    body = r.json()
    assert body["endpoints"]["workflows"] == "/v1/workflows"


# --- cross-validation reject ---


def test_create_rejects_orphan_agent_ref(client, auth_headers, sample_definition):
    bad = {
        **sample_definition,
        "agents": {},  # node references agent_1 which is missing
    }
    r = client.post(
        "/v1/workflows",
        json={"name": "bad", "definition": bad, "provenance": "api"},
        headers=auth_headers,
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "VALIDATION_FAILED"


def test_create_accepts_formerly_reserved_node_name(client, auth_headers, sample_definition):
    """AC-7: there is no Spren reserved-name validator post-reframe; the
    framework NodeSpec accepts a node named 'User'/'system'/'tool'."""
    ok = {
        **sample_definition,
        "topology": {**sample_definition["topology"], "nodes": [{"name": "User", "kind": "agent"}]},
        "agents": {},
    }
    r = client.post(
        "/v1/workflows",
        json={"name": "ok", "definition": ok, "provenance": "api"},
        headers=auth_headers,
    )
    assert r.status_code == 201, r.text


# --- idempotency ---


def test_idempotency_replay_same_key(client, auth_headers, sample_definition):
    headers = {**auth_headers, "Idempotency-Key": "abc-123"}
    r1 = client.post(
        "/v1/workflows",
        json={"name": "idem", "definition": sample_definition, "provenance": "api"},
        headers=headers,
    )
    assert r1.status_code == 201
    id1 = r1.json()["id"]

    r2 = client.post(
        "/v1/workflows",
        json={"name": "idem", "definition": sample_definition, "provenance": "api"},
        headers=headers,
    )
    assert r2.status_code == 201
    id2 = r2.json()["id"]
    assert id1 == id2

    # only ONE workflow really exists
    listed = client.get("/v1/workflows", headers=auth_headers).json()
    assert len(listed["items"]) == 1


def test_idempotency_same_key_different_method_is_fresh(client, auth_headers, sample_definition):
    """Cache key is (method, path, key) — same key on different method is a fresh request."""
    headers_post = {**auth_headers, "Idempotency-Key": "abc"}
    r1 = client.post(
        "/v1/workflows",
        json={"name": "first", "definition": sample_definition, "provenance": "api"},
        headers=headers_post,
    )
    wf_id = r1.json()["id"]

    headers_delete = {**auth_headers, "Idempotency-Key": "abc"}
    r2 = client.delete(f"/v1/workflows/{wf_id}", headers=headers_delete)
    assert r2.status_code == 204  # not a replay of the POST


def test_idempotency_no_key_no_cache(client, auth_headers, sample_definition):
    r1 = _create(client, auth_headers, sample_definition, name="a")
    r2 = _create(client, auth_headers, sample_definition, name="a")
    assert r1.json()["id"] != r2.json()["id"]


def test_idempotency_replays_for_patch(client, auth_headers, sample_definition):
    """idempotency works for PATCH too, not just POST."""
    wf_id = _create(client, auth_headers, sample_definition).json()["id"]
    headers = {**auth_headers, "Idempotency-Key": "patch-1"}
    r1 = client.patch(f"/v1/workflows/{wf_id}", json={"name": "renamed"}, headers=headers)
    assert r1.status_code == 200
    # second replay returns the cached body
    r2 = client.patch(f"/v1/workflows/{wf_id}", json={"name": "renamed"}, headers=headers)
    assert r2.status_code == 200
    assert r2.json()["updated_at"] == r1.json()["updated_at"]


def test_idempotency_replays_for_put(client, auth_headers, sample_definition):
    """idempotency works for PUT."""
    wf_id = _create(client, auth_headers, sample_definition).json()["id"]
    headers = {**auth_headers, "Idempotency-Key": "put-1"}
    body = {"name": "replaced", "definition": sample_definition, "provenance": "api"}
    r1 = client.put(f"/v1/workflows/{wf_id}", json=body, headers=headers)
    assert r1.status_code == 200
    r2 = client.put(f"/v1/workflows/{wf_id}", json=body, headers=headers)
    assert r2.status_code == 200
    assert r2.json()["updated_at"] == r1.json()["updated_at"]


def test_idempotency_replays_for_delete(client, auth_headers, sample_definition):
    """idempotency works for DELETE."""
    wf_id = _create(client, auth_headers, sample_definition).json()["id"]
    headers = {**auth_headers, "Idempotency-Key": "del-1"}
    r1 = client.delete(f"/v1/workflows/{wf_id}", headers=headers)
    assert r1.status_code == 204
    # The workflow is gone now; replaying must return the cached 204 (NOT 404).
    r2 = client.delete(f"/v1/workflows/{wf_id}", headers=headers)
    assert r2.status_code == 204


def test_idempotency_key_on_get_is_ignored(client, auth_headers, sample_definition):
    """Idempotency-Key on GET is ignored (no caching, no replay)."""
    headers = {**auth_headers, "Idempotency-Key": "get-1"}
    _create(client, auth_headers, sample_definition, name="seen")
    r1 = client.get("/v1/workflows", headers=headers)
    assert r1.status_code == 200
    assert len(r1.json()["items"]) == 1

    # add another workflow; if Idempotency-Key were honored on GET, the second
    # response would replay the cached body with len == 1
    _create(client, auth_headers, sample_definition, name="another")
    r2 = client.get("/v1/workflows", headers=headers)
    assert r2.status_code == 200
    assert len(r2.json()["items"]) == 2


# --- PATCH preserves untouched fields ---


def test_patch_preserves_untouched_fields(client, auth_headers, sample_definition):
    """PATCH must leave fields not in the body unchanged."""
    created = _create(client, auth_headers, sample_definition, name="orig").json()
    wf_id = created["id"]

    r = client.patch(f"/v1/workflows/{wf_id}", json={"is_archived": True}, headers=auth_headers)
    assert r.status_code == 200
    body = r.json()
    assert body["is_archived"] is True
    assert body["name"] == "orig"
    assert body["definition"] == created["definition"]
    assert body["created_at"] == created["created_at"]


# --- error envelope shape ---


def test_error_envelope_has_full_triplet(client, auth_headers):
    """every non-2xx error follows {error: {code, message, details}}."""
    r = client.get("/v1/workflows/nonexistent", headers=auth_headers)
    assert r.status_code == 404
    body = r.json()
    assert set(body.keys()) == {"error"}
    assert set(body["error"].keys()) == {"code", "message", "details"}
    assert isinstance(body["error"]["code"], str)
    assert isinstance(body["error"]["message"], str)
    assert isinstance(body["error"]["details"], dict)
