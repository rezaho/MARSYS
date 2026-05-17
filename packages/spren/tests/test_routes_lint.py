"""HTTP-level tests for ``POST /v1/workflows/{id}/lint``.

The deeper rule coverage lives in ``test_lint_workflow.py``. This file
exercises the route surface: auth, 404 for unknown id, response shape, and
that a known workflow returns 200 even when findings are empty.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def created_workflow(client, auth_headers, sample_definition):
    payload = {
        "name": "lint-test",
        "definition": sample_definition,
        "provenance": "api",
    }
    response = client.post("/v1/workflows", headers=auth_headers, json=payload)
    assert response.status_code == 201, response.text
    return response.json()


def test_lint_requires_auth(client) -> None:
    response = client.post("/v1/workflows/nonexistent/lint")
    assert response.status_code == 401


def test_lint_unknown_workflow_returns_404(client, auth_headers) -> None:
    response = client.post("/v1/workflows/01ABCDEF/lint", headers=auth_headers)
    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "WORKFLOW_NOT_FOUND"


def test_lint_returns_findings_shape(client, auth_headers, created_workflow) -> None:
    response = client.post(
        f"/v1/workflows/{created_workflow['id']}/lint",
        headers=auth_headers,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert "findings" in body
    assert isinstance(body["findings"], list)


def test_lint_finding_has_required_keys(client, auth_headers, sample_definition) -> None:
    # Replace the clean fixture with a workflow that references an unknown tool
    # so we get at least one finding back.
    sample_definition["agents"]["Researcher"]["tools"] = ["this_tool_does_not_exist"]
    create = client.post(
        "/v1/workflows",
        headers=auth_headers,
        json={
            "name": "lint-shape",
            "definition": sample_definition,
            "provenance": "api",
        },
    )
    workflow_id = create.json()["id"]
    response = client.post(f"/v1/workflows/{workflow_id}/lint", headers=auth_headers)
    body = response.json()
    assert body["findings"], "expected at least one finding for unknown tool"
    finding = body["findings"][0]
    assert set(finding.keys()) == {"severity", "code", "node_name", "edge", "message", "suggestion"}


def test_lint_for_clean_workflow_returns_empty_findings(client, auth_headers, sample_definition) -> None:
    # Use a framework-registered tool so unknown_tool doesn't fire.
    sample_definition["agents"]["Researcher"]["tools"] = ["web_search"]
    create = client.post(
        "/v1/workflows",
        headers=auth_headers,
        json={
            "name": "clean",
            "definition": sample_definition,
            "provenance": "api",
        },
    )
    workflow_id = create.json()["id"]
    response = client.post(f"/v1/workflows/{workflow_id}/lint", headers=auth_headers)
    body = response.json()
    assert body["findings"] == []
