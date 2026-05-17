"""HTTP-level tests for ``POST /v1/workflows/{id}/lint``.

The deeper rule coverage lives in ``test_lint_workflow.py``. This file
exercises the route surface: auth, response shape, clean-empty, and the
WF-BUG-LINT-REACTIVITY contract — lint reflects the **submitted** body
(the live canvas), not any stored definition, and works for a canvas
that was never saved (no stored row). The pre-change
``test_lint_unknown_workflow_returns_404`` was intentionally removed:
linting is now a pure function of the submitted definition, so a
stored-row lookup (and its 404) no longer exists by design.
"""
from __future__ import annotations

import copy


def test_lint_requires_auth(client, sample_definition) -> None:
    # Auth must win even with a valid body.
    response = client.post("/v1/workflows/anything/lint", json=sample_definition)
    assert response.status_code == 401


def test_lint_returns_findings_shape(client, auth_headers, sample_definition) -> None:
    response = client.post(
        "/v1/workflows/wf-ctx/lint",
        headers=auth_headers,
        json=sample_definition,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert "findings" in body
    assert isinstance(body["findings"], list)


def test_lint_finding_has_required_keys(client, auth_headers, sample_definition) -> None:
    broken = copy.deepcopy(sample_definition)
    broken["agents"]["Researcher"]["tools"] = ["this_tool_does_not_exist"]
    response = client.post(
        "/v1/workflows/wf-ctx/lint", headers=auth_headers, json=broken
    )
    body = response.json()
    assert body["findings"], "expected at least one finding for unknown tool"
    finding = body["findings"][0]
    assert set(finding.keys()) == {
        "severity",
        "code",
        "node_name",
        "edge",
        "message",
        "suggestion",
    }


def test_lint_for_clean_workflow_returns_empty_findings(
    client, auth_headers, sample_definition
) -> None:
    clean = copy.deepcopy(sample_definition)
    clean["agents"]["Researcher"]["tools"] = ["web_search"]
    response = client.post(
        "/v1/workflows/wf-ctx/lint", headers=auth_headers, json=clean
    )
    assert response.status_code == 200, response.text
    assert response.json()["findings"] == []


def test_lint_uses_submitted_definition_not_stored(
    client, auth_headers, sample_definition
) -> None:
    """WF-BUG-LINT-REACTIVITY core regression: a CLEAN workflow is
    stored, but linting that id with a BROKEN body returns findings for
    the BROKEN body — proving lint reflects the live canvas, not the
    persisted definition (the old behaviour reported stale errors until
    a save + reload)."""
    clean = copy.deepcopy(sample_definition)
    clean["agents"]["Researcher"]["tools"] = ["web_search"]
    created = client.post(
        "/v1/workflows",
        headers=auth_headers,
        json={"name": "stored-clean", "definition": clean, "provenance": "api"},
    )
    assert created.status_code == 201, created.text
    workflow_id = created.json()["id"]

    broken = copy.deepcopy(sample_definition)
    broken["agents"]["Researcher"]["tools"] = ["this_tool_does_not_exist"]
    response = client.post(
        f"/v1/workflows/{workflow_id}/lint", headers=auth_headers, json=broken
    )
    assert response.status_code == 200, response.text
    findings = response.json()["findings"]
    assert findings, "lint must reflect the submitted (broken) body, not the stored clean def"
    assert any("this_tool_does_not_exist" in (f["message"] or "") for f in findings)


def test_lint_works_for_never_saved_canvas(
    client, auth_headers, sample_definition
) -> None:
    """A brand-new canvas has no stored row yet; linting it must still
    work (the old stored-fetch path 404'd here)."""
    response = client.post(
        "/v1/workflows/01NEVERSAVED/lint",
        headers=auth_headers,
        json=sample_definition,
    )
    assert response.status_code == 200, response.text
    assert "findings" in response.json()
