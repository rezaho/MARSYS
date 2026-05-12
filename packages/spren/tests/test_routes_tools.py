"""Tests for ``GET /v1/tools``.

Acceptance criteria AC-1 .. AC-8 (the tool registry surface). The route is
auth-gated, returns a stable list across calls, sources the registry from
the framework's ``AVAILABLE_TOOLS`` dict, and emits each callable's
``__doc__`` first line (or ``null``).
"""
from __future__ import annotations

import pytest

from marsys.environment.tools import AVAILABLE_TOOLS


def test_tools_route_requires_auth(client) -> None:
    response = client.get("/v1/tools")
    assert response.status_code == 401


def test_tools_route_returns_items_for_authed_caller(client, auth_headers) -> None:
    response = client.get("/v1/tools", headers=auth_headers)
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, dict)
    assert set(body.keys()) == {"items"}
    assert isinstance(body["items"], list)


def test_tools_response_items_have_required_keys(client, auth_headers) -> None:
    response = client.get("/v1/tools", headers=auth_headers)
    items = response.json()["items"]
    assert items, "framework registry must not be empty in v0.3"
    for item in items:
        assert set(item.keys()) == {"name", "source", "description"}
        assert isinstance(item["name"], str) and item["name"]
        assert item["source"] == "framework"
        assert item["description"] is None or isinstance(item["description"], str)


def test_tools_response_covers_every_framework_tool(client, auth_headers) -> None:
    response = client.get("/v1/tools", headers=auth_headers)
    items = response.json()["items"]
    names = {item["name"] for item in items}
    assert names == set(AVAILABLE_TOOLS.keys())


def test_tools_response_description_is_first_doc_line(client, auth_headers) -> None:
    response = client.get("/v1/tools", headers=auth_headers)
    items = {item["name"]: item for item in response.json()["items"]}

    for name, fn in AVAILABLE_TOOLS.items():
        doc = (fn.__doc__ or "").strip()
        if not doc:
            assert items[name]["description"] is None, (
                f"expected None description for {name!r} (no __doc__)"
            )
            continue
        first_line = next(
            (line.strip() for line in fn.__doc__.splitlines() if line.strip()),
            None,
        )
        assert items[name]["description"] == first_line


@pytest.mark.parametrize("n", [1, 2, 5])
def test_tools_response_is_stable_across_calls(client, auth_headers, n: int) -> None:
    bodies = [client.get("/v1/tools", headers=auth_headers).json() for _ in range(n)]
    assert all(b == bodies[0] for b in bodies)
