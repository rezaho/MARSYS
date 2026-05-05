"""Server-level tests: routes, auth, CORS."""
from __future__ import annotations

from fastapi.testclient import TestClient


class TestHealthz:
    def test_returns_200_without_auth(self, client: TestClient) -> None:
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


class TestBootstrap:
    def test_401_without_auth(self, client: TestClient) -> None:
        r = client.get("/v1/bootstrap")
        assert r.status_code == 401

    def test_401_with_wrong_token(self, client: TestClient) -> None:
        r = client.get(
            "/v1/bootstrap",
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert r.status_code == 401

    def test_200_with_correct_token(self, client: TestClient, auth_token: str) -> None:
        r = client.get(
            "/v1/bootstrap",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["framework"]["version"]
        assert body["spren"]["version"] == "0.3.0"
        assert body["spren"]["active"] is False
        assert body["surfaces"] == ["gui"]
        assert isinstance(body["capabilities"], dict)
        assert isinstance(body["endpoints"], dict)
        assert body["data_dir"]
        assert body["started_at"]


class TestCORS:
    def test_preflight_allowed_localhost_vite(self, client: TestClient) -> None:
        r = client.options(
            "/v1/bootstrap",
            headers={
                "Origin": "http://127.0.0.1:5173",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Authorization",
            },
        )
        assert r.status_code == 200
        assert r.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"

    def test_preflight_allowed_tauri_origin(self, client: TestClient) -> None:
        r = client.options(
            "/v1/bootstrap",
            headers={
                "Origin": "tauri://localhost",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Authorization",
            },
        )
        assert r.status_code == 200
        assert r.headers["access-control-allow-origin"] == "tauri://localhost"

    def test_preflight_rejected_malicious_origin(self, client: TestClient) -> None:
        r = client.options(
            "/v1/bootstrap",
            headers={
                "Origin": "http://malicious.example",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Authorization",
            },
        )
        assert "access-control-allow-origin" not in r.headers
