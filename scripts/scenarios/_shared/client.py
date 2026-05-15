"""Thin httpx wrapper that talks to a live sidecar with auth wired."""
from __future__ import annotations

from typing import Any

import httpx

from .sidecar import SidecarHandle


class SprenClient:
    """Synchronous client; one per scenario run."""

    def __init__(self, handle: SidecarHandle, *, timeout: float = 10.0) -> None:
        self._handle = handle
        self._http = httpx.Client(
            base_url=handle.base_url,
            headers=handle.auth_headers,
            timeout=timeout,
        )

    @property
    def base_url(self) -> str:
        return self._handle.base_url

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "SprenClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # -- raw request access for findings logging ----------------------

    def request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        return self._http.request(method, path, **kwargs)

    # -- ergonomic helpers --------------------------------------------

    def bootstrap(self) -> httpx.Response:
        return self._http.get("/v1/bootstrap")

    def healthz(self) -> httpx.Response:
        # No auth on healthz by design.
        return httpx.get(f"{self._handle.base_url}/healthz", timeout=5)

    def list_workflows(self, **params: Any) -> httpx.Response:
        return self._http.get("/v1/workflows", params=params)

    def create_workflow(self, payload: dict[str, Any]) -> httpx.Response:
        return self._http.post("/v1/workflows", json=payload)

    def get_workflow(self, workflow_id: str) -> httpx.Response:
        return self._http.get(f"/v1/workflows/{workflow_id}")

    def replace_workflow(self, workflow_id: str, payload: dict[str, Any]) -> httpx.Response:
        return self._http.put(f"/v1/workflows/{workflow_id}", json=payload)

    def patch_workflow(self, workflow_id: str, payload: dict[str, Any]) -> httpx.Response:
        return self._http.patch(f"/v1/workflows/{workflow_id}", json=payload)

    def delete_workflow(self, workflow_id: str) -> httpx.Response:
        return self._http.delete(f"/v1/workflows/{workflow_id}")

    def lint_workflow(self, workflow_id: str) -> httpx.Response:
        return self._http.post(f"/v1/workflows/{workflow_id}/lint")

    def list_tools(self) -> httpx.Response:
        return self._http.get("/v1/tools")

    def import_python(self, file_path: str, *, filename: str | None = None) -> httpx.Response:
        with open(file_path, "rb") as fh:
            files = {"file": (filename or file_path.split("/")[-1], fh.read(), "text/x-python")}
        return self._http.post("/v1/workflows/import-python", files=files)


def request_summary(response: httpx.Response) -> dict[str, Any]:
    """Compact summary for findings — never the full body if it's large."""
    body: Any
    try:
        body = response.json()
    except Exception:
        text = response.text
        body = text if len(text) <= 500 else text[:500] + "...[truncated]"
    return {
        "method": response.request.method,
        "url": str(response.request.url),
        "status": response.status_code,
        "body": body,
    }
