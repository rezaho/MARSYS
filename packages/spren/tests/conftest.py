"""Shared pytest fixtures for the Spren package."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from spren.auth import generate_token
from spren.server import create_app


@pytest.fixture
def auth_token() -> str:
    return generate_token()


@pytest.fixture
def app_with_token(auth_token: str, tmp_path: Path):
    return create_app(
        token=auth_token,
        port=8765,
        data_dir=tmp_path,
        started_at=datetime(2026, 5, 4, tzinfo=timezone.utc),
    )


@pytest.fixture
def client(app_with_token):
    return TestClient(app_with_token)
