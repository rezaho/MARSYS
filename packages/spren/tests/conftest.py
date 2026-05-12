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
def data_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def app_with_token(auth_token: str, data_dir: Path):
    return create_app(
        token=auth_token,
        port=8765,
        data_dir=data_dir,
        started_at=datetime(2026, 5, 4, tzinfo=timezone.utc),
        enable_draft_sweeper=False,
    )


@pytest.fixture
def client(app_with_token):
    with TestClient(app_with_token) as c:
        yield c


@pytest.fixture
def auth_headers(auth_token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {auth_token}"}


@pytest.fixture
def sample_definition() -> dict:
    return {
        "topology": {
            "nodes": [
                {"name": "Researcher", "node_type": "agent", "agent_ref": "agent_1"},
            ],
            "edges": [],
            "rules": [],
        },
        "agents": {
            "agent_1": {
                "agent_model": {"type": "api", "name": "gpt-4o", "provider": "openai"},
                "name": "Researcher",
                "goal": "do research",
                "instruction": "research things",
                "tools": ["search_web"],
                "memory_retention": "session",
                "allowed_peers": [],
            },
        },
        "execution_config": {},
    }
