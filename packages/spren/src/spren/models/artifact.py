"""Artifact info shape — files written under <data-dir>/data/runs/{id}/artifacts/."""
from __future__ import annotations

from datetime import datetime
from typing import List

from pydantic import BaseModel, ConfigDict


class ArtifactInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    size_bytes: int
    mime_type: str
    created_at: datetime


class ArtifactListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: List[ArtifactInfo]
