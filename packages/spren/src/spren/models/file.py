"""File metadata + upload response shapes."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class FileMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    id: str
    original_name: str
    mime_type: str
    size_bytes: int
    sha256: str
    created_at: datetime


class FileUploadResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    file_id: str
    original_name: str
    mime_type: str
    size_bytes: int
    sha256: str
