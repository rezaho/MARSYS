"""Pydantic envelope for the Spren API error contract.

All non-2xx responses follow ``{"error": {"code": ErrorCode, "message": str,
"details": dict}}``. ``ErrorCode`` is a Literal union so the generated
TypeScript types render as a discriminating string union on the frontend.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


ErrorCode = Literal[
    "WORKFLOW_NOT_FOUND",
    "WORKFLOW_HAS_RUNS",
    "WORKFLOW_ARCHIVED",
    "PYTHON_IMPORT_REJECTED",
    "MIGRATION_FAILED",
    "VALIDATION_FAILED",
    "INVALID_CURSOR",
    "INTERNAL_ERROR",
    "RUN_NOT_FOUND",
    "RUN_NOT_CANCELLABLE",
    "ATTACHMENTS_NOT_YET_SUPPORTED",
    "TRIGGER_NOT_YET_SUPPORTED",
    "INVALID_TASK_INPUT",
]


class ErrorPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: ErrorCode
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error: ErrorPayload
