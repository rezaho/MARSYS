"""Spren consumes the framework's canonical execution-config wire types.

SP-005: the framework owns these shapes; Spren does not mirror them. The
prior Spren mirror was a divergent *subset* (missing tracing + most
StatusConfig fields) — the divergence this reframe removes. Materialization
uses ``pydantic_to_execution_config`` (see ``runs/materialize.py``).
"""
from __future__ import annotations

from marsys.coordination.serialize import (
    ConvergencePolicyConfigSpec,
    ExecutionConfigSpec,
    StatusConfigSpec,
    TracingConfigSpec,
)

__all__ = [
    "ConvergencePolicyConfigSpec",
    "ExecutionConfigSpec",
    "StatusConfigSpec",
    "TracingConfigSpec",
]
