"""Pydantic mirror of marsys's ExecutionConfig + ConvergencePolicyConfig.

Mirrors the load-bearing subset Spren needs at the API boundary; full timeout
and tracing knobs are passed through opaquely. The convergence-policy field
accepts the same ``Union[float, str, ConvergencePolicyConfigSpec]``
polymorphism marsys exposes upstream.
"""
from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class ConvergencePolicyConfigSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_ratio: float = Field(1.0, ge=0.0, le=1.0)
    on_insufficient: Literal["proceed", "fail", "user"] = "fail"
    terminate_orphans: bool = True
    log_level: Literal["info", "warning", "error"] = "warning"


class StatusConfigSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    verbosity: int | None = None


class ExecutionConfigSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steering_mode: Literal["auto", "always", "error"] = "error"
    dynamic_convergence_enabled: bool = True
    parent_completes_on_spawn: bool = True
    auto_detect_convergence: bool = True

    convergence_timeout: float = 300.0
    convergence_policy: Union[float, str, ConvergencePolicyConfigSpec] = 1.0
    branch_timeout: float = 600.0
    agent_acquisition_timeout: float = 240.0
    step_timeout: float = 600.0
    tool_execution_timeout: float = 120.0
    user_interaction_timeout: float = 300.0

    status: StatusConfigSpec = Field(default_factory=StatusConfigSpec)

    user_first: bool = False
    initial_user_msg: str | None = None
    user_interaction: Literal["terminal", "web", "none"] = "none"

    auto_cleanup_agents: bool = True
    cleanup_scope: Literal["topology_nodes", "used_agents"] = "topology_nodes"

    response_format: str = "json"
    content_only_steering_threshold: int = 2
    content_only_hard_limit: int = 10
