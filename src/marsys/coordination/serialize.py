"""
Pydantic wire-shape mirrors of the coordination dataclasses (``ExecutionConfig``,
``ConvergencePolicyConfig``, ``TracingConfig``, ``StatusConfig``) plus the
round-trip helpers ``execution_config_to_pydantic`` / ``pydantic_to_execution_config``.

The four specs compose into :class:`marsys.coordination.topology.serialize.WorkflowDefinition`
through ``WorkflowDefinition.execution_config``. They are grouped here (next to
``config.py``) rather than colocated next to each dataclass: the four travel
together on the wire so splitting them across four ``serialize.py`` modules
adds ceremony without separation benefit.

The polymorphic ``convergence_policy`` field is intentionally kept open over the
three input shapes that ``ConvergencePolicyConfig.from_value`` accepts at
runtime (bare-float, named-string, full spec). A
``model_validator(mode="before")`` keeps each input shape intact through
round-trip so producers see exactly the discriminant branch they emitted.
"""

from __future__ import annotations

from typing import Any, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import (
    ConvergencePolicyConfig,
    ExecutionConfig,
    StatusConfig,
    VerbosityLevel,
)
from .tracing.config import TracingConfig


class ConvergencePolicyConfigSpec(BaseModel):
    """Wire mirror of :class:`ConvergencePolicyConfig`."""

    model_config = ConfigDict(extra="forbid")

    min_ratio: float = Field(1.0, ge=0.0, le=1.0)
    on_insufficient: Literal["proceed", "fail", "user"] = "fail"
    terminate_orphans: bool = True
    log_level: Literal["info", "warning", "error"] = "warning"


class StatusConfigSpec(BaseModel):
    """Wire mirror of :class:`StatusConfig`."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    verbosity: Optional[int] = None

    cli_output: bool = True
    cli_colors: bool = True
    show_thoughts: bool = True
    show_tool_calls: bool = True
    show_timings: bool = True

    aggregation_window_ms: int = 500
    aggregate_parallel: bool = True

    max_events_per_session: int = 10000
    session_cleanup_after_s: int = 3600

    channels: List[str] = Field(default_factory=lambda: ["cli"])

    show_agent_prefixes: bool = True
    prefix_width: int = 20
    prefix_alignment: str = "left"

    follow_up_timeout: float = 30.0


class TracingConfigSpec(BaseModel):
    """Wire mirror of :class:`TracingConfig`.

    Excludes ``sinks`` and ``message_store`` fields (runtime objects that
    cannot be persisted) and the ``redactor`` instance — these get supplied
    at runtime by the consumer wiring its own :class:`TelemetrySink` set.
    Persisting only the toggles preserves trace-shape intent across runs.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    output_dir: str = "./traces"
    include_generation_details: bool = True
    include_message_content: bool = True
    include_tool_results: bool = True
    capture_full_input: bool = False


class ExecutionConfigSpec(BaseModel):
    """Wire mirror of :class:`ExecutionConfig`.

    ``convergence_policy`` accepts the same ``Union[float, str,
    ConvergencePolicyConfigSpec]`` polymorphism that ``ExecutionConfig.convergence_policy``
    accepts at runtime. The ``model_validator(mode="before")`` normalizes
    dict inputs to ``ConvergencePolicyConfigSpec`` while leaving bare floats
    and bare strings on their own discriminant branch.
    """

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
    tracing: TracingConfigSpec = Field(default_factory=TracingConfigSpec)

    user_first: bool = False
    initial_user_msg: Optional[str] = None
    user_interaction: str = "terminal"

    auto_cleanup_agents: bool = True
    cleanup_scope: Literal["topology_nodes", "used_agents"] = "topology_nodes"

    response_format: str = "json"

    content_only_steering_threshold: int = 2
    content_only_hard_limit: int = 10

    @model_validator(mode="before")
    @classmethod
    def _normalize_convergence_policy(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "convergence_policy" not in data:
            return data
        value = data["convergence_policy"]
        if isinstance(value, str):
            return data
        if isinstance(value, bool):
            # Treat bools as floats per ConvergencePolicyConfig.from_value's
            # numeric branch — keep this off the named-string branch.
            return {**data, "convergence_policy": float(value)}
        if isinstance(value, (int, float)):
            return data
        if isinstance(value, dict):
            return {
                **data,
                "convergence_policy": ConvergencePolicyConfigSpec(**value),
            }
        return data


def _status_to_spec(status: StatusConfig) -> StatusConfigSpec:
    return StatusConfigSpec(
        enabled=status.enabled,
        verbosity=int(status.verbosity) if status.verbosity is not None else None,
        cli_output=status.cli_output,
        cli_colors=status.cli_colors,
        show_thoughts=status.show_thoughts,
        show_tool_calls=status.show_tool_calls,
        show_timings=status.show_timings,
        aggregation_window_ms=status.aggregation_window_ms,
        aggregate_parallel=status.aggregate_parallel,
        max_events_per_session=status.max_events_per_session,
        session_cleanup_after_s=status.session_cleanup_after_s,
        channels=list(status.channels),
        show_agent_prefixes=status.show_agent_prefixes,
        prefix_width=status.prefix_width,
        prefix_alignment=status.prefix_alignment,
        follow_up_timeout=status.follow_up_timeout,
    )


def _spec_to_status(spec: StatusConfigSpec) -> StatusConfig:
    return StatusConfig(
        enabled=spec.enabled,
        verbosity=VerbosityLevel(spec.verbosity) if spec.verbosity is not None else None,
        cli_output=spec.cli_output,
        cli_colors=spec.cli_colors,
        show_thoughts=spec.show_thoughts,
        show_tool_calls=spec.show_tool_calls,
        show_timings=spec.show_timings,
        aggregation_window_ms=spec.aggregation_window_ms,
        aggregate_parallel=spec.aggregate_parallel,
        max_events_per_session=spec.max_events_per_session,
        session_cleanup_after_s=spec.session_cleanup_after_s,
        channels=list(spec.channels),
        show_agent_prefixes=spec.show_agent_prefixes,
        prefix_width=spec.prefix_width,
        prefix_alignment=spec.prefix_alignment,
        follow_up_timeout=spec.follow_up_timeout,
    )


def _tracing_to_spec(tracing: TracingConfig) -> TracingConfigSpec:
    return TracingConfigSpec(
        enabled=tracing.enabled,
        output_dir=tracing.output_dir,
        include_generation_details=tracing.include_generation_details,
        include_message_content=tracing.include_message_content,
        include_tool_results=tracing.include_tool_results,
        capture_full_input=tracing.capture_full_input,
    )


def _spec_to_tracing(spec: TracingConfigSpec) -> TracingConfig:
    return TracingConfig(
        enabled=spec.enabled,
        output_dir=spec.output_dir,
        include_generation_details=spec.include_generation_details,
        include_message_content=spec.include_message_content,
        include_tool_results=spec.include_tool_results,
        capture_full_input=spec.capture_full_input,
    )


def _convergence_policy_to_spec_value(
    value: Union[float, str, ConvergencePolicyConfig],
) -> Union[float, str, ConvergencePolicyConfigSpec]:
    if isinstance(value, ConvergencePolicyConfig):
        return ConvergencePolicyConfigSpec(
            min_ratio=value.min_ratio,
            on_insufficient=value.on_insufficient,
            terminate_orphans=value.terminate_orphans,
            log_level=value.log_level,
        )
    return value


def _spec_value_to_convergence_policy(
    value: Union[float, str, ConvergencePolicyConfigSpec],
) -> ConvergencePolicyConfig:
    """Reduce the three discriminant branches to the canonical runtime config.

    ``ConvergencePolicyConfig.from_value`` already normalizes ``float`` and
    ``str`` inputs; this function adds the ``ConvergencePolicyConfigSpec``
    -> ``ConvergencePolicyConfig`` translation for the full-spec branch.
    """
    if isinstance(value, ConvergencePolicyConfigSpec):
        return ConvergencePolicyConfig(
            min_ratio=value.min_ratio,
            on_insufficient=value.on_insufficient,
            terminate_orphans=value.terminate_orphans,
            log_level=value.log_level,
        )
    return ConvergencePolicyConfig.from_value(value)


def execution_config_to_pydantic(config: ExecutionConfig) -> ExecutionConfigSpec:
    """Build an :class:`ExecutionConfigSpec` from a runtime :class:`ExecutionConfig`."""
    return ExecutionConfigSpec(
        steering_mode=config.steering_mode,
        dynamic_convergence_enabled=config.dynamic_convergence_enabled,
        parent_completes_on_spawn=config.parent_completes_on_spawn,
        auto_detect_convergence=config.auto_detect_convergence,
        convergence_timeout=config.convergence_timeout,
        convergence_policy=_convergence_policy_to_spec_value(config.convergence_policy),
        branch_timeout=config.branch_timeout,
        agent_acquisition_timeout=config.agent_acquisition_timeout,
        step_timeout=config.step_timeout,
        tool_execution_timeout=config.tool_execution_timeout,
        user_interaction_timeout=config.user_interaction_timeout,
        status=_status_to_spec(config.status),
        tracing=_tracing_to_spec(config.tracing),
        user_first=config.user_first,
        initial_user_msg=config.initial_user_msg,
        user_interaction=config.user_interaction,
        auto_cleanup_agents=config.auto_cleanup_agents,
        cleanup_scope=config.cleanup_scope,
        response_format=config.response_format,
        content_only_steering_threshold=config.content_only_steering_threshold,
        content_only_hard_limit=config.content_only_hard_limit,
    )


def pydantic_to_execution_config(spec: ExecutionConfigSpec) -> ExecutionConfig:
    """Materialize a runtime :class:`ExecutionConfig` from a spec."""
    return ExecutionConfig(
        steering_mode=spec.steering_mode,
        dynamic_convergence_enabled=spec.dynamic_convergence_enabled,
        parent_completes_on_spawn=spec.parent_completes_on_spawn,
        auto_detect_convergence=spec.auto_detect_convergence,
        convergence_timeout=spec.convergence_timeout,
        convergence_policy=_spec_value_to_convergence_policy(spec.convergence_policy),
        branch_timeout=spec.branch_timeout,
        agent_acquisition_timeout=spec.agent_acquisition_timeout,
        step_timeout=spec.step_timeout,
        tool_execution_timeout=spec.tool_execution_timeout,
        user_interaction_timeout=spec.user_interaction_timeout,
        status=_spec_to_status(spec.status),
        tracing=_spec_to_tracing(spec.tracing),
        user_first=spec.user_first,
        initial_user_msg=spec.initial_user_msg,
        user_interaction=spec.user_interaction,
        auto_cleanup_agents=spec.auto_cleanup_agents,
        cleanup_scope=spec.cleanup_scope,
        response_format=spec.response_format,
        content_only_steering_threshold=spec.content_only_steering_threshold,
        content_only_hard_limit=spec.content_only_hard_limit,
    )
