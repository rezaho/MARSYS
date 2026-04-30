"""
Execution components for the MARS coordination system.

Public surface for the unified-barrier orchestrator and its supporting
runtime / step executor / coordination event types.
"""

from ..events import BranchCompletedEvent, BranchCreatedEvent
from .det_nodes import (
    DeterministicNode,
    EndNode,
    RESERVED_DETNODE_NAMES,
    StartNode,
    UserNode,
)
from .deterministic_runtime import DeterministicRuntime
from .orchestrator import Orchestrator
from .orchestrator_types import (
    Barrier,
    Branch,
    ConvergencePolicy,
    DetNodeContext,
    Invocation,
    Runtime,
    StepResult,
    TopologyLike,
    WorkflowResult,
    new_barrier_id,
    new_branch_id,
    reset_ids,
)
from .real_runtime import RealRuntime
from .step_executor import StepContext, StepExecutor, ToolExecutionResult
from .tool_executor import RealToolExecutor

__all__ = [
    # Events
    "BranchCompletedEvent",
    "BranchCreatedEvent",
    # Orchestrator core
    "Orchestrator",
    "WorkflowResult",
    "Branch",
    "Barrier",
    "ConvergencePolicy",
    "Invocation",
    "StepResult",
    "Runtime",
    "TopologyLike",
    "DetNodeContext",
    "new_branch_id",
    "new_barrier_id",
    "reset_ids",
    # Deterministic nodes
    "DeterministicNode",
    "StartNode",
    "EndNode",
    "UserNode",
    "RESERVED_DETNODE_NAMES",
    # Runtimes
    "RealRuntime",
    "DeterministicRuntime",
    # Step execution
    "StepExecutor",
    "StepContext",
    "RealToolExecutor",
    "ToolExecutionResult",
]
