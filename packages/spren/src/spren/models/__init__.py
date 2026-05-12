"""Pydantic schemas for the Spren API boundary.

Mirror marsys's dataclass topology, model-config, and execution-config types,
plus Spren's own workflow envelope. The mirror is the storage-time contract;
the conversion to runnable marsys types (with credentials resolved from the
secrets store) happens at execution time in Session 04+.
"""
from __future__ import annotations

from .agent import AgentSpec, MemoryRetention
from .errors import ErrorCode, ErrorEnvelope, ErrorPayload
from .execution_config import (
    ConvergencePolicyConfigSpec,
    ExecutionConfigSpec,
    StatusConfigSpec,
)
from .lint import LintCode, LintFinding, LintResponse, LintSeverity
from .model_config import ApiProvider, ModelConfigSpec, ModelType
from .tools import (
    ImportWarningCode,
    ImportWarningPayload,
    ToolInfo,
    ToolListResponse,
    ToolSource,
    WorkflowImportResponse,
)
from .topology import EdgePattern, EdgeSpec, EdgeType, NodeSpec, NodeType, TopologySpec
from .workflow import (
    Workflow,
    WorkflowCreateRequest,
    WorkflowDefinition,
    WorkflowListResponse,
    WorkflowProvenance,
    WorkflowUpdateRequest,
)

__all__ = [
    "AgentSpec",
    "ApiProvider",
    "ConvergencePolicyConfigSpec",
    "EdgePattern",
    "EdgeSpec",
    "EdgeType",
    "ErrorCode",
    "ErrorEnvelope",
    "ErrorPayload",
    "ExecutionConfigSpec",
    "ImportWarningCode",
    "ImportWarningPayload",
    "LintCode",
    "LintFinding",
    "LintResponse",
    "LintSeverity",
    "MemoryRetention",
    "ModelConfigSpec",
    "ModelType",
    "NodeSpec",
    "NodeType",
    "StatusConfigSpec",
    "ToolInfo",
    "ToolListResponse",
    "ToolSource",
    "TopologySpec",
    "Workflow",
    "WorkflowCreateRequest",
    "WorkflowDefinition",
    "WorkflowImportResponse",
    "WorkflowListResponse",
    "WorkflowProvenance",
    "WorkflowUpdateRequest",
]
