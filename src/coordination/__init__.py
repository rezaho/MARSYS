"""Coordination system components for multi-agent framework."""

# Context management
from .context_manager import ContextSelector, ContextTemplate

# Routing components
from .routing import Router, RoutingDecision, ExecutionStep, StepType, BranchSpec, RoutingContext

# Branch types
from .branches.types import (
    ExecutionBranch,
    BranchType,
    BranchTopology,
    BranchState,
    BranchStatus,
    BranchResult,
    StepResult,
    ExecutionState,
)

# Topology components
from .topology import TopologyAnalyzer
from .topology.graph import TopologyGraph, TopologyEdge
from .topology.core import Topology

# Note: Validation components should be imported directly from their module
# to avoid circular imports with agents module
# from .validation.response_validator import ValidationProcessor, ValidationResult, etc.

# Execution components
from .execution.branch_executor import BranchExecutor
from .execution.branch_spawner import DynamicBranchSpawner
from .execution.step_executor import StepExecutor

# High-level orchestration
from .orchestra import Orchestra, OrchestraResult

# State management
from .state import StateManager, StorageBackend, FileStorageBackend, CheckpointManager

# Rules engine
from .rules import RulesEngine, Rule, RuleResult, RuleContext, TimeoutRule, MaxAgentsRule, MaxStepsRule

__all__ = [
    # Context
    'ContextSelector', 
    'ContextTemplate',
    # Routing
    'Router',
    'RoutingDecision',
    'ExecutionStep',
    'StepType',
    'BranchSpec',
    'RoutingContext',
    # Branches
    'ExecutionBranch',
    'BranchType',
    'BranchTopology',
    'BranchState',
    'BranchStatus',
    'BranchResult',
    'StepResult',
    'ExecutionState',
    # Topology
    'Topology',
    'TopologyAnalyzer',
    'TopologyGraph',
    'TopologyEdge',
    # Execution
    'BranchExecutor',
    'DynamicBranchSpawner',
    'StepExecutor',
    # Orchestra
    'Orchestra',
    'OrchestraResult',
    # State management
    'StateManager',
    'StorageBackend',
    'FileStorageBackend',
    'CheckpointManager',
    # Rules engine
    'RulesEngine',
    'Rule',
    'RuleResult',
    'RuleContext',
    'TimeoutRule',
    'MaxAgentsRule',
    'MaxStepsRule',
]