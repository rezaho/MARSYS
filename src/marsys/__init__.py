"""
MARSYS - Multi-Agent Reasoning Systems

A powerful framework for orchestrating collaborative AI agents with
sophisticated reasoning, planning, and autonomous capabilities.

Author: Reza Hosseini
License: Apache-2.0
"""

__version__ = "0.1.0"

# Core agent classes
from .agents import (
    Agent,
    BaseAgent,
    AgentPool,
    BrowserAgent,
    BaseLearnableAgent,
    LearnableAgent,
    AgentRegistry,
    ConversationMemory,
    Message,
    MemoryManager,
)

# Model configuration
from .models import ModelConfig

# Coordination system
from .coordination import (
    Orchestra,
    OrchestraResult,
    Topology,
    ExecutionBranch,
    BranchResult,
)

# Common imports for convenience
__all__ = [
    # Version
    "__version__",
    # Agents
    "Agent",
    "BaseAgent",
    "AgentPool",
    "BrowserAgent",
    "BaseLearnableAgent",
    "LearnableAgent",
    "AgentRegistry",
    # Memory
    "ConversationMemory",
    "Message",
    "MemoryManager",
    # Models
    "ModelConfig",
    # Coordination
    "Orchestra",
    "OrchestraResult",
    "Topology",
    "ExecutionBranch",
    "BranchResult",
]
