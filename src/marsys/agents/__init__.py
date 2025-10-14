"""
This package provides the core agent infrastructure for the Multi-Agent Reasoning Systems (MARSYS) system.
"""

from .agents import Agent, BaseAgent
from .agent_pool import AgentPool
from .browser_agent import BrowserAgent
from .learnable_agents import BaseLearnableAgent, LearnableAgent
from .memory import ConversationMemory, KGMemory, MemoryManager, Message
from .pool_factory import (
    create_agent_pool,
    create_agent_pool_sync,
    create_browser_agent_pool,
    create_agents_with_pools,
    cleanup_all_pools,
    get_pool_statistics,
)
from .registry import AgentRegistry
from .utils import LogLevel, ProgressLogger, RequestContext

__all__ = [
    # Core agent classes
    "BaseAgent",
    "Agent",
    "AgentPool",
    "BrowserAgent",
    "BaseLearnableAgent",
    "LearnableAgent",
    # Pool factory functions
    "create_agent_pool",
    "create_agent_pool_sync",
    "create_browser_agent_pool",
    "create_agents_with_pools",
    "cleanup_all_pools",
    "get_pool_statistics",
    # Memory components
    "MemoryManager",
    "Message",
    "ConversationMemory",
    "KGMemory",
    # Registry
    "AgentRegistry",
    # Utilities
    "RequestContext",
    "LogLevel",
    "ProgressLogger",
]
