"""
Factory functions for creating pooled agents.

This module provides convenient factory functions for creating agent pools,
especially for agents that require async initialization like BrowserAgent.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Type, Union

from .agent_pool import AgentPool
from .agents import Agent, BaseAgent
from .registry import AgentRegistry

logger = logging.getLogger(__name__)


async def create_agent_pool(
    agent_class: Type[BaseAgent],
    num_instances: int = 1,
    register: bool = True,
    *args,
    **kwargs
) -> Union[BaseAgent, AgentPool]:
    """
    Create an agent or agent pool based on num_instances.
    
    Args:
        agent_class: The agent class to instantiate
        num_instances: Number of instances (1 returns single agent, >1 returns pool)
        register: Whether to register with AgentRegistry
        *args: Positional arguments for agent constructor
        **kwargs: Keyword arguments for agent constructor
        
    Returns:
        Single agent if num_instances=1, AgentPool otherwise
    """
    if num_instances < 1:
        raise ValueError(f"num_instances must be at least 1, got {num_instances}")
    
    if num_instances == 1:
        # Single instance - create directly
        if hasattr(agent_class, 'create_safe'):
            # Async creation (e.g., BrowserAgent)
            agent = await agent_class.create_safe(*args, **kwargs)
        else:
            # Sync creation
            agent = agent_class(*args, **kwargs)
        
        if register and not hasattr(agent, '_registered'):
            # Agent's __init__ usually handles registration, but ensure it's done
            agent._registered = True
        
        return agent
    else:
        # Multiple instances - create pool
        if hasattr(agent_class, 'create_safe'):
            # Async creation
            pool = await AgentPool.create_async(
                agent_class,
                num_instances,
                *args,
                **kwargs
            )
        else:
            # Sync creation
            pool = AgentPool(
                agent_class,
                num_instances,
                *args,
                **kwargs
            )
        
        if register:
            AgentRegistry.register_pool(pool)
        
        return pool


async def create_browser_agent_pool(
    num_instances: int = 1,
    model_config: Optional[Any] = None,
    name: str = "browser_agent",
    headless: bool = True,
    register: bool = True,
    **kwargs
) -> Union['BrowserAgent', AgentPool]:
    """
    Create a BrowserAgent or pool of BrowserAgents.
    
    Args:
        num_instances: Number of browser instances to create
        model_config: Model configuration for the agents
        name: Base name for the agent(s)
        headless: Whether to run browsers in headless mode
        register: Whether to register with AgentRegistry
        **kwargs: Additional arguments for BrowserAgent
        
    Returns:
        Single BrowserAgent if num_instances=1, AgentPool otherwise
    """
    from .browser_agent import BrowserAgent
    
    if not model_config:
        raise ValueError("model_config is required for BrowserAgent")

    # Set default goal and instruction if not provided
    if 'goal' not in kwargs:
        kwargs['goal'] = "Extract information from web pages and perform browser automation tasks"

    if 'instruction' not in kwargs:
        kwargs['instruction'] = (
            "A browser automation agent capable of web navigation, "
            "element interaction, and information extraction."
        )

    return await create_agent_pool(
        BrowserAgent,
        num_instances=num_instances,
        register=register,
        model_config=model_config,
        name=name,
        headless=headless,
        **kwargs
    )


def create_agent_pool_sync(
    agent_class: Type[BaseAgent],
    num_instances: int = 1,
    register: bool = True,
    *args,
    **kwargs
) -> Union[BaseAgent, AgentPool]:
    """
    Synchronous version of create_agent_pool for non-async agents.
    
    Args:
        agent_class: The agent class to instantiate
        num_instances: Number of instances (1 returns single agent, >1 returns pool)
        register: Whether to register with AgentRegistry
        *args: Positional arguments for agent constructor
        **kwargs: Keyword arguments for agent constructor
        
    Returns:
        Single agent if num_instances=1, AgentPool otherwise
    """
    if num_instances < 1:
        raise ValueError(f"num_instances must be at least 1, got {num_instances}")
    
    if hasattr(agent_class, 'create_safe'):
        raise ValueError(
            f"{agent_class.__name__} requires async initialization. "
            f"Use create_agent_pool() instead."
        )
    
    if num_instances == 1:
        # Single instance
        agent = agent_class(*args, **kwargs)
        if register and not hasattr(agent, '_registered'):
            agent._registered = True
        return agent
    else:
        # Pool
        pool = AgentPool(
            agent_class,
            num_instances,
            *args,
            **kwargs
        )
        if register:
            AgentRegistry.register_pool(pool)
        return pool


async def create_agents_with_pools(
    agent_configs: List[Dict[str, Any]],
    register: bool = True
) -> Dict[str, Union[BaseAgent, AgentPool]]:
    """
    Create multiple agents and pools from configuration list.
    
    Args:
        agent_configs: List of agent configuration dictionaries
        register: Whether to register agents/pools with AgentRegistry
        
    Returns:
        Dictionary mapping agent names to agents/pools
        
    Example config:
        [
            {
                "class": Agent,
                "name": "analyzer",
                "num_instances": 1,
                "model_config": config,
                "description": "Analyzes data"
            },
            {
                "class": BrowserAgent,
                "name": "browser",
                "num_instances": 3,
                "model_config": config,
                "headless": True
            }
        ]
    """
    agents = {}
    
    # Separate async and sync agents
    async_configs = []
    sync_configs = []
    
    for config in agent_configs:
        agent_class = config.pop('class')
        if hasattr(agent_class, 'create_safe'):
            async_configs.append((agent_class, config))
        else:
            sync_configs.append((agent_class, config))
    
    # Create async agents in parallel
    if async_configs:
        async_tasks = []
        async_names = []
        
        for agent_class, config in async_configs:
            name = config.pop('name', agent_class.__name__)
            num_instances = config.pop('num_instances', 1)
            
            task = create_agent_pool(
                agent_class,
                num_instances=num_instances,
                register=register,
                agent_name=name,
                **config
            )
            async_tasks.append(task)
            async_names.append(name)
        
        # Wait for all async agents to be created
        async_agents = await asyncio.gather(*async_tasks)
        
        # Add to result dict
        for name, agent in zip(async_names, async_agents):
            agents[name] = agent
    
    # Create sync agents
    for agent_class, config in sync_configs:
        name = config.pop('name', agent_class.__name__)
        num_instances = config.pop('num_instances', 1)
        
        agent = create_agent_pool_sync(
            agent_class,
            num_instances=num_instances,
            register=register,
            agent_name=name,
            **config
        )
        agents[name] = agent
    
    logger.info(f"Created {len(agents)} agents/pools: {list(agents.keys())}")
    return agents


def get_pool_statistics(pool_or_name: Union[str, AgentPool]) -> Dict[str, Any]:
    """
    Get statistics for an agent pool.
    
    Args:
        pool_or_name: Pool instance or name registered in AgentRegistry
        
    Returns:
        Dictionary of pool statistics
    """
    if isinstance(pool_or_name, str):
        pool = AgentRegistry.get_pool(pool_or_name)
        if not pool:
            return {"error": f"Pool '{pool_or_name}' not found"}
    else:
        pool = pool_or_name
    
    return pool.get_statistics()


async def cleanup_all_pools() -> None:
    """
    Clean up all registered agent pools.
    
    This should be called at the end of a session to ensure
    proper cleanup of resources (e.g., browser instances).
    """
    from .registry import AgentRegistry
    
    pools = []
    for name in list(AgentRegistry._pools.keys()):
        pool = AgentRegistry.get_pool(name)
        if pool:
            pools.append(pool)
    
    if pools:
        logger.info(f"Cleaning up {len(pools)} agent pools")
        
        # Clean up all pools in parallel
        cleanup_tasks = [pool.cleanup() for pool in pools]
        await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        logger.info("All agent pools cleaned up")