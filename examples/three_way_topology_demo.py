"""
Demonstration of the three ways to define multi-agent topologies.

This example shows how to create the same hub-and-spoke topology using:
1. String notation (simple and readable)
2. Object notation (type-safe with IDE support)
3. Pattern configuration (pre-defined patterns)
"""

import asyncio
from typing import Dict, Any

from src.coordination.orchestra import Orchestra
from src.coordination.topology.core import Edge, EdgePattern
from src.coordination.topology.patterns import PatternConfig
from src.coordination.rules.basic_rules import TimeoutRule, ParallelRule
from src.agents.agents import Agent
from src.agents.registry import AgentRegistry


# Create some mock agents for demonstration
class ResearchAgent(Agent):
    """Mock research agent."""
    
    async def _run(self, prompt: str, context: Dict[str, Any], **kwargs) -> Any:
        return {
            "next_action": "final_response",
            "content": f"Research findings on: {prompt}"
        }


class AnalysisAgent(Agent):
    """Mock analysis agent."""
    
    async def _run(self, prompt: str, context: Dict[str, Any], **kwargs) -> Any:
        return {
            "next_action": "final_response", 
            "content": f"Analysis results for: {prompt}"
        }


class CoordinatorAgent(Agent):
    """Mock coordinator agent."""
    
    async def _run(self, prompt: str, context: Dict[str, Any], **kwargs) -> Any:
        # In a real scenario, this would intelligently route to research/analysis
        return {
            "next_action": "parallel_invoke",
            "agents": ["researcher", "analyst"]
        }


def example_1_string_notation():
    """
    Method 1: String Notation
    
    Simplest way - just strings in a dictionary.
    Perfect for configuration files or quick prototypes.
    """
    print("\n=== Method 1: String Notation ===")
    
    topology = {
        "nodes": ["User", "coordinator", "researcher", "analyst"],
        "edges": [
            "User -> coordinator",
            "coordinator <=> researcher",  # Reflexive (returns to coordinator)
            "coordinator <=> analyst"      # Reflexive (returns to coordinator)
        ],
        "rules": [
            "parallel(researcher, analyst)",  # Execute in parallel
            "timeout(30)",                    # 30 second timeout
            "max_steps(10)"                   # Maximum 10 steps
        ],
        "metadata": {
            "description": "Research coordination system"
        }
    }
    
    print("Topology definition:")
    print(f"  Nodes: {topology['nodes']}")
    print(f"  Edges: {topology['edges']}")
    print(f"  Rules: {topology['rules']}")
    
    return topology


def example_2_object_notation():
    """
    Method 2: Object Notation
    
    Type-safe with IDE support. Mix strings, objects, and agent instances.
    Best for programmatic topology creation with validation.
    """
    print("\n=== Method 2: Object Notation ===")
    
    # Create agent instances
    coordinator = CoordinatorAgent(name="coordinator")
    researcher = ResearchAgent(name="researcher")
    analyst = AnalysisAgent(name="analyst")
    
    topology = {
        "nodes": [
            "User",          # String for User node
            coordinator,     # Agent instance
            researcher,      # Agent instance
            analyst         # Agent instance
        ],
        "edges": [
            "User -> coordinator",  # String notation still works
            Edge(   # Edge object with explicit properties
                source="coordinator",
                target="researcher",
                bidirectional=True,
                pattern=EdgePattern.REFLEXIVE
            ),
            Edge(
                source="coordinator",
                target="analyst",
                bidirectional=True,
                pattern=EdgePattern.REFLEXIVE
            )
        ],
        "rules": [
            ParallelRule(   # Rule object
                agents=["researcher", "analyst"],
                trigger_agent="coordinator",
                wait_for_all=True
            ),
            TimeoutRule(max_duration_seconds=30),  # Rule object
            "max_steps(10)"  # String notation still works
        ],
        "metadata": {
            "description": "Research coordination system",
            "agents": {
                "coordinator": coordinator.__class__.__name__,
                "researcher": researcher.__class__.__name__,
                "analyst": analyst.__class__.__name__
            }
        }
    }
    
    print("Topology definition:")
    print(f"  Nodes: {[n.name if hasattr(n, 'name') else n for n in topology['nodes']]}")
    print(f"  Edges: 1 string + 2 Edge objects")
    print(f"  Rules: 2 Rule objects + 1 string")
    
    return topology


def example_3_pattern_configuration():
    """
    Method 3: Pattern Configuration
    
    Pre-defined patterns for common topologies.
    Fastest way to create standard architectures.
    """
    print("\n=== Method 3: Pattern Configuration ===")
    
    # Create hub-and-spoke pattern
    pattern = PatternConfig.hub_and_spoke(
        hub="coordinator",
        spokes=["researcher", "analyst"],
        parallel_spokes=True,     # Execute spokes in parallel
        reflexive=True,           # Spokes return results to hub
        timeout=30,              # Metadata: 30 second timeout
        max_steps=10,            # Metadata: max 10 steps
        description="Research coordination system"
    )
    
    print("Pattern configuration:")
    print(f"  Pattern: {pattern.pattern.value}")
    print(f"  Hub: {pattern.params['hub']}")
    print(f"  Spokes: {pattern.params['spokes']}")
    print(f"  Parallel: {pattern.params['parallel_spokes']}")
    print(f"  Reflexive: {pattern.params['reflexive']}")
    
    return pattern


async def run_example(topology, example_name: str):
    """Run an example topology with Orchestra."""
    print(f"\n--- Running {example_name} ---")
    
    # Register our agents
    registry = AgentRegistry()
    registry.register("coordinator", CoordinatorAgent)
    registry.register("researcher", ResearchAgent) 
    registry.register("analyst", AnalysisAgent)
    
    try:
        # Run the topology
        result = await Orchestra.run(
            task="Research the benefits of renewable energy",
            topology=topology,
            agent_registry=registry,
            max_steps=20
        )
        
        print(f"Success: {result.success}")
        print(f"Total steps: {result.total_steps}")
        print(f"Duration: {result.total_duration:.2f}s")
        print(f"Branches: {len(result.branch_results)}")
        
        if result.final_response:
            print(f"Final response: {result.final_response}")
            
    except Exception as e:
        print(f"Error: {e}")


async def main():
    """Run all three examples."""
    print("=" * 60)
    print("Three Ways to Define Multi-Agent Topologies")
    print("=" * 60)
    
    # Example 1: String Notation
    topology1 = example_1_string_notation()
    await run_example(topology1, "String Notation")
    
    # Example 2: Object Notation  
    topology2 = example_2_object_notation()
    await run_example(topology2, "Object Notation")
    
    # Example 3: Pattern Configuration
    topology3 = example_3_pattern_configuration()
    await run_example(topology3, "Pattern Configuration")
    
    print("\n" + "=" * 60)
    print("All three methods create equivalent topologies!")
    print("Choose based on your needs:")
    print("- String notation: Simple, readable, great for configs")
    print("- Object notation: Type-safe, IDE support, validation")
    print("- Pattern config: Fast, pre-defined, battle-tested")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())