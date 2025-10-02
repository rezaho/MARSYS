"""
Example demonstrating Router integration with the coordination system.

This example shows how the Router works with ValidationProcessor,
TopologyGraph, and BranchExecutor to handle different routing scenarios.
"""

import asyncio
from typing import Dict, Any

from src.coordination.routing import Router, RoutingContext
from src.coordination.validation.response_validator import (
    ResponseValidator, ValidationResult, ActionType
)
from src.coordination.topology.graph import TopologyGraph
from src.coordination.topology.analyzer import TopologyAnalyzer
from src.coordination.branches.types import (
    ExecutionBranch, BranchType, BranchTopology, BranchState, BranchStatus
)


async def demonstrate_routing():
    """Demonstrate various routing scenarios."""
    
    # 1. Create topology
    print("1. Creating topology...")
    topology_def = {
        "nodes": ["User", "PlannerAgent", "ResearchAgent", "WriterAgent", "ReviewerAgent"],
        "edges": [
            "User -> PlannerAgent",
            "PlannerAgent -> ResearchAgent",
            "PlannerAgent -> WriterAgent",
            "ResearchAgent -> WriterAgent",
            "WriterAgent -> ReviewerAgent",
            "ReviewerAgent -> PlannerAgent"  # Feedback loop
        ],
        "rules": []
    }
    
    # 2. Build topology graph
    analyzer = TopologyAnalyzer()
    graph = analyzer.analyze(topology_def)
    print(f"   Topology built with {len(graph.nodes)} nodes")
    
    # 3. Create Router
    router = Router(graph)
    print("   Router initialized")
    
    # 4. Create ResponseValidator
    validator = ResponseValidator(graph)
    print("   ResponseValidator initialized")
    
    print("\n" + "="*50 + "\n")
    
    # Scenario 1: Sequential Agent Invocation
    print("Scenario 1: Sequential Agent Invocation")
    print("-" * 30)
    
    # Current state
    branch = ExecutionBranch(
        id="main_branch",
        type=BranchType.SIMPLE,
        topology=BranchTopology(
            agents=["PlannerAgent"],
            entry_agent="PlannerAgent"
        ),
        state=BranchState(status=BranchStatus.RUNNING)
    )
    
    context = RoutingContext(
        current_branch_id="main_branch",
        current_agent="PlannerAgent",
        conversation_history=[],
        branch_agents=["PlannerAgent"]
    )
    
    # Simulate agent response
    agent_response = {
        "next_action": "invoke_agent",
        "action_input": "Research the latest AI trends"
    }
    
    # Validate response
    validation_result = await validator.process_response(
        agent_response, 
        "PlannerAgent",
        branch.topology
    )
    validation_result.next_agents = ["ResearchAgent"]  # Set target
    
    # Route decision
    decision = await router.route(validation_result, branch, context)
    
    print(f"Validation: {validation_result.is_valid}")
    print(f"Action Type: {validation_result.action_type}")
    print(f"Routing Decision:")
    print(f"  - Should Continue: {decision.should_continue}")
    print(f"  - Next Steps: {len(decision.next_steps)}")
    if decision.next_steps:
        step = decision.next_steps[0]
        print(f"  - Step Type: {step.step_type}")
        print(f"  - Target Agent: {step.agent_name}")
        print(f"  - Request: {step.request}")
    
    print("\n" + "="*50 + "\n")
    
    # Scenario 2: Parallel Agent Invocation
    print("Scenario 2: Parallel Agent Invocation")
    print("-" * 30)
    
    # Update context
    context.current_agent = "PlannerAgent"
    
    # Simulate parallel invocation response
    agent_response = {
        "next_action": "parallel_invoke",
        "agents": ["ResearchAgent", "WriterAgent"],
        "action_input": {
            "ResearchAgent": "Find sources on AI ethics",
            "WriterAgent": "Draft introduction paragraph"
        }
    }
    
    # Manually create validation result for parallel
    validation_result = ValidationResult(
        is_valid=True,
        action_type=ActionType.PARALLEL_INVOKE,
        parsed_response=agent_response,
        next_agents=["ResearchAgent", "WriterAgent"]
    )
    
    # Route decision
    decision = await router.route(validation_result, branch, context)
    
    print(f"Routing Decision:")
    print(f"  - Should Continue: {decision.should_continue}")
    print(f"  - Should Wait: {decision.should_wait}")
    print(f"  - Child Branches: {len(decision.child_branch_specs)}")
    
    for i, spec in enumerate(decision.child_branch_specs):
        print(f"\n  Child Branch {i+1}:")
        print(f"    - Agents: {spec.agents}")
        print(f"    - Entry Agent: {spec.entry_agent}")
        print(f"    - Initial Request: {spec.initial_request}")
    
    print("\n" + "="*50 + "\n")
    
    # Scenario 3: Tool Execution
    print("Scenario 3: Tool Execution")
    print("-" * 30)
    
    # Update context
    context.current_agent = "ResearchAgent"
    
    # Simulate tool call response
    agent_response = {
        "next_action": "call_tool",
        "tool_calls": [
            {"name": "web_search", "args": {"query": "AI ethics 2024"}},
            {"name": "summarize", "args": {"text": "..."}}
        ]
    }
    
    validation_result = ValidationResult(
        is_valid=True,
        action_type=ActionType.CALL_TOOL,
        parsed_response=agent_response,
        tool_calls=agent_response["tool_calls"]
    )
    
    # Route decision
    decision = await router.route(validation_result, branch, context)
    
    print(f"Routing Decision:")
    print(f"  - Should Continue: {decision.should_continue}")
    print(f"  - Next Steps: {len(decision.next_steps)}")
    
    if decision.next_steps:
        step = decision.next_steps[0]
        print(f"  - Step Type: {step.step_type}")
        print(f"  - Tool Calls: {len(step.tool_calls)}")
        for tool in step.tool_calls:
            print(f"    - {tool['name']}({tool['args']})")
    
    print("\n" + "="*50 + "\n")
    
    # Scenario 4: Invalid Transition
    print("Scenario 4: Invalid Transition Handling")
    print("-" * 30)
    
    # Try invalid transition
    context.current_agent = "ResearchAgent"
    
    agent_response = {
        "next_action": "invoke_agent",
        "action_input": "Review this"
    }
    
    # Try to go directly to ReviewerAgent (not allowed)
    validation_result = ValidationResult(
        is_valid=True,
        action_type=ActionType.INVOKE_AGENT,
        parsed_response=agent_response,
        next_agents=["ReviewerAgent"]
    )
    
    # Route decision
    decision = await router.route(validation_result, branch, context)
    
    print(f"Routing Decision:")
    print(f"  - Should Continue: {decision.should_continue}")
    print(f"  - Completion Reason: {decision.completion_reason}")
    
    # Suggest alternative
    alternative = router.suggest_alternative_route(
        "ResearchAgent", 
        "ReviewerAgent",
        context
    )
    print(f"  - Alternative Route: {alternative}")
    
    print("\n" + "="*50 + "\n")
    
    # Scenario 5: Final Response
    print("Scenario 5: Final Response")
    print("-" * 30)
    
    context.current_agent = "ReviewerAgent"
    
    agent_response = {
        "next_action": "final_response",
        "final_response": "The article has been reviewed and approved. Quality score: 9/10."
    }
    
    validation_result = ValidationResult(
        is_valid=True,
        action_type=ActionType.FINAL_RESPONSE,
        parsed_response=agent_response
    )
    
    # Route decision
    decision = await router.route(validation_result, branch, context)
    
    print(f"Routing Decision:")
    print(f"  - Should Continue: {decision.should_continue}")
    print(f"  - Completion Reason: {decision.completion_reason}")
    print(f"  - Final Response: {decision.next_steps[0].request}")


def main():
    """Run the routing demonstration."""
    print("Router Integration Example")
    print("========================\n")
    
    asyncio.run(demonstrate_routing())
    
    print("\nExample complete!")


if __name__ == "__main__":
    main()