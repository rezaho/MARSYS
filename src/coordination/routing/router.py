"""
Router component for intelligent routing decisions in the MARS coordination system.

The Router takes validation results and determines the next execution steps,
handling various routing patterns including sequential, parallel, tool execution,
and conversation continuation.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from ..validation.response_validator import ValidationResult, ActionType
from ..topology.graph import TopologyGraph
from ..branches.types import ExecutionBranch, BranchType, ConversationPattern
from .types import (
    RoutingDecision, 
    ExecutionStep, 
    StepType, 
    BranchSpec,
    RoutingContext
)

logger = logging.getLogger(__name__)


class Router:
    """
    Intelligent routing component that converts validation results into execution steps.
    
    The Router is responsible for:
    1. Taking ValidationResult and determining next steps21
    2. Converting action types to ExecutionStep objects
    3. Handling different routing patterns (sequential, parallel, conversation)
    4. Integrating with TopologyGraph for permission checks
    5. Providing routing suggestions for retry scenarios
    """
    
    def __init__(self, topology_graph: TopologyGraph):
        """
        Initialize the Router with a topology graph.
        
        Args:
            topology_graph: The topology graph for permission validation and path finding
        """
        self.topology_graph = topology_graph
        logger.info("Router initialized with topology graph")
    
    async def route(
        self,
        validation_result: ValidationResult,
        current_branch: ExecutionBranch,
        routing_context: RoutingContext
    ) -> RoutingDecision:
        """
        Main routing method that determines next execution steps.
        
        Args:
            validation_result: The result from response validation
            current_branch: The currently executing branch
            routing_context: Additional context for routing decisions
            
        Returns:
            RoutingDecision with next steps and branch specifications
        """
        logger.debug(f"Routing for action type: {validation_result.action_type}")
        
        # Handle invalid validation results
        if not validation_result.is_valid:
            return self._handle_invalid_result(validation_result, routing_context)
        
        # Route based on action type
        action_type = validation_result.action_type
        
        if action_type == ActionType.INVOKE_AGENT:
            return self._route_agent_invocation(
                validation_result, current_branch, routing_context
            )
        elif action_type == ActionType.PARALLEL_INVOKE:
            return self._route_parallel_invocation(
                validation_result, current_branch, routing_context
            )
        elif action_type == ActionType.CALL_TOOL:
            return self._route_tool_execution(
                validation_result, current_branch, routing_context
            )
        elif action_type == ActionType.FINAL_RESPONSE:
            return self._route_final_response(
                validation_result, current_branch, routing_context
            )
        elif action_type == ActionType.END_CONVERSATION:
            return self._route_end_conversation(
                validation_result, current_branch, routing_context
            )
        elif action_type == ActionType.WAIT_AND_AGGREGATE:
            return self._route_wait_and_aggregate(
                validation_result, current_branch, routing_context
            )
        else:
            logger.warning(f"Unknown action type: {action_type}")
            return self._create_completion_decision("Unknown action type")
    
    def _route_agent_invocation(
        self,
        validation_result: ValidationResult,
        current_branch: ExecutionBranch,
        routing_context: RoutingContext
    ) -> RoutingDecision:
        """Handle single agent invocation routing."""
        if not validation_result.next_agents:
            logger.error("No target agent specified for invocation")
            return self._create_completion_decision("No target agent specified")
        
        target_agent = validation_result.next_agents[0]
        current_agent = routing_context.current_agent
        
        # Check if this is a conversation continuation
        if self._is_conversation_continuation(
            current_branch, current_agent, target_agent
        ):
            logger.debug(f"Continuing conversation: {current_agent} -> {target_agent}")
            return self._route_conversation_continuation(
                target_agent, validation_result, current_branch, routing_context
            )
        
        # Check topology permissions
        if not self._validate_transition(current_agent, target_agent):
            logger.warning(f"Invalid transition: {current_agent} -> {target_agent}")
            return self._create_completion_decision(
                f"Transition from {current_agent} to {target_agent} not allowed"
            )
        
        # Create execution step for agent invocation
        # Use agent_requests for User interactions
        if target_agent == "User" and "agent_requests" in validation_result.parsed_response:
            action_input = validation_result.parsed_response["agent_requests"].get("User", "")
        else:
            action_input = validation_result.parsed_response.get("action_input", "")
        
        step = ExecutionStep(
            step_type=StepType.AGENT,
            agent_name=target_agent,
            request=action_input,
            metadata={
                "from_agent": current_agent,
                "action_type": "invoke_agent"
            }
        )
        
        return RoutingDecision(
            next_steps=[step],
            should_continue=True,
            metadata={"routing_type": "sequential_agent"}
        )
    
    def _route_parallel_invocation(
        self,
        validation_result: ValidationResult,
        current_branch: ExecutionBranch,
        routing_context: RoutingContext
    ) -> RoutingDecision:
        """Handle parallel agent invocation routing."""
        if not validation_result.next_agents or len(validation_result.next_agents) < 2:
            logger.error("Parallel invocation requires at least 2 agents")
            return self._create_completion_decision(
                "Parallel invocation requires multiple agents"
            )
        
        current_agent = routing_context.current_agent
        target_agents = validation_result.next_agents
        
        # Validate all transitions
        invalid_transitions = []
        for target in target_agents:
            if not self._validate_transition(current_agent, target):
                invalid_transitions.append(f"{current_agent} -> {target}")
        
        if invalid_transitions:
            logger.warning(f"Invalid transitions for parallel: {invalid_transitions}")
            return self._create_completion_decision(
                f"Invalid transitions: {', '.join(invalid_transitions)}"
            )
        
        # Create branch specifications for parallel execution
        action_input = validation_result.parsed_response.get("action_input", {})
        branch_specs = []
        
        for target_agent in target_agents:
            # Each parallel agent gets its own branch
            agent_input = action_input.get(target_agent, "") if isinstance(action_input, dict) else action_input
            
            branch_spec = BranchSpec(
                agents=[target_agent],
                entry_agent=target_agent,
                initial_request=agent_input,
                branch_type="simple",
                metadata={
                    "parent_agent": current_agent,
                    "parallel_group": validation_result.parsed_response.get("parallel_group_id")
                }
            )
            branch_specs.append(branch_spec)
        
        # Parent branch should wait for children
        return RoutingDecision(
            next_steps=[ExecutionStep(step_type=StepType.WAIT)],
            should_continue=False,  # Parent pauses
            should_wait=True,      # Wait for children
            child_branch_specs=branch_specs,
            metadata={
                "routing_type": "parallel_agent",
                "child_count": len(branch_specs)
            }
        )
    
    def _route_tool_execution(
        self,
        validation_result: ValidationResult,
        current_branch: ExecutionBranch,
        routing_context: RoutingContext
    ) -> RoutingDecision:
        """Handle tool execution routing."""
        if not validation_result.tool_calls:
            logger.error("No tool calls specified")
            return self._create_completion_decision("No tool calls specified")
        
        # Create tool execution step
        step = ExecutionStep(
            step_type=StepType.TOOL,
            agent_name=routing_context.current_agent,
            tool_calls=validation_result.tool_calls,
            metadata={
                "tool_count": len(validation_result.tool_calls)
            }
        )
        
        return RoutingDecision(
            next_steps=[step],
            should_continue=True,  # Continue after tool execution
            metadata={"routing_type": "tool_execution"}
        )
    
    def _route_final_response(
        self,
        validation_result: ValidationResult,
        current_branch: ExecutionBranch,
        routing_context: RoutingContext
    ) -> RoutingDecision:
        """Handle final response routing."""
        # Create completion step
        step = ExecutionStep(
            step_type=StepType.COMPLETE,
            agent_name=routing_context.current_agent,
            request=validation_result.parsed_response.get("final_response"),
            metadata={
                "completion_type": "final_response"
            }
        )
        
        return RoutingDecision(
            next_steps=[step],
            should_continue=False,
            completion_reason="Agent provided final response",
            metadata={"routing_type": "completion"}
        )
    
    def _route_end_conversation(
        self,
        validation_result: ValidationResult,
        current_branch: ExecutionBranch,
        routing_context: RoutingContext
    ) -> RoutingDecision:
        """Handle end of conversation routing."""
        if not current_branch.is_conversation_branch():
            logger.warning("End conversation called on non-conversation branch")
        
        # Create completion step
        step = ExecutionStep(
            step_type=StepType.COMPLETE,
            agent_name=routing_context.current_agent,
            request=validation_result.parsed_response.get("conclusion"),
            metadata={
                "completion_type": "end_conversation",
                "conversation_turns": routing_context.conversation_turns
            }
        )
        
        return RoutingDecision(
            next_steps=[step],
            should_continue=False,
            completion_reason="Conversation ended",
            metadata={"routing_type": "conversation_end"}
        )
    
    def _route_wait_and_aggregate(
        self,
        validation_result: ValidationResult,
        current_branch: ExecutionBranch,
        routing_context: RoutingContext
    ) -> RoutingDecision:
        """Handle wait and aggregate routing."""
        # Create aggregation step
        step = ExecutionStep(
            step_type=StepType.AGGREGATE,
            agent_name=routing_context.current_agent,
            metadata={
                "aggregation_type": validation_result.parsed_response.get("aggregation_type", "default")
            }
        )
        
        return RoutingDecision(
            next_steps=[step],
            should_continue=False,
            should_wait=True,
            metadata={"routing_type": "wait_aggregate"}
        )
    
    def _route_conversation_continuation(
        self,
        target_agent: str,
        validation_result: ValidationResult,
        current_branch: ExecutionBranch,
        routing_context: RoutingContext
    ) -> RoutingDecision:
        """Handle conversation continuation within the same branch."""
        # Update branch type if needed
        if current_branch.type == BranchType.SIMPLE:
            current_branch.type = BranchType.CONVERSATION
            current_branch.topology.conversation_pattern = ConversationPattern.DIALOGUE
            logger.info(f"Branch {current_branch.id} converted to CONVERSATION type")
        
        # Create execution step for next agent in conversation
        action_input = validation_result.parsed_response.get("action_input", "")
        step = ExecutionStep(
            step_type=StepType.AGENT,
            agent_name=target_agent,
            request=action_input,
            metadata={
                "from_agent": routing_context.current_agent,
                "conversation_turn": routing_context.conversation_turns + 1,
                "action_type": "conversation_continuation"
            }
        )
        
        return RoutingDecision(
            next_steps=[step],
            should_continue=True,
            metadata={
                "routing_type": "conversation_continuation",
                "conversation_pattern": current_branch.topology.conversation_pattern.value
            }
        )
    
    def _handle_invalid_result(
        self,
        validation_result: ValidationResult,
        routing_context: RoutingContext
    ) -> RoutingDecision:
        """Handle invalid validation results with retry suggestions."""
        logger.error(f"Invalid validation result: {validation_result.error_message}")
        
        # If retry is suggested, create a retry step
        if validation_result.retry_suggestion:
            retry_step = ExecutionStep(
                step_type=StepType.AGENT,
                agent_name=routing_context.current_agent,
                request=validation_result.retry_suggestion,
                metadata={
                    "retry_reason": validation_result.error_message,
                    "retry_attempt": routing_context.metadata.get("retry_count", 0) + 1
                }
            )
            
            return RoutingDecision(
                next_steps=[retry_step],
                should_continue=True,
                metadata={
                    "routing_type": "retry",
                    "retry_count": routing_context.metadata.get("retry_count", 0) + 1
                }
            )
        
        # No retry possible, complete with error
        return self._create_completion_decision(
            f"Validation error: {validation_result.error_message}"
        )
    
    def _validate_transition(self, from_agent: str, to_agent: str) -> bool:
        """Validate if a transition is allowed based on topology."""
        # Check topology graph for permission
        if self.topology_graph.has_edge(from_agent, to_agent):
            return True
        
        # Check if it's part of a conversation loop
        if self.topology_graph.is_in_conversation_loop(from_agent, to_agent):
            return True
        
        # Check if transition is explicitly allowed in adjacency
        adjacency = self.topology_graph.adjacency.get(from_agent, [])
        return to_agent in adjacency
    
    def _is_conversation_continuation(
        self,
        branch: ExecutionBranch,
        current_agent: str,
        target_agent: str
    ) -> bool:
        """Check if this is a continuation of an existing conversation."""
        # Only stay in conversation if already in conversation branch
        if branch.type == BranchType.CONVERSATION:
            return target_agent in branch.topology.agents
        
        # NEVER auto-convert to conversation based on edges
        # This fixes the hub-and-spoke pattern issue
        return False
    
    def _create_completion_decision(self, reason: str) -> RoutingDecision:
        """Create a decision that completes the branch."""
        return RoutingDecision(
            next_steps=[ExecutionStep(step_type=StepType.COMPLETE)],
            should_continue=False,
            completion_reason=reason,
            metadata={"routing_type": "error_completion"}
        )
    
    def suggest_alternative_route(
        self,
        current_agent: str,
        failed_target: str,
        routing_context: RoutingContext
    ) -> Optional[str]:
        """
        Suggest an alternative agent when the intended route fails.
        
        Args:
            current_agent: The current agent
            failed_target: The agent that couldn't be reached
            routing_context: Current routing context
            
        Returns:
            Alternative agent name or None if no alternatives exist
        """
        # Get all possible next agents
        possible_agents = self.topology_graph.get_next_agents(current_agent)
        
        # Filter out the failed target and already visited agents
        alternatives = [
            agent for agent in possible_agents
            if agent != failed_target and agent not in routing_context.branch_agents
        ]
        
        if alternatives:
            logger.info(f"Suggesting alternative route: {alternatives[0]}")
            return alternatives[0]
        
        return None