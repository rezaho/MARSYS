"""
Branch executor - executes individual branches with different patterns.

This module handles the execution of different branch types:
- Simple branches: Sequential agent execution
- Conversation branches: Bidirectional agent dialogue
- Nested branches: Branches containing sub-branches
- Parent-child branches: Branches that can pause for child execution
"""

import asyncio
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from ...agents.registry import AgentRegistry

if TYPE_CHECKING:
    from ..routing.router import Router
    from ..topology.graph import TopologyGraph
    from .step_executor import StepExecutor

from ..branches.types import (
    AgentDecidedCompletion,
    BranchResult,
    BranchState,
    BranchStatus,
    BranchTopology,
    BranchType,
    CompletionCondition,
    ConversationPattern,
    ConversationTurnsCompletion,
    ExecutionBranch,
    MaxStepsCompletion,
    StepResult,
)
from ..rules.rules_engine import RuleContext, RulesEngine, RuleType
from ..validation.response_validator import (
    ActionType,
    ValidationProcessor,
    ValidationResult,
)

logger = logging.getLogger(__name__)


@dataclass
class BranchExecutionContext:
    """Context passed through branch execution."""
    branch_id: str
    session_id: str
    initial_request: Any
    shared_context: Dict[str, Any] = field(default_factory=dict)
    branch_memory: Dict[str, List[Dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_memory(self, agent_name: str, message: Dict[str, Any]) -> None:
        """Add a message to branch-local memory."""
        self.branch_memory[agent_name].append(message)
    
    def get_agent_memory(self, agent_name: str) -> List[Dict[str, Any]]:
        """Get memory for a specific agent."""
        return self.branch_memory.get(agent_name, [])
    
    def get_conversation_memory(self, agents: List[str]) -> List[Dict[str, Any]]:
        """Get interleaved conversation memory for multiple agents."""
        # Combine and sort by timestamp
        all_messages = []
        for agent in agents:
            for msg in self.branch_memory.get(agent, []):
                if "timestamp" not in msg:
                    msg["timestamp"] = time.time()
                all_messages.append(msg)
        
        # Sort by timestamp to maintain conversation order
        return sorted(all_messages, key=lambda x: x.get("timestamp", 0))


class BranchExecutor:
    """
    Executes different types of branches.
    
    This is responsible for the actual execution logic within a branch,
    including agent transitions, memory management, and completion detection.
    """
    
    def __init__(
        self,
        agent_registry: AgentRegistry,
        step_executor: Optional['StepExecutor'] = None,
        response_validator: Optional[ValidationProcessor] = None,
        router: Optional['Router'] = None,
        rules_engine: Optional[RulesEngine] = None,
        topology_graph: Optional['TopologyGraph'] = None,
        max_retries: int = 10
    ):
        self.agent_registry = agent_registry
        self.step_executor = step_executor
        self.response_validator = response_validator
        self.router = router
        self.rules_engine = rules_engine
        self.topology_graph = topology_graph
        self.max_retries = max_retries
        
        # Track execution metrics
        self.execution_metrics = defaultdict(lambda: {
            "total_executions": 0,
            "successful_executions": 0,
            "average_steps": 0,
            "average_duration": 0
        })
        
        # Track branch waiting states and child results
        self.waiting_for_children: Dict[str, Set[str]] = {}  # branch_id -> set of child_ids
        self.child_results: Dict[str, Dict[str, Any]] = {}  # branch_id -> aggregated results
        self.branch_continuation: Dict[str, Dict[str, Any]] = {}  # branch_id -> continuation state
        
        # Track last step result for User node message extraction
        self._last_step_result: Optional[StepResult] = None
        self._last_agent_name: Optional[str] = None
    
    async def execute_branch(
        self,
        branch: ExecutionBranch,
        initial_request: Any,
        context: Dict[str, Any],
        resume_with_results: Optional[Dict[str, Any]] = None
    ) -> BranchResult:
        """
        Main entry point for branch execution.
        
        Args:
            branch: The branch to execute
            initial_request: Initial request to the branch
            context: Shared execution context
            resume_with_results: Aggregated child results for resumption
            
        Returns:
            BranchResult with execution outcome
        """
        # Check if this is a resumption
        if resume_with_results and branch.id in self.branch_continuation:
            logger.info(f"Resuming branch '{branch.name}' with child results")
            return await self.resume_branch(branch.id, resume_with_results)
        
        start_time = time.time()
        logger.info(f"Starting execution of branch '{branch.name}' (type: {branch.type})")
        
        # Create branch execution context
        exec_context = BranchExecutionContext(
            branch_id=branch.id,
            session_id=context.get("session_id", ""),
            initial_request=initial_request,
            shared_context=context,
            metadata=branch.metadata
        )
        
        # Update branch state
        branch.state.status = BranchStatus.RUNNING
        branch.state.start_time = start_time

        try:
            # Route to appropriate executor based on branch type
            if branch.type == BranchType.SIMPLE:
                result = await self._execute_simple_branch(branch, exec_context)
            elif branch.type == BranchType.CONVERSATION:
                result = await self._execute_conversation_branch(branch, exec_context)
            elif branch.type == BranchType.NESTED:
                result = await self._execute_nested_branch(branch, exec_context)
            elif branch.type == BranchType.USER_INTERACTION:
                # Handle user interaction branch like a simple branch
                result = await self._execute_simple_branch(branch, exec_context)
            else:
                raise ValueError(f"Unknown branch type: {branch.type}")
            
            # Check if branch is waiting for children
            if branch.state.status == BranchStatus.WAITING:
                logger.info(f"Branch '{branch.name}' is waiting for child branches")
                return result  # Temporary result, will be replaced when resumed
            
            # Update metrics
            self._update_metrics(branch.type, result, start_time)
            
            # Update branch state
            branch.state.status = BranchStatus.COMPLETED if result.success else BranchStatus.FAILED
            branch.state.end_time = time.time()
            branch.state.total_steps = result.total_steps
            
            # Release any single agents allocated to this branch
            if hasattr(self, 'branch_spawner') and self.branch_spawner:
                await self._release_branch_agents(branch.id)
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing branch '{branch.name}': {e}")
            branch.state.status = BranchStatus.FAILED
            branch.state.end_time = time.time()
            branch.state.error = str(e)
            
            return BranchResult(
                branch_id=branch.id,
                success=False,
                final_response=None,
                total_steps=branch.state.current_step,
                execution_trace=[],
                branch_memory=exec_context.branch_memory,
                error=str(e)
            )
        finally:
            # Release any pool instances used by this branch
            for agent_name in branch.topology.agents:
                if self.agent_registry.is_pool(agent_name):
                    self.agent_registry.release_to_pool(agent_name, branch.id)
                    logger.debug(f"Released pool instance of '{agent_name}' for branch '{branch.id}'")
    
    async def _execute_simple_branch(
        self,
        branch: ExecutionBranch,
        context: BranchExecutionContext
    ) -> BranchResult:
        """
        Execute a simple sequential branch.
        ium,  
        This handles linear agent execution with potential tool usage.
        """
        logger.debug(f"Executing simple branch with agents: {branch.topology.agents}")
        
        current_request = context.initial_request
        execution_trace = []
        
        # Start with the entry agent
        current_agent = branch.topology.entry_agent
        
        # Track retry attempts per agent (stored on branch for persistence across pause/resume)
        if not hasattr(branch, 'retry_counts'):
            branch.retry_counts = defaultdict(int)
        retry_counts = branch.retry_counts
        
        while True:
            # Check completion condition
            if await self._should_complete(branch, context, execution_trace):
                break

            # Pass actual retry count in context metadata (for future use)
            context.metadata["agent_retry_count"] = retry_counts.get(current_agent, 0)

            # Execute current agent
            step_result = await self._execute_agent_step(
                current_agent,
                current_request,
                context,
                branch
            )
            
            execution_trace.append(step_result)
            branch.state.current_step += 1
            
            # Update branch execution trace for User node
            if not hasattr(branch, '_execution_trace'):
                branch._execution_trace = []
            branch._execution_trace.append(step_result)
            
            if not step_result.success:
                # Handle failure
                if step_result.requires_retry and retry_counts[current_agent] < self.max_retries:
                    retry_counts[current_agent] += 1
                    logger.warning(f"Retrying agent '{current_agent}' after failure (attempt {retry_counts[current_agent]}/{self.max_retries})")
                    current_request = None  # Use None for retry to let agent continue with memory
                    continue
                else:
                    if retry_counts[current_agent] >= self.max_retries:
                        logger.error(f"Max retries ({self.max_retries}) reached for agent '{current_agent}'")
                    return BranchResult(
                        branch_id=branch.id,
                        success=False,
                        final_response=step_result.response,
                        total_steps=branch.state.current_step,
                        execution_trace=execution_trace,
                        branch_memory=context.branch_memory,
                        error=step_result.error
                    )
            
            # Check if branch should wait for children
            if step_result.waiting_for_children:
                logger.info(f"Branch '{branch.name}' pausing to wait for child branches")
                # Branch will be resumed later by DynamicBranchSpawner
                return BranchResult(
                    branch_id=branch.id,
                    success=True,
                    final_response=None,
                    total_steps=branch.state.current_step,
                    execution_trace=execution_trace,
                    branch_memory=context.branch_memory,
                    metadata={"waiting": True, "waiting_for": step_result.child_branch_ids}
                )

            # Check if branch should end after spawning (fire-and-forget pattern)
            if getattr(step_result, "should_end_branch", False):
                logger.info(
                    f"Branch '{branch.name}' ending after '{current_agent}' "
                    f"requested completion (action: {step_result.action_type})"
                )
                return BranchResult(
                    branch_id=branch.id,
                    success=True,
                    final_response=step_result.response,
                    total_steps=branch.state.current_step,
                    execution_trace=execution_trace,
                    branch_memory=context.branch_memory,
                    metadata={
                        "completion_reason": "explicit_end_after_parallel_invoke",
                        "action_type": step_result.action_type,
                        "last_agent": current_agent
                    }
                )

            # Check if this is a final response
            if step_result.action_type == "final_response" and not step_result.next_agent:
                # Extract the actual content from parsed response if available
                final_content = step_result.response
                if step_result.parsed_response and isinstance(step_result.parsed_response, dict):
                    # Check for final_response field in parsed data
                    if "final_response" in step_result.parsed_response:
                        # Debug logging when agent returns final_response
                        logger.debug(f"Agent {current_agent} returning final_response, checking reflexive pattern...")
                        final_content = step_result.parsed_response["final_response"]
                    elif "content" in step_result.parsed_response:
                        final_content = step_result.parsed_response["content"]
                
                return BranchResult(
                    branch_id=branch.id,
                    success=True,
                    final_response=final_content,
                    total_steps=branch.state.current_step,
                    execution_trace=execution_trace,
                    branch_memory=context.branch_memory
                )
            
            # Determine next agent
            next_agent = await self._determine_next_agent(
                current_agent,
                step_result,
                branch.topology.allowed_transitions,
                branch.id  # Pass branch ID for convergence checking
            )

            if not next_agent:
                # Check if stopped at convergence
                convergence_target = getattr(step_result, 'convergence_target', None)
                if convergence_target:
                    # Branch reached a convergence point
                    logger.info(f"Branch '{branch.id}' reached convergence point '{convergence_target}'")

                    # Persist convergence state on branch and group
                    branch.metadata['reached_convergence'] = True
                    branch.metadata['convergence_point'] = convergence_target
                    branch.metadata['convergence_timestamp'] = time.time()

                    # Also record in the parallel group if it exists
                    if self.branch_spawner and hasattr(self.branch_spawner, 'branch_to_group'):
                        group_id = self.branch_spawner.branch_to_group.get(branch.id)
                        if group_id:
                            group = self.branch_spawner.parallel_groups.get(group_id)
                            if group:
                                # Record convergence with response for aggregation
                                group.record_branch_convergence(
                                    branch.id,
                                    convergence_target,
                                    {
                                        'branch_id': branch.id,
                                        'response': step_result.response,
                                        'timestamp': time.time(),
                                        'agent': current_agent
                                    }
                                )
                                logger.info(
                                    f"Recorded branch '{branch.id}' convergence at '{convergence_target}' in group '{group_id}' "
                                    f"({len(group.convergence_aggregations.get(convergence_target, []))} total)"
                                )

                    # Include continuation branch in metadata if one was created
                    result_metadata = {
                        "reached_convergence": True,
                        "convergence_point": convergence_target,
                        "last_agent": current_agent,
                        "hold_reason": getattr(step_result, 'hold_reason', None)
                    }
                    continuation_branch = getattr(step_result, 'continuation_branch', None)
                    if continuation_branch:
                        result_metadata['continuation_branch'] = continuation_branch

                    # The branch completes here - group handles aggregation
                    return BranchResult(
                        branch_id=branch.id,
                        success=True,
                        final_response=step_result.response,
                        total_steps=branch.state.current_step,
                        execution_trace=execution_trace,
                        branch_memory=context.branch_memory,
                        metadata=result_metadata
                    )

                # Check if this was an error recovery attempt that failed due to no User node
                if step_result.metadata and step_result.metadata.get('needs_error_display'):
                    # User node was attempted but not available - log error and fail gracefully
                    error_details = step_result.metadata.get('error_details', {})
                    error_type = step_result.metadata.get('error_action_type', 'error')
                    error_message = error_details.get('message', 'Unknown error')
                    error_classification = error_details.get('classification', 'unknown')

                    # Log error (following existing pattern from line 716)
                    logger.error(
                        f"[{error_type.upper()}] {error_classification}: {error_message}. "
                        f"User node not available in topology. Terminating workflow."
                    )

                    # Return failed branch result with error details
                    return BranchResult(
                        branch_id=branch.id,
                        success=False,
                        final_response=None,
                        total_steps=branch.state.current_step,
                        execution_trace=execution_trace,
                        branch_memory=context.branch_memory,
                        error=f"[{error_type}] {error_classification}: {error_message}",
                        metadata=step_result.metadata
                    )

                # No next agent - branch completes
                # Extract the actual content from parsed response if available
                final_content = step_result.response
                if step_result.parsed_response and isinstance(step_result.parsed_response, dict):
                    # Check for final_response field in parsed data
                    if "final_response" in step_result.parsed_response:
                        final_content = step_result.parsed_response["final_response"]
                    elif "content" in step_result.parsed_response:
                        final_content = step_result.parsed_response["content"]
                
                return BranchResult(
                    branch_id=branch.id,
                    success=True,
                    final_response=final_content,
                    total_steps=branch.state.current_step,
                    execution_trace=execution_trace,
                    branch_memory=context.branch_memory
                )
            
            
            # Prepare request for next agent
            current_request = self._prepare_next_request(step_result, context)
            logger.info(f"Transitioning from {current_agent} to {next_agent} with request: {current_request}")
            
            # Update context metadata for next agent (important for reflexive rules)
            context.metadata["from_agent"] = current_agent
            
            current_agent = next_agent
            branch.topology.current_agent = current_agent
        
        # If we exit the loop due to completion condition (e.g., max steps)
        # return the current state as the result
        return BranchResult(
            branch_id=branch.id,
            success=True,
            final_response=execution_trace[-1].response if execution_trace else None,
            total_steps=branch.state.current_step,
            execution_trace=execution_trace,
            branch_memory=context.branch_memory,
            metadata={"completion_reason": "max_steps_reached"}
        )
    
    async def _execute_conversation_branch(
        self,
        branch: ExecutionBranch,
        context: BranchExecutionContext
    ) -> BranchResult:
        """
        Execute a conversation branch with bidirectional communication.
        
        This handles dialogue patterns between agents.
        """
        logger.debug(f"Executing conversation branch between: {branch.topology.agents}")
        
        if branch.topology.conversation_pattern == ConversationPattern.DIALOGUE:
            return await self._execute_dialogue_pattern(branch, context)
        elif branch.topology.conversation_pattern == ConversationPattern.DEBATE:
            return await self._execute_debate_pattern(branch, context)
        elif branch.topology.conversation_pattern == ConversationPattern.INTERVIEW:
            return await self._execute_interview_pattern(branch, context)
        else:
            # Default to dialogue
            return await self._execute_dialogue_pattern(branch, context)
    
    async def _execute_dialogue_pattern(
        self,
        branch: ExecutionBranch,
        context: BranchExecutionContext
    ) -> BranchResult:
        """Execute a dialogue pattern between two agents."""
        agents = branch.topology.agents
        if len(agents) != 2:
            raise ValueError(f"Dialogue requires exactly 2 agents, got {len(agents)}")
        
        agent1, agent2 = agents
        current_agent = branch.topology.entry_agent
        other_agent = agent2 if current_agent == agent1 else agent1
        
        current_request = context.initial_request
        execution_trace = []
        conversation_turns = 0
        
        while conversation_turns < branch.topology.max_iterations:
            # Execute current agent
            step_result = await self._execute_agent_step(
                current_agent,
                current_request,
                context,
                branch,
                conversation_context={
                    "turn": conversation_turns,
                    "partner": other_agent,
                    "pattern": "dialogue"
                }
            )
            
            execution_trace.append(step_result)
            branch.state.current_step += 1
            
            # Update branch execution trace for User node
            if not hasattr(branch, '_execution_trace'):
                branch._execution_trace = []
            branch._execution_trace.append(step_result)
            
            if not step_result.success:
                return BranchResult(
                    branch_id=branch.id,
                    success=False,
                    final_response=step_result.response,
                    total_steps=branch.state.current_step,
                    execution_trace=execution_trace,
                    branch_memory=context.branch_memory,
                    error=step_result.error
                )
            
            # Check if branch should wait for children
            if step_result.waiting_for_children:
                logger.info(f"Conversation branch '{branch.name}' pausing for child branches")
                # Store current conversation state
                self.branch_continuation[branch.id]["conversation_turns"] = conversation_turns
                self.branch_continuation[branch.id]["other_agent"] = other_agent
                return BranchResult(
                    branch_id=branch.id,
                    success=True,
                    final_response=None,
                    total_steps=branch.state.current_step,
                    execution_trace=execution_trace,
                    branch_memory=context.branch_memory,
                    metadata={
                        "waiting": True,
                        "waiting_for": step_result.child_branch_ids,
                        "conversation_turns": conversation_turns
                    }
                )
            
            # Check for conversation end signals
            if step_result.action_type == "end_conversation":
                return BranchResult(
                    branch_id=branch.id,
                    success=True,
                    final_response=step_result.response,
                    total_steps=branch.state.current_step,
                    execution_trace=execution_trace,
                    branch_memory=context.branch_memory,
                    metadata={"conversation_turns": conversation_turns}
                )
            
            # Swap agents for next turn
            current_request = step_result.response
            current_agent, other_agent = other_agent, current_agent
            conversation_turns += 1
            
            # Update branch topology
            branch.topology.current_agent = current_agent
            branch.topology.conversation_turns = conversation_turns
        
        # Max turns reached
        return BranchResult(
            branch_id=branch.id,
            success=True,
            final_response=execution_trace[-1].response if execution_trace else None,
            total_steps=branch.state.current_step,
            execution_trace=execution_trace,
            branch_memory=context.branch_memory,
            metadata={
                "conversation_turns": conversation_turns,
                "max_turns_reached": True
            }
        )
    
    async def _execute_debate_pattern(
        self,
        branch: ExecutionBranch,
        context: BranchExecutionContext
    ) -> BranchResult:
        """Execute a debate pattern with multiple agents."""
        # TODO: Implement debate pattern
        # This would involve multiple agents taking turns with a moderator
        raise NotImplementedError("Debate pattern not yet implemented")
    
    async def _execute_interview_pattern(
        self,
        branch: ExecutionBranch,
        context: BranchExecutionContext
    ) -> BranchResult:
        """Execute an interview pattern with interviewer and interviewee."""
        # TODO: Implement interview pattern
        # This would have asymmetric roles (interviewer asks, interviewee responds)
        raise NotImplementedError("Interview pattern not yet implemented")
    
    async def _execute_nested_branch(
        self,
        branch: ExecutionBranch,
        context: BranchExecutionContext
    ) -> BranchResult:
        """
        Execute a nested branch containing sub-branches.
        
        This allows for complex hierarchical execution patterns.
        """
        # TODO: Implement nested branch execution
        # This would spawn and manage sub-branches
        raise NotImplementedError("Nested branches not yet implemented")
    
    async def _execute_agent_step(
        self,
        agent_name: str,
        request: Any,
        context: BranchExecutionContext,
        branch: ExecutionBranch,
        conversation_context: Optional[Dict[str, Any]] = None
    ) -> StepResult:
        """Execute a single agent step."""
        logger.debug(f"Executing agent '{agent_name}' in branch '{branch.name}'")
        
        # Special handling for User node
        if agent_name.lower() == "user":
            if self.step_executor and self.step_executor.user_node_handler:
                # Prepare the request with message if available
                user_request = request
                
                # If request is a dict with message field, use it directly
                if isinstance(request, dict) and "message" in request:
                    user_request = request
                # If request is a dict with request field (from agent_requests)
                elif isinstance(request, dict) and "request" in request:
                    user_request = request["request"]
                # If request is a string, wrap it
                elif isinstance(request, str):
                    user_request = {"message": request}
                    
                # Add from_agent info if we have it
                if self._last_agent_name:
                    if isinstance(user_request, dict):
                        user_request["from_agent"] = self._last_agent_name
                    else:
                        user_request = {
                            "message": str(user_request),
                            "from_agent": self._last_agent_name
                        }
                
                # Store execution trace on branch for UserNodeHandler
                if not hasattr(branch, '_execution_trace'):
                    branch._execution_trace = []
                
                # Pass branch object in context
                context_with_branch = {
                    **context.shared_context,
                    "branch": branch,
                    "branch_id": context.branch_id,
                    "session_id": context.session_id,
                    "step_number": branch.state.current_step,
                    "execution_trace": branch._execution_trace
                }
                
                result = await self.step_executor.execute_step(
                    agent="User",  # Pass as string
                    request=user_request,
                    memory=[],  # Empty memory - let agent use its own state
                    context=context_with_branch
                )
                
                # Store result for next iteration
                self._last_step_result = result
                self._last_agent_name = agent_name
                
                return result
            else:
                # Check for error recovery request without handler
                if isinstance(request, dict) and request.get('error_recovery'):
                    # Error recovery requested but no handler - log error details and fail
                    error_details = request.get('error_details', {})
                    logger.error(f"Error recovery requested but User node handler not configured. Error: {error_details.get('message', 'Unknown error')}")
                    return StepResult(
                        agent_name="User",
                        success=False,
                        error=f"Cannot handle error recovery - User node handler not configured. Original error: {error_details.get('message', 'Unknown error')}",
                        metadata={"error_details": error_details}
                    )

                # Check if this is an auto-injected User node
                if branch.metadata.get("auto_injected_user"):
                    # Check if communication manager is available in context
                    comm_manager = context.shared_context.get('communication_manager')
                    config = context.shared_context.get('auto_run_config')  # Get AutoRunConfig if available

                    if comm_manager:
                        # Use proper user interaction even for auto-injected nodes
                        # This allows auto_run to work with user interactions seamlessly
                        from ..communication.user_node_handler import UserNodeHandler
                        handler = UserNodeHandler(comm_manager, self.event_bus)
                        return await handler.handle_user_node(
                            branch=branch,
                            incoming_message=request,
                            context=context.shared_context
                        )
                    elif config and config.user_interaction.warn_on_missing_handler:
                        # Emit warning through logger if configured
                        logger.warning(
                            "User node reached but no communication manager configured. "
                            "Consider setting user_interaction='terminal' in auto_run. "
                            "Auto-completing user interaction."
                        )
                        # Auto-complete for backward compatibility
                        return StepResult(
                            agent_name="User",
                            success=True,
                            response=request if request else {"message": "Auto-injected User node completed"},
                            metadata={"auto_completed": True, "warning": "No handler configured"}
                        )
                    else:
                        # Silent auto-complete for backward compatibility
                        return StepResult(
                            agent_name="User",
                            success=True,
                            response=request if request else {"message": "Auto-injected User node completed"},
                            metadata={"auto_completed": True}
                        )
                else:
                    # Explicit User node without handler is an error
                    return StepResult(
                        agent_name="User",
                        success=False,
                        error="User node handler not configured for explicit User node"
                    )
        
        # Get agent instance for normal agents (pool-aware)
        agent = self.agent_registry.get_or_acquire(agent_name, branch.id)
        if not agent:
            return StepResult(
                agent_name=agent_name,
                success=False,
                error=f"Agent '{agent_name}' not found in registry"
            )
        
        try:
            # CHECK PRE-EXECUTION RULES
            if self.rules_engine:
                rule_context = RuleContext(
                    rule_type=RuleType.PRE_EXECUTION,
                    session_id=context.session_id,
                    branch=branch,
                    agent_name=agent_name,
                    current_step=branch.state.current_step,
                    total_steps=branch.state.total_steps,
                    elapsed_time=time.time() - (branch.state.start_time or time.time()),
                    active_agents=len(branch.topology.agents),
                    active_branches=1,  # TODO: Get from branch spawner
                    metadata={
                        "request": request,
                        "conversation_context": conversation_context,
                        "branch_type": branch.type.value,
                    },
                    branch_metadata=branch.metadata  # For rule state persistence
                )
                
                allow_execution, pre_results = await self.rules_engine.check_pre_execution(rule_context)
                
                if not allow_execution:
                    blocking_rules = [r for r in pre_results if r.should_block]
                    logger.warning(f"Pre-execution blocked by rules: {[r.rule_name for r in blocking_rules]}")
                    return StepResult(
                        agent_name=agent_name,
                        success=False,
                        error=f"Execution blocked by rules: {', '.join([r.rule_name for r in blocking_rules])}"
                    )
            
            # If we have a step executor, use it
            if self.step_executor:
                # Check if this is a tool continuation (agent continuing with itself after tools)
                is_tool_continuation = (
                    self._last_agent_name == agent_name and
                    self._last_step_result and
                    hasattr(self._last_step_result, 'metadata') and
                    self._last_step_result.metadata.get('tool_continuation')
                )
                
                result = await self.step_executor.execute_step(
                    agent=agent,
                    request=request,
                    memory=[],  # Empty memory - let agent use its own state
                    context={
                        **context.shared_context,
                        "branch_id": context.branch_id,
                        "session_id": context.session_id,
                        "step_number": branch.state.current_step,
                        "conversation": conversation_context,
                        "tool_continuation": is_tool_continuation,
                        "branch": branch,  # CRITICAL: Pass branch for reflexive metadata
                        "topology_graph": self.topology_graph,  # CRITICAL: Pass topology for next agents
                        "metadata": context.metadata  # CRITICAL: Pass metadata for retry tracking
                    }
                )
            else:
                # Direct execution (fallback)
                response = await agent.run_step(
                    request,
                    context={
                        "branch_id": context.branch_id,
                        "session_id": context.session_id
                    }
                )
                
                # Create step result
                result = StepResult(
                    agent_name=agent_name,
                    success=True,
                    response=response,
                    action_type="continue"
                )
            
            # Use memory updates from step result to preserve complete message sequence
            if hasattr(result, 'memory_updates') and result.memory_updates:
                # Add each memory update to branch memory
                for memory_update in result.memory_updates:
                    # Ensure timestamp exists
                    if 'timestamp' not in memory_update:
                        memory_update['timestamp'] = time.time()
                    context.add_memory(agent_name, memory_update)
            else:
                # Fallback for backward compatibility
                context.add_memory(
                    agent_name,
                    {
                        "role": "assistant",
                        "content": result.response,
                        "name": agent_name,
                        "timestamp": time.time(),
                    },
                )

            # Validate response if validator available
            # Skip validation if this is a tool continuation (tools already executed)
            if self.response_validator and result.success and not (result.tool_results and result.metadata.get('tool_continuation')):
                # Create a mock ExecutionState for validation
                from ..branches.types import ExecutionState
                exec_state = ExecutionState(
                    session_id=context.session_id,
                    current_step=branch.state.current_step,
                    status="running"
                )
                
                validation = await self.response_validator.process_response(
                    raw_response=result.parsed_response if result.response is None and result.parsed_response else result.response,
                    agent=agent,
                    branch=branch,
                    exec_state=exec_state
                )
                
                if validation.is_valid:
                    result.action_type = validation.action_type.value if validation.action_type else "continue"
                    result.parsed_response = validation.parsed_response

                    # Handle error recovery routing - transfer error details to metadata
                    if validation.action_type in [ActionType.ERROR_RECOVERY, ActionType.TERMINAL_ERROR]:
                        # Extract error details
                        error_details = {}
                        if validation.invocations and len(validation.invocations) > 0:
                            user_invocation = validation.invocations[0]
                            if hasattr(user_invocation, 'request'):
                                error_details = user_invocation.request.get('error_details', {})

                        # Transfer error details to result metadata
                        if not result.metadata:
                            result.metadata = {}
                        result.metadata['error_details'] = error_details
                        result.metadata['error_action_type'] = validation.action_type.value

                        # Mark for error recovery routing - _execute_simple_branch will handle User node availability check
                        result.next_agent = "User"  # Attempt to route to User
                        result.metadata['needs_error_display'] = True  # Flag for graceful fallback if User not available

                    # Handle auto-retry for timeout/network errors
                    elif validation.action_type == ActionType.AUTO_RETRY:
                        # Extract error info and mark for retry
                        parsed = validation.parsed_response or {}
                        error_info = parsed.get('error_info', {})

                        result.success = False
                        result.requires_retry = True  # Trigger retry logic
                        result.error = error_info.get('message', 'API error - retrying')
                        if not result.metadata:
                            result.metadata = {}
                        result.metadata.update({
                            'auto_retry': True,
                            'error_classification': error_info.get('classification'),
                            'retry_after': error_info.get('retry_after', 3)
                        })
                        logger.info(f"Auto-retry scheduled for {agent_name}: {error_info.get('classification')}")

                    # Handle parallel invocation
                    elif validation.action_type == ActionType.PARALLEL_INVOKE:
                        logger.info(f"Agent '{agent_name}' requested parallel invocation")
                        return await self._handle_parallel_invocation(
                            agent_name, validation, result, context, branch
                        )
                    
                    # Handle single agent invocation
                    if validation.next_agents and len(validation.next_agents) > 0:
                        result.next_agent = validation.next_agents[0]
                else:
                    result.success = False
                    result.error = validation.error_message
                    result.requires_retry = True

                    # FIX: Send validation error back to agent as user message
                    if hasattr(agent, 'memory') and validation.error_message:
                        error_message = f"Your response format was invalid. {validation.error_message}"
                        if validation.retry_suggestion:
                            error_message += f"\n{validation.retry_suggestion}"

                        # Add error message to agent's memory
                        agent.memory.add(
                            role="user",
                            content=error_message
                        )
                        logger.debug(f"Added validation error to {agent_name}'s memory for retry")
            
            # Apply FLOW_CONTROL rules if we have a rules engine
            if self.rules_engine and result.success:
                # Create rule context
                rule_context = RuleContext(
                    rule_type=RuleType.FLOW_CONTROL,
                    session_id=context.session_id,
                    branch=branch,
                    agent_name=agent_name,
                    current_step=branch.state.current_step,
                    total_steps=branch.state.total_steps,
                    elapsed_time=time.time() - (branch.state.start_time or time.time()),
                    active_agents=len(branch.topology.agents),
                    active_branches=1,  # TODO: Get from branch spawner
                    metadata={
                        "action_type": result.action_type,
                        "current_agent": agent_name,
                        "target_agent": result.next_agent,
                        "branch_type": branch.type.value,
                    },
                    branch_metadata=branch.metadata  # Important for AlternatingAgentRule
                )
                
                # Check rules
                rule_results = await self.rules_engine.apply_flow_control(
                    rule_context
                )
                
                # Apply modifications from rules
                for rule_result in rule_results:
                    if rule_result.action == "modify" and rule_result.modifications:
                        # Handle override_next_agent
                        if "override_next_agent" in rule_result.modifications:
                            result.next_agent = rule_result.modifications["override_next_agent"]
                            logger.info(f"Rule {rule_result.rule_name} overrode next agent to {result.next_agent}")
                        
                        # Handle override_action_type
                        if "override_action_type" in rule_result.modifications:
                            result.action_type = rule_result.modifications["override_action_type"]
                            logger.info(f"Rule {rule_result.rule_name} overrode action type to {result.action_type}")
                        
                        # Handle state updates with proper persistence
                        if "update_state" in rule_result.modifications:
                            # Update branch metadata (persistent across steps)
                            branch.metadata.update(rule_result.modifications["update_state"])
                            
                            # Also update context metadata for immediate use
                            context.metadata.update(rule_result.modifications["update_state"])
                            
                            # Log state changes for debugging
                            logger.debug(f"Rule {rule_result.rule_name} updated branch state: {rule_result.modifications['update_state']}")
                        
                        
                        # Handle forced completion
                        if rule_result.modifications.get("force_completion"):
                            result.should_end_branch = True
                            result.action_type = "final_response"
                            logger.info(f"Rule {rule_result.rule_name} forced completion: {rule_result.modifications.get('completion_reason')}")
            
            # CHECK POST_EXECUTION RULES
            if self.rules_engine and result.success:
                post_context = RuleContext(
                    rule_type=RuleType.POST_EXECUTION,
                    session_id=context.session_id,
                    branch=branch,
                    agent_name=agent_name,
                    current_step=branch.state.current_step,
                    total_steps=branch.state.total_steps,
                    elapsed_time=time.time() - (branch.state.start_time or time.time()),
                    active_agents=len(branch.topology.agents),
                    active_branches=1,  # TODO: Get from branch spawner
                    metadata={
                        "response": result.response,
                        "action_type": result.action_type,
                        "next_agent": result.next_agent,
                        "branch_type": branch.type.value,
                        "step_result": result,  # Pass the complete step result for rules to access
                    },
                    branch_metadata=branch.metadata
                )
                
                post_results = await self.rules_engine.check_post_execution(
                    post_context,
                    result
                )
                
                # Apply any post-execution modifications
                for rule_result in post_results:
                    if rule_result.action == "terminate":
                        result.should_end_branch = True
                        result.action_type = "final_response"
                        logger.warning(f"Post-execution termination by {rule_result.rule_name}")
                    
                    # Apply other modifications if any
                    if rule_result.modifications:
                        if "update_state" in rule_result.modifications:
                            branch.metadata.update(rule_result.modifications["update_state"])
                            context.metadata.update(rule_result.modifications["update_state"])
                            logger.debug(f"Post-execution rule {rule_result.rule_name} updated state: {rule_result.modifications['update_state']}")
            
            # Store result for potential User node invocation
            self._last_step_result = result
            self._last_agent_name = agent_name
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing agent '{agent_name}': {e}")
            return StepResult(
                agent_name=agent_name,
                success=False,
                error=str(e),
                requires_retry=True
            )
    
    async def _should_complete(
        self,
        branch: ExecutionBranch,
        context: BranchExecutionContext,
        execution_trace: List[StepResult]
    ) -> bool:
        """Check if branch should complete based on completion condition."""
        condition = branch.completion_condition
        
        if isinstance(condition, MaxStepsCompletion):
            return branch.state.current_step >= condition.max_steps
        
        elif isinstance(condition, ConversationTurnsCompletion):
            turns = branch.topology.conversation_turns or 0
            return turns >= condition.max_turns
        
        elif isinstance(condition, AgentDecidedCompletion):
            # Check if last step indicated completion
            if execution_trace and execution_trace[-1].action_type in ["final_response", "end_conversation"]:
                # But not if there's a next_agent override (e.g., from reflexive rule)
                if not execution_trace[-1].next_agent:
                    return True
        
        # Check max steps as safety
        max_steps = branch.topology.metadata.get("max_steps", 30)
        return branch.state.current_step >= max_steps
    
    async def _should_hold_for_convergence(
        self,
        current_agent: str,
        next_agent: str,
        branch_id: str
    ) -> Tuple[bool, Optional[str], Optional['ExecutionBranch']]:
        """
        Determine if execution should hold at a convergence point.
        Returns: (should_hold, reason, continuation_branch)
        
        Now checks for parallel invocation groups, not just active branches.
        Includes deadlock detection with timeout.
        
        Returns:
            (should_hold, reason)
        """
        if not self.topology_graph or not self.branch_spawner:
            return False, None, None

        # Check if next agent is a convergence point
        node = self.topology_graph.get_node(next_agent)
        if not (node and hasattr(node, 'is_convergence_point') and node.is_convergence_point):
            return False, None, None
        
        # IMPORTANT: Check if this branch is part of a parallel invocation group
        if hasattr(self.branch_spawner, 'branch_to_group'):
            group_id = self.branch_spawner.branch_to_group.get(branch_id)
            if group_id:
                group = self.branch_spawner.parallel_groups.get(group_id)
                if group and (next_agent in group.shared_convergence_points or next_agent in group.sub_group_convergences):
                    # Check for timeout/deadlock
                    from datetime import datetime
                    if hasattr(group, 'created_at'):
                        age_seconds = (datetime.now() - group.created_at).total_seconds()
                        # Use timeout from branch_spawner (default 600s = 10 minutes)
                        timeout = getattr(self.branch_spawner, 'group_timeout_seconds', 600)
                        if age_seconds > timeout:
                            # NEW: Check convergence policy
                            policy = getattr(self.branch_spawner, 'convergence_policy', None)
                            if policy is None:
                                # Fallback to old behavior for backward compatibility
                                from ...coordination.config import ConvergencePolicyConfig
                                policy = ConvergencePolicyConfig.from_value(0.67)

                            # Count how many branches reached THIS convergence point
                            arrived_at_target = group.get_arrived_at_convergence(next_agent)
                            arrived_count = len(arrived_at_target)
                            total_count = len(group.branch_ids)
                            arrival_ratio = arrived_count / total_count if total_count > 0 else 0.0

                            # Check if minimum ratio met
                            min_ratio = policy.min_ratio

                            if arrival_ratio >= min_ratio:
                                # Quorum met: allow partial convergence
                                log_method = getattr(logger, policy.log_level, logger.warning)
                                log_method(
                                    f"⚠️  Group '{group_id}' convergence timeout after {age_seconds:.1f}s. "
                                    f"Proceeding with PARTIAL convergence at '{next_agent}': "
                                    f"{arrived_count}/{total_count} branches ready ({arrival_ratio:.0%} ≥ {min_ratio:.0%}). "
                                    f"Missing branches will be excluded."
                                )

                                # Terminate orphaned branches if policy requires it
                                if policy.terminate_orphans:
                                    orphaned_branches = set(group.branch_ids) - arrived_at_target
                                    if orphaned_branches:
                                        await self._terminate_orphaned_branches(
                                            group_id=group_id,
                                            convergence_point=next_agent,
                                            orphaned_branch_ids=orphaned_branches
                                        )

                                return False, f"Partial convergence: {arrived_count}/{total_count} arrivals", None

                            else:
                                # Quorum NOT met
                                if policy.on_insufficient == "proceed":
                                    # Risky: proceed anyway
                                    logger.error(
                                        f"❌ Group '{group_id}' convergence timeout after {age_seconds:.1f}s. "
                                        f"Only {arrived_count}/{total_count} branches at '{next_agent}' "
                                        f"({arrival_ratio:.0%} < {min_ratio:.0%}), but policy allows proceeding. "
                                        f"⚠️  WARNING: Results may be incomplete!"
                                    )

                                    # Still terminate orphans
                                    if policy.terminate_orphans:
                                        orphaned_branches = set(group.branch_ids) - arrived_at_target
                                        if orphaned_branches:
                                            await self._terminate_orphaned_branches(
                                                group_id=group_id,
                                                convergence_point=next_agent,
                                                orphaned_branch_ids=orphaned_branches
                                            )

                                    return False, f"Forced convergence: {arrived_count}/{total_count} arrivals (below threshold)", None

                                elif policy.on_insufficient == "user":
                                    # TODO: Ask user for confirmation (future enhancement)
                                    # For now, fail
                                    logger.error(
                                        f"❌ Group '{group_id}' convergence timeout after {age_seconds:.1f}s. "
                                        f"Only {arrived_count}/{total_count} branches at '{next_agent}' "
                                        f"({arrival_ratio:.0%} < {min_ratio:.0%}). "
                                        f"User confirmation not yet implemented - failing workflow."
                                    )
                                    from ...agents.exceptions import WorkflowTimeoutError
                                    raise WorkflowTimeoutError(
                                        f"Convergence quorum not met: {arrived_count}/{total_count} < {min_ratio:.0%}",
                                        timeout_seconds=timeout,
                                        context={
                                            "group_id": group_id,
                                            "convergence_point": next_agent,
                                            "arrived_branches": list(arrived_at_target),
                                            "total_branches": total_count,
                                            "arrival_ratio": arrival_ratio,
                                            "required_ratio": min_ratio
                                        }
                                    )

                                else:  # "fail"
                                    # Strict: fail workflow
                                    logger.error(
                                        f"❌ Group '{group_id}' convergence FAILED after {age_seconds:.1f}s. "
                                        f"Only {arrived_count}/{total_count} branches at '{next_agent}' "
                                        f"({arrival_ratio:.0%} < {min_ratio:.0%}). Workflow cannot proceed."
                                    )
                                    from ...agents.exceptions import WorkflowTimeoutError
                                    raise WorkflowTimeoutError(
                                        f"Convergence quorum not met: {arrived_count}/{total_count} < {min_ratio:.0%}",
                                        timeout_seconds=timeout,
                                        context={
                                            "group_id": group_id,
                                            "convergence_point": next_agent,
                                            "arrived_branches": list(arrived_at_target),
                                            "total_branches": total_count,
                                            "arrival_ratio": arrival_ratio,
                                            "required_ratio": min_ratio
                                        }
                                    )

                    # Track continuation branch to return
                    continuation_branch = None

                    # Check if we need to create a continuation branch at this convergence point
                    # This happens when the convergence point is different from the parent agent
                    if next_agent != group.requesting_agent:
                        # Check if continuation already exists for this convergence point
                        if next_agent not in group.convergence_branches_spawned:
                            # This is the first branch reaching a non-parent convergence point
                            # Create a continuation branch that will wait for remaining children
                            cont_branch_id = f"continuation_{next_agent}_{group.group_id[:8]}"

                            # Continuation must wait for ALL branches in the group
                            # At this point NO branches have completed yet (including current branch
                            # which is still executing). All branches must complete before convergence.
                            continuation_waiting_for = set(group.branch_ids)

                            # Create the continuation branch with proper topology
                            cont_branch = ExecutionBranch(
                                id=cont_branch_id,
                                name=f"Continuation: {next_agent}",
                                type=BranchType.SIMPLE,
                                topology=BranchTopology(
                                    agents=[next_agent],
                                    entry_agent=next_agent,
                                    current_agent=next_agent,  # Critical: set for resumption
                                    allowed_transitions=self.topology_graph.get_subgraph_from(next_agent) if self.topology_graph else {}
                                ),
                                state=BranchState(status=BranchStatus.WAITING, current_step=0),
                                metadata={
                                    "is_continuation": True,
                                    "convergence_node": next_agent,
                                    "group_id": group.group_id,
                                    "parent_branch_id": group.parent_branch_id,
                                    "requesting_agent": group.requesting_agent,
                                    "is_convergence_branch": True
                                }
                            )

                            # Register the continuation branch in branch_spawner
                            self.branch_spawner.branch_info[cont_branch_id] = cont_branch

                            # Set up waiting mechanism (reuse existing infrastructure!)
                            # The continuation waits for ALL children in the parallel group
                            self.branch_spawner.waiting_branches[cont_branch_id] = continuation_waiting_for

                            # Set up parent-child mapping for result aggregation
                            self.branch_spawner.parent_child_map[cont_branch_id] = list(group.branch_ids)

                            # Track that we created continuation for this convergence point
                            group.convergence_branches_spawned.add(next_agent)

                            # Store the continuation branch to return it
                            continuation_branch = cont_branch

                            logger.info(
                                f"Created continuation branch '{cont_branch_id}' at convergence '{next_agent}' "
                                f"(parent was '{group.requesting_agent}') waiting for {len(continuation_waiting_for)} children: "
                                f"{continuation_waiting_for}"
                            )

                    # Persist convergence state on the branch before holding
                    # This ensures the state survives retries and validation failures
                    current_branch = self.branch_spawner.branch_info.get(branch_id)
                    if current_branch:
                        current_branch.metadata['reached_convergence'] = True
                        current_branch.metadata['convergence_point'] = next_agent
                        current_branch.metadata['convergence_timestamp'] = time.time()
                        logger.debug(
                            f"Persisted convergence state for branch '{branch_id}': "
                            f"convergence_point='{next_agent}'"
                        )

                    # This branch is part of a group heading to this convergence point
                    pending = group.get_pending_count()
                    if pending > 0:
                        logger.info(
                            f"Branch '{branch_id}' holding at convergence '{next_agent}' - "
                            f"waiting for {pending} more branches from group '{group_id}'"
                        )
                        return True, f"Part of parallel group '{group_id}', {pending} branches pending", continuation_branch
        
        # Fallback: Check for other active branches (for non-group scenarios)
        active_branches = self.branch_spawner.active_branches
        
        # Exclude current branch
        other_branches = {bid: task for bid, task in active_branches.items() 
                         if bid != branch_id and not task.done()}
        
        if not other_branches:
            # No other active branches - can proceed
            logger.info(f"Branch '{branch_id}' is only active branch - proceeding to convergence point '{next_agent}'")
            return False, None, None

        # Check if any other branch could reach this convergence point
        could_converge_here = []

        for other_branch_id in other_branches:
            # Get current position of other branch
            other_branch_info = self.branch_spawner.branch_info.get(other_branch_id)
            if not other_branch_info:
                continue

            current_position = other_branch_info.topology.current_agent
            if not current_position:
                continue

            # Check if this branch could reach the convergence point
            if self.topology_graph.can_reach(current_position, next_agent):
                could_converge_here.append(other_branch_id)
                logger.debug(f"Branch '{other_branch_id}' at '{current_position}' could reach '{next_agent}'")

        if not could_converge_here:
            # No other branches could reach this convergence point
            logger.info(f"No other branches can reach convergence point '{next_agent}' - proceeding")
            return False, None, None

        # Should hold - other branches could converge here
        logger.info(f"Branch '{branch_id}' holding at convergence point '{next_agent}' - "
                   f"waiting for {len(could_converge_here)} other branches")

        # Track which branches could arrive
        if hasattr(self.branch_spawner, 'convergence_tracker'):
            self.branch_spawner.convergence_tracker.potential_arrivals[next_agent] = \
                set(could_converge_here + [branch_id])

        return True, f"Waiting for {len(could_converge_here)} branches: {could_converge_here}", None
    
    async def _determine_next_agent(
        self,
        current_agent: str,
        step_result: StepResult,
        allowed_transitions: Dict[str, List[str]],
        branch_id: Optional[str] = None
    ) -> Optional[str]:
        """Determine the next agent based on step result and allowed transitions."""
        # If step result specifies next agent
        if step_result.next_agent:
            # SPECIAL CASE: Self-continuation
            # Use centralized helper to check if agents are the same
            is_self_continuation = self.agent_registry.are_same_agent(
                step_result.next_agent,
                current_agent
            )

            if is_self_continuation:
                # Check if this is a valid self-continuation scenario
                if hasattr(step_result, 'metadata'):
                    metadata = step_result.metadata
                    if (metadata.get('tool_continuation') or
                        metadata.get('invalid_response') or
                        metadata.get('has_tool_calls') or
                        metadata.get('has_tool_results')):
                        logger.debug(f"Allowing self-continuation for '{current_agent}' (tools/retry)")
                        return current_agent  # Return pool name

                # Otherwise, check if self-loops are allowed in topology
                allowed = allowed_transitions.get(current_agent, [])
                if current_agent in allowed:
                    return current_agent
                else:
                    logger.warning(f"Agent '{current_agent}' attempted self-invocation without valid reason")
                    return None
            
            # Normal validation for other agents
            allowed = allowed_transitions.get(current_agent, [])
            if step_result.next_agent in allowed:
                next_agent = step_result.next_agent  # Store instead of returning immediately
            else:
                logger.warning(f"Agent '{current_agent}' tried to invoke '{step_result.next_agent}' "
                             f"but it's not in allowed transitions: {allowed}")
                return None
        else:
            # Default: take first allowed transition
            allowed = allowed_transitions.get(current_agent, [])
            next_agent = allowed[0] if allowed else None
        
        # Check if we should hold for convergence (only if branch_id provided and next_agent determined)
        if next_agent and branch_id:
            should_hold, reason, continuation_branch = await self._should_hold_for_convergence(
                current_agent, next_agent, branch_id
            )

            if should_hold:
                # Store convergence target and request data
                step_result.convergence_target = next_agent
                step_result.hold_reason = reason

                # Store continuation branch if one was created
                if continuation_branch:
                    step_result.continuation_branch = continuation_branch
                    step_result.metadata['continuation_branch'] = continuation_branch

                # Add to pending requests
                if hasattr(self.branch_spawner, 'convergence_tracker'):
                    request_data = self._prepare_convergence_request(step_result)
                    self.branch_spawner.convergence_tracker.add_pending_request(
                        next_agent, branch_id, current_agent, request_data
                    )

                return None  # Stop branch execution
        
        return next_agent

    async def _terminate_orphaned_branches(
        self,
        group_id: str,
        convergence_point: str,
        orphaned_branch_ids: Set[str]
    ) -> int:
        """
        Terminate branches that didn't reach convergence before timeout.

        When a parallel group times out and proceeds with partial convergence,
        this method cancels the remaining branches that didn't converge,
        preventing wasted computation on work that won't be used.

        Args:
            group_id: Parallel group identifier
            convergence_point: Expected convergence point name
            orphaned_branch_ids: Set of branch IDs that didn't reach convergence

        Returns:
            Number of branches successfully terminated
        """
        terminated_count = 0

        for branch_id in orphaned_branch_ids:
            try:
                # Check if branch has an active task
                active_task = self.branch_spawner.active_branches.get(branch_id)

                if active_task and not active_task.done():
                    # Cancel the asyncio task
                    active_task.cancel()
                    logger.warning(
                        f"🛑 Terminated orphaned branch '{branch_id}' "
                        f"(didn't reach '{convergence_point}' before timeout)"
                    )
                    terminated_count += 1

                    # Wait briefly for cancellation to propagate
                    try:
                        await asyncio.wait_for(active_task, timeout=1.0)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        pass  # Expected

                    # Remove from active branches
                    self.branch_spawner.active_branches.pop(branch_id, None)

                    # Mark branch as abandoned in branch info
                    if branch_id in self.branch_spawner.branch_info:
                        branch = self.branch_spawner.branch_info[branch_id]
                        if branch.metadata is None:
                            branch.metadata = {}
                        branch.metadata["abandoned"] = True
                        branch.metadata["abandonment_reason"] = "convergence_timeout"
                        branch.metadata["expected_convergence"] = convergence_point

                    # Try to release agent/pool resources
                    if branch_id in self.branch_spawner.branch_info:
                        branch = self.branch_spawner.branch_info[branch_id]
                        agent_name = branch.metadata.get("current_agent") if branch.metadata else None

                        if agent_name:
                            # Check if agent is from a pool
                            from ...agents.registry import AgentRegistry
                            agent_or_pool = AgentRegistry.get_instance(agent_name)

                            # If it's a pool, release the instance
                            if agent_or_pool and hasattr(agent_or_pool, 'release'):
                                try:
                                    agent_or_pool.release(branch_id)
                                    logger.debug(f"Released pool instance for terminated branch '{branch_id}'")
                                except Exception as e:
                                    logger.warning(f"Failed to release pool instance for '{branch_id}': {e}")

                else:
                    logger.debug(
                        f"Branch '{branch_id}' already completed or not active, skipping termination"
                    )

            except Exception as e:
                logger.error(
                    f"Error terminating orphaned branch '{branch_id}': {e}",
                    exc_info=True
                )

        if terminated_count > 0:
            logger.info(
                f"✂️  Terminated {terminated_count} orphaned branch(es) from group '{group_id}' "
                f"that didn't reach convergence point '{convergence_point}'"
            )

        return terminated_count

    def _prepare_convergence_request(self, step_result: StepResult) -> Any:
        """Prepare request data for convergence point."""
        # Extract the actual content to send to convergence point
        if step_result.parsed_response and isinstance(step_result.parsed_response, dict):
            # Use the parsed response action_input as the request
            action_input = step_result.parsed_response.get("action_input")
            if action_input:
                return action_input
        
        # Fallback to response content
        return step_result.response
    
    def _prepare_next_request(
        self,
        step_result: StepResult,
        context: BranchExecutionContext
    ) -> Any:
        """Prepare request for next agent including saved context."""
        
        # CASE 0a: Retry failed step - use original request from retry context
        if step_result.action_type == "retry_failed_step":
            retry_context = step_result.metadata.get("retry_context", {})
            original_request = retry_context.get("request")
            if original_request:
                return original_request
            # Fallback to normal flow if no request in context

        # CASE 0b: Error recovery or terminal error - pass error details to User
        if step_result.action_type in ["error_recovery", "terminal_error"]:
            # Pass the full error details metadata to User node
            return {
                "error_recovery": step_result.action_type == "error_recovery",
                "error_type": "terminal" if step_result.action_type == "terminal_error" else "fixable",
                "error_details": step_result.metadata.get("error_details", {}),
                "retry_context": step_result.metadata.get("retry_context", {}),
                "failed_agent": step_result.metadata.get("failed_agent"),
                "message": step_result.metadata.get("error_details", {}).get("message", "An error occurred"),
                "suggested_actions": step_result.metadata.get("error_details", {}).get("suggested_actions", [])
            }

        # CASE 1: Error from previous step - don't propagate
        if not step_result.success and step_result.error:
            # Special handling for invalid response errors
            if hasattr(step_result, 'metadata') and step_result.metadata.get('invalid_response'):
                # Return the error which contains format instructions
                return step_result.error
            # Regular error - return clean message
            error_msg = f"Previous agent '{step_result.agent_name}' encountered an error. Please proceed with your task."
            logger.warning(f"Preventing error propagation from {step_result.agent_name}: {step_result.error}")
            return error_msg
        
        # CASE 2: Tool continuation - no additional message needed
        if hasattr(step_result, 'metadata') and step_result.metadata.get('tool_continuation'):
            # Tool results are already in memory from step_executor
            # No additional continuation message needed
            return None
        
        # CASE 3: Invalid response retry - return format error
        if hasattr(step_result, 'metadata') and step_result.metadata.get('invalid_response'):
            if step_result.error:
                return step_result.error  # This already contains format instructions
            else:
                return "Your previous response was not in the expected format. Please provide a valid JSON response."
        
        # CASE 4: Normal continuation
        base_request = self._get_base_request(step_result)

        # Validation logging for debugging
        if isinstance(base_request, list):
            logger.warning(f"Base request is a list, might be malformed: {base_request}")
        elif isinstance(base_request, dict) and "agent_name" in base_request:
            logger.warning(f"Base request looks like an invocation dict: {base_request}")

        # Prepend caller agent name to request content (for agent-to-agent invocations)
        base_request = self._prepend_caller_name(base_request, step_result.agent_name)

        # FIX: Check for saved_context instead of context_selection
        if not hasattr(step_result, 'saved_context') or not step_result.saved_context:
            return base_request

        # Format and include saved context
        return self._include_context_in_request(
            base_request,
            step_result.saved_context,
            step_result.agent_name
        )
    
    def _get_base_request(self, step_result: StepResult) -> Any:
        """Extract base request from step result (existing logic)."""
        if step_result.action_type == "final_response":
            if step_result.parsed_response:
                return (step_result.parsed_response.get("content") or 
                       step_result.parsed_response.get("final_response") or 
                       step_result.response)
            return step_result.response

        # Handle invoke_agent with invocation array
        if step_result.parsed_response and "action_input" in step_result.parsed_response:

            action_input = step_result.parsed_response["action_input"]

            # Check if action_input is an invocation array (reflexive case)
            if isinstance(action_input, list) and len(action_input) > 0:
                first_item = action_input[0]
                if isinstance(first_item, dict) and "request" in first_item:
                    # This is an invocation array - extract the request
                    return first_item.get("request", "")

            # Normal case - action_input is the direct request
            return action_input

        return step_result.response

    def _prepend_caller_name(self, request: Any, caller_name: str) -> Any:
        """
        Prepend the caller agent's name to the request content.
        This makes it clear to the receiving agent who is requesting the task.

        Args:
            request: The request content (can be string, dict, or other)
            caller_name: Name of the agent making the request

        Returns:
            Modified request with caller name prepended
        """
        if not request or not caller_name:
            return request

        # Skip prepending for None or empty requests (e.g., tool continuations)
        if request is None or request == "":
            return request

        caller_prefix = f"[Request from {caller_name}]\n"

        # Handle string requests
        if isinstance(request, str):
            return f"{caller_prefix}{request}"

        # Handle dict requests
        elif isinstance(request, dict):
            # Make a copy to avoid mutating the original
            request_copy = request.copy()

            # If there's a 'prompt' or 'task' field, prepend to it
            if 'prompt' in request_copy:
                if isinstance(request_copy['prompt'], str):
                    request_copy['prompt'] = f"{caller_prefix}{request_copy['prompt']}"
            elif 'task' in request_copy:
                if isinstance(request_copy['task'], str):
                    request_copy['task'] = f"{caller_prefix}{request_copy['task']}"
            elif 'message' in request_copy:
                if isinstance(request_copy['message'], str):
                    request_copy['message'] = f"{caller_prefix}{request_copy['message']}"
            else:
                # No standard field found, add a 'from_agent' field instead
                request_copy['from_agent'] = caller_name

            return request_copy

        # For other types (list, etc.), return as-is
        # Lists are typically handled at a higher level (parallel invocations)
        return request

    def _include_context_in_request(
        self, 
        base_request: Any, 
        context_selection: Dict[str, Any],
        from_agent: str
    ) -> Any:
        """Include saved context in the request to next agent."""
        
        # Format context as readable text
        context_text = self._format_context_for_agent(context_selection, from_agent)
        
        # Handle different request types
        if isinstance(base_request, str):
            # String request - append context
            return f"{base_request}\n\n{context_text}"
        
        elif isinstance(base_request, dict):
            # Dict request - add context fields
            base_request = base_request.copy()
            base_request["passed_context"] = context_selection
            base_request["context_summary"] = context_text
            return base_request
        
        elif isinstance(base_request, list):
            # Array format (for parallel invocations)
            updated_requests = []
            for item in base_request:
                if isinstance(item, dict):
                    item = item.copy()
                    if "request" in item:
                        # Update the nested request
                        item["request"] = self._include_context_in_request(
                            item["request"], 
                            context_selection,
                            from_agent
                        )
                    updated_requests.append(item)
                else:
                    updated_requests.append(item)
            return updated_requests
        
        # Fallback - return with context appended
        return f"{base_request}\n\n{context_text}"
    
    def _format_context_for_agent(
        self, 
        context_selection: Dict[str, Any],
        from_agent: str
    ) -> str:
        """Format saved context for readable inclusion."""
        lines = [f"[Saved Context from {from_agent}]"]
        
        for key, messages in context_selection.items():
            lines.append(f"\n### {key}")
            
            # Show first few messages
            for i, msg in enumerate(messages[:3]):
                role = msg.get('role', 'unknown')
                content = str(msg.get('content', ''))
                
                # Truncate long content
                if len(content) > 300:
                    content = content[:300] + "..."
                
                # Format based on role
                if role == 'tool':
                    name = msg.get('name', 'unknown_tool')
                    lines.append(f"{i+1}. Tool [{name}]: {content}")
                else:
                    lines.append(f"{i+1}. {role.title()}: {content}")
            
            if len(messages) > 3:
                lines.append(f"... and {len(messages) - 3} more messages")
        
        return "\n".join(lines)
    
    def _update_metrics(
        self,
        branch_type: BranchType,
        result: BranchResult,
        start_time: float
    ) -> None:
        """Update execution metrics."""
        duration = time.time() - start_time
        metrics = self.execution_metrics[branch_type.value]
        
        metrics["total_executions"] += 1
        if result.success:
            metrics["successful_executions"] += 1
        
        # Update averages
        prev_avg_steps = metrics["average_steps"]
        prev_avg_duration = metrics["average_duration"]
        n = metrics["total_executions"]
        
        metrics["average_steps"] = (prev_avg_steps * (n - 1) + result.total_steps) / n
        metrics["average_duration"] = (prev_avg_duration * (n - 1) + duration) / n
    
    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get execution metrics."""
        return dict(self.execution_metrics)
    
    def get_last_agent(self, result: BranchResult) -> Optional[str]:
        """Get the last agent that executed in a branch."""
        if result.execution_trace:
            return result.execution_trace[-1].agent_name
        return None
    
    async def _handle_parallel_invocation(
        self,
        agent_name: str,
        validation: ValidationResult,
        result: StepResult,
        context: BranchExecutionContext,
        branch: ExecutionBranch
    ) -> StepResult:
        """Handle agent-initiated parallel invocation."""
        
        # Extract target agents from validation
        target_agents = []
        if validation.invocations:
            target_agents = [inv.agent_name for inv in validation.invocations]
        
        logger.info(f"Agent '{agent_name}' initiating parallel invocation of {target_agents}")
        
        # CRITICAL: Pre-analyze topology BEFORE deciding to wait
        should_parent_wait = True  # Default
        shared_conv = set()
        sub_conv = {}
        
        if self.branch_spawner and target_agents:
            try:
                should_parent_wait, shared_conv, sub_conv = await self.branch_spawner.analyze_parallel_invocation(
                    agent_name, target_agents
                )
            except Exception as e:
                logger.warning(f"Failed to analyze parallel invocation: {e}")
                should_parent_wait = True  # Default to safe behavior
        
        # Store analysis for branch_spawner to use later
        branch.metadata["parallel_analysis"] = {
            "should_parent_wait": should_parent_wait,
            "shared_convergence": list(shared_conv),
            "sub_group_convergence": {k: list(v) for k, v in sub_conv.items()},
            "target_agents": target_agents
        }
        
        # Store in instance variable for continuation
        self.branch_continuation[branch.id] = {
            "agent_name": agent_name,
            "context": context,
            "branch": branch,
            "invocations": validation.invocations,
            "parsed_response": validation.parsed_response
        }
        
        if should_parent_wait:
            # Traditional behavior - parent waits
            logger.info(f"Parent branch '{branch.id}' entering waiting state")
            
            branch.state.status = BranchStatus.WAITING
            self.waiting_for_children[branch.id] = set(target_agents)
            
            return StepResult(
                agent_name=agent_name,
                success=True,
                response=f"Parent waiting for parallel execution of {len(target_agents)} agents",
                action_type=ActionType.PARALLEL_INVOKE,
                parsed_response=validation.parsed_response,
                waiting_for_children=True,
                child_branch_ids=target_agents
            )
        else:
            # New behavior - parent completes
            logger.info(f"Parent branch '{branch.id}' completing after spawning")
            
            return StepResult(
                agent_name=agent_name,
                success=True,
                response=f"Initiated parallel execution of {len(target_agents)} agents",
                action_type=ActionType.PARALLEL_INVOKE,
                parsed_response=validation.parsed_response,
                waiting_for_children=False,
                should_end_branch=True  # Branch completes
            )
    
    async def resume_branch(
        self,
        branch_id: str,
        aggregated_results: Dict[str, Any]
    ) -> BranchResult:
        """
        Resume a branch that was waiting for child branches.
        
        Args:
            branch_id: ID of the branch to resume
            aggregated_results: Results from child branches
            
        Returns:
            BranchResult from continued execution
        """
        logger.info(f"Resuming branch '{branch_id}' with aggregated results")
        
        # Get continuation state
        continuation = self.branch_continuation.get(branch_id)
        if not continuation:
            logger.error(f"No continuation state found for branch '{branch_id}'")
            return BranchResult(
                branch_id=branch_id,
                success=False,
                final_response=None,
                total_steps=0,
                error="No continuation state found"
            )
        
        # Extract state
        agent_name = continuation["agent_name"]
        context = continuation["context"]
        branch = continuation["branch"]
        
        # Update branch state
        branch.state.status = BranchStatus.RUNNING
        
        # Add aggregated results to context
        context.shared_context["child_results"] = aggregated_results
        
        # FIX 3: Check if parent should flow to convergence
        current_agent = agent_name
        if 'convergence_data' in branch.metadata:
            conv_data = branch.metadata['convergence_data']
            # Update the request to include convergence data
            resume_request = {
                "aggregated_requests": conv_data['aggregated_requests'],
                "source_count": conv_data['source_count'],
                "is_convergence": True,
                "resumed_from_parallel": True
            }
            # Set current agent to convergence point
            current_agent = conv_data['target']
            logger.info(f"Parent resuming to convergence point '{current_agent}' with {conv_data['source_count']} aggregated results")
        else:
            # Create a synthetic request with child results
            # Extract request data from invocations if present
            original_request = continuation.get("parsed_response", {})

            # Convert AgentInvocation objects to clean request data
            if "invocations" in original_request and original_request["invocations"]:
                # Extract request data from AgentInvocation objects
                invocation_summary = []
                for inv in original_request["invocations"]:
                    if hasattr(inv, "agent_name"):  # It's an AgentInvocation object
                        invocation_summary.append({"agent": inv.agent_name, "request": inv.request})
                    else:  # It's already a dict (shouldn't happen with our fix)
                        invocation_summary.append(
                            {
                                "agent": inv.get("agent_name", "unknown"),
                                "request": inv.get("request", {}),
                            }
                        )

                resume_request = {
                    "original_invocations": invocation_summary,
                    "child_results": aggregated_results,
                    "resumed_from_parallel": True,
                }
            else:
                # No invocation data, just pass child results
                resume_request = {
                    "child_results": aggregated_results,
                    "resumed_from_parallel": True,
                }

        # Continue execution from where we left off
        try:
            # Execute the next step after aggregation
            if branch.type == BranchType.SIMPLE:
                # Continue simple branch execution
                result = await self._continue_simple_branch(
                    branch, context, current_agent, resume_request
                )
            elif branch.type == BranchType.CONVERSATION:
                # Continue conversation branch
                result = await self._continue_conversation_branch(
                    branch, context, agent_name, resume_request
                )
            else:
                raise ValueError(f"Unsupported branch type for resumption: {branch.type}")
            
            # Clean up continuation state
            del self.branch_continuation[branch_id]
            if branch_id in self.waiting_for_children:
                del self.waiting_for_children[branch_id]
            
            return result
            
        except Exception as e:
            logger.error(f"Error resuming branch '{branch_id}': {e}")
            branch.state.status = BranchStatus.FAILED
            return BranchResult(
                branch_id=branch_id,
                success=False,
                final_response=None,
                total_steps=branch.state.current_step,
                error=str(e)
            )
    
    async def _continue_simple_branch(
        self,
        branch: ExecutionBranch,
        context: BranchExecutionContext,
        current_agent: str,
        resume_request: Any
    ) -> BranchResult:
        """Continue a simple branch after resumption."""
        execution_trace = []

        # Track retry attempts per agent (use branch's retry_counts for persistence)
        if not hasattr(branch, 'retry_counts'):
            branch.retry_counts = defaultdict(int)
        retry_counts = branch.retry_counts

        # Continue from current agent with aggregated results
        current_request = resume_request

        while True:
            # Check completion condition
            if await self._should_complete(branch, context, execution_trace):
                break

            # Pass actual retry count in context metadata (same as _execute_simple_branch)
            context.metadata["agent_retry_count"] = retry_counts.get(current_agent, 0)

            # Determine next agent (might be the same agent continuing)
            step_result = await self._execute_agent_step(
                current_agent,
                current_request,
                context,
                branch
            )
            
            execution_trace.append(step_result)
            branch.state.current_step += 1
            
            # Update branch execution trace for User node
            if not hasattr(branch, '_execution_trace'):
                branch._execution_trace = []
            branch._execution_trace.append(step_result)

            if not step_result.success:
                # Handle failure with retry logic (same as _execute_simple_branch)
                if step_result.requires_retry and retry_counts[current_agent] < self.max_retries:
                    retry_counts[current_agent] += 1
                    logger.warning(f"Retrying agent '{current_agent}' after failure (attempt {retry_counts[current_agent]}/{self.max_retries})")
                    current_request = None  # Use None for retry to let agent continue
                    continue  # Continue the while loop for retry
                else:
                    if retry_counts[current_agent] >= self.max_retries:
                        logger.error(f"Max retries ({self.max_retries}) reached for agent '{current_agent}'")
                    return BranchResult(
                        branch_id=branch.id,
                        success=False,
                        final_response=step_result.response,
                        total_steps=branch.state.current_step,
                        execution_trace=execution_trace,
                        branch_memory=context.branch_memory,
                        error=step_result.error
                    )

            # Check if this is a final response
            if step_result.action_type == "final_response":
                # Extract the actual content from parsed response if available
                final_content = step_result.response
                if step_result.parsed_response and isinstance(step_result.parsed_response, dict):
                    # Check for final_response field in parsed data
                    if "final_response" in step_result.parsed_response:
                        final_content = step_result.parsed_response["final_response"]
                    elif "content" in step_result.parsed_response:
                        final_content = step_result.parsed_response["content"]
                
                return BranchResult(
                    branch_id=branch.id,
                    success=True,
                    final_response=final_content,
                    total_steps=branch.state.current_step,
                    execution_trace=execution_trace,
                    branch_memory=context.branch_memory
                )
            
            # Check if branch should wait again
            if step_result.waiting_for_children:
                # Store updated continuation state
                self.branch_continuation[branch.id]["parsed_response"] = step_result.parsed_response
                return BranchResult(
                    branch_id=branch.id,
                    success=True,
                    final_response=None,
                    total_steps=branch.state.current_step,
                    execution_trace=execution_trace,
                    branch_memory=context.branch_memory,
                    metadata={"waiting": True}
                )
            
            # Determine next agent
            next_agent = await self._determine_next_agent(
                current_agent,
                step_result,
                branch.topology.allowed_transitions,
                branch.id  # Pass branch ID for convergence checking
            )

            # Debug log the agent transition
            logger.debug(f"Agent transition: {current_agent} → {next_agent or 'END'} in branch {branch.id}")

            if not next_agent:
                # Check if stopped at convergence
                convergence_target = getattr(step_result, 'convergence_target', None)
                if convergence_target:
                    # Branch reached a convergence point
                    logger.info(f"Branch '{branch.id}' reached convergence point '{convergence_target}'")

                    # Include continuation branch in metadata if one was created
                    result_metadata = {
                        "reached_convergence": True,
                        "convergence_point": convergence_target,
                        "last_agent": current_agent,
                        "hold_reason": getattr(step_result, 'hold_reason', None)
                    }
                    continuation_branch = getattr(step_result, 'continuation_branch', None)
                    if continuation_branch:
                        result_metadata['continuation_branch'] = continuation_branch

                    # The branch completes here - group handles aggregation
                    return BranchResult(
                        branch_id=branch.id,
                        success=True,
                        final_response=step_result.response,
                        total_steps=branch.state.current_step,
                        execution_trace=execution_trace,
                        branch_memory=context.branch_memory,
                        metadata=result_metadata
                    )

                # Check if this was an error recovery attempt that failed due to no User node
                if step_result.metadata and step_result.metadata.get('needs_error_display'):
                    # User node was attempted but not available - log error and fail gracefully
                    error_details = step_result.metadata.get('error_details', {})
                    error_type = step_result.metadata.get('error_action_type', 'error')
                    error_message = error_details.get('message', 'Unknown error')
                    error_classification = error_details.get('classification', 'unknown')

                    # Log error (following existing pattern from line 716)
                    logger.error(
                        f"[{error_type.upper()}] {error_classification}: {error_message}. "
                        f"User node not available in topology. Terminating workflow."
                    )

                    # Return failed branch result with error details
                    return BranchResult(
                        branch_id=branch.id,
                        success=False,
                        final_response=None,
                        total_steps=branch.state.current_step,
                        execution_trace=execution_trace,
                        branch_memory=context.branch_memory,
                        error=f"[{error_type}] {error_classification}: {error_message}",
                        metadata=step_result.metadata
                    )

                # No next agent - branch completes
                # Extract the actual content from parsed response if available
                final_content = step_result.response
                if step_result.parsed_response and isinstance(step_result.parsed_response, dict):
                    # Check for final_response field in parsed data
                    if "final_response" in step_result.parsed_response:
                        final_content = step_result.parsed_response["final_response"]
                    elif "content" in step_result.parsed_response:
                        final_content = step_result.parsed_response["content"]
                
                return BranchResult(
                    branch_id=branch.id,
                    success=True,
                    final_response=final_content,
                    total_steps=branch.state.current_step,
                    execution_trace=execution_trace,
                    branch_memory=context.branch_memory
                )
            
            # Prepare request for next agent
            current_request = self._prepare_next_request(step_result, context)
            logger.info(f"Transitioning from {current_agent} to {next_agent} with request: {current_request}")

            # Context metadata update removed - from_agent not needed anymore

            current_agent = next_agent
            branch.topology.current_agent = current_agent

        # If we exit the loop due to completion condition (e.g., max steps)
        # return the current state as the result
        return BranchResult(
            branch_id=branch.id,
            success=True,
            final_response=execution_trace[-1].response if execution_trace else None,
            total_steps=branch.state.current_step,
            execution_trace=execution_trace,
            branch_memory=context.branch_memory,
            metadata={"completion_reason": "max_steps_reached"},
        )

    async def _continue_conversation_branch(
        self,
        branch: ExecutionBranch,
        context: BranchExecutionContext,
        current_agent: str,
        resume_request: Any
    ) -> BranchResult:
        """Continue a conversation branch after resumption."""
        # For now, just delegate to simple branch continuation
        # Could add specialized conversation resumption logic here
        return await self._continue_simple_branch(branch, context, current_agent, resume_request)
    
    async def _release_branch_agents(self, branch_id: str) -> None:
        """
        Release any single agents allocated to this branch.
        
        Args:
            branch_id: ID of the branch whose agents should be released
        """
        if not hasattr(self, 'branch_spawner') or not self.branch_spawner:
            return

        # Check if branch has allocation and get agent name for logging
        agent_name = None
        async with self.branch_spawner.resource_lock:
            if branch_id in self.branch_spawner.agent_allocations:
                agent_name = self.branch_spawner.agent_allocations[branch_id].get("agent_name", "unknown")

        # Release outside the lock since _release_agent_for_branch modifies the dict
        if agent_name:
            released = self.branch_spawner._release_agent_for_branch(branch_id)
            if released:
                logger.debug(f"Released agent '{agent_name}' from completed branch '{branch_id}'")
