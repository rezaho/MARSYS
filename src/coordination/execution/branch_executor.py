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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING
from collections import defaultdict

from ...agents.registry import AgentRegistry

if TYPE_CHECKING:
    from .step_executor import StepExecutor
    from ..routing.router import Router
    from ..topology.graph import TopologyGraph
from ..branches.types import (
    ExecutionBranch,
    BranchType,
    BranchStatus,
    BranchResult,
    StepResult,
    ConversationPattern,
    CompletionCondition,
    AgentDecidedCompletion,
    MaxStepsCompletion,
    ConversationTurnsCompletion,
)
from ..validation.response_validator import ValidationProcessor, ActionType, ValidationResult
from ..rules.rules_engine import RulesEngine, RuleContext, RuleType

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
    
    def _update_agent_constraints(self, branch: ExecutionBranch) -> None:
        """
        Update agent constraints based on topology.
        
        This sets which agents can return final responses based on
        their connections to User nodes.
        """
        if not self.topology_graph:
            # Need reference to topology graph
            logger.warning("No topology_graph available for constraint setting")
            return
        
        # Get agents that can return final response
        agents_with_user_access = self.topology_graph.get_agents_with_user_access()
        
        # NEW: Check for agents in reflexive edges (they can also return final response)
        agents_in_reflexive_edges = set()
        for edge in self.topology_graph.edges:
            if edge.metadata.get("reflexive") or edge.metadata.get("pattern") == "boomerang":
                # Target of reflexive edge can return final response
                agents_in_reflexive_edges.add(edge.target)
        
        logger.info(f"Agents with User access: {agents_with_user_access}")
        logger.info(f"Agents in reflexive edges: {agents_in_reflexive_edges}")
        
        # Update each agent in the branch
        for agent_name in branch.topology.agents:
            # Skip User nodes - they're not real agents
            if agent_name == "User":
                continue
            agent = self.agent_registry.get(agent_name)
            if agent and hasattr(agent, 'set_topology_constraints'):
                # Agent can return final if connected to User OR is target of reflexive edge
                can_return_final = (
                    agent_name in agents_with_user_access or 
                    agent_name in agents_in_reflexive_edges
                )
                agent.set_topology_constraints({
                    "can_return_final_response": can_return_final
                })
                logger.info(f"Set constraints for {agent_name}: can_return_final={can_return_final}")
            else:
                logger.warning(f"Agent {agent_name} not found or doesn't support constraints")
    
    def _is_reflexive_branch(self, branch: ExecutionBranch) -> bool:
        """Check if this branch was created from a reflexive edge."""
        return branch.metadata.get("is_reflexive", False)
    
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
        
        # Set agent constraints based on topology
        self._update_agent_constraints(branch)
        
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
        
        This handles linear agent execution with potential tool usage.
        """
        logger.debug(f"Executing simple branch with agents: {branch.topology.agents}")
        
        current_request = context.initial_request
        execution_trace = []
        
        # Start with the entry agent
        current_agent = branch.topology.entry_agent
        
        # Track retry attempts per agent
        retry_counts = defaultdict(int)
        
        while True:
            # Check completion condition
            if await self._should_complete(branch, context, execution_trace):
                break
            
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
            
            # Check if this is a final response
            if step_result.action_type == "final_response" and not step_result.next_agent:
                # Only end if there's no next_agent override from rules
                # Extract the actual content from parsed response if available
                final_content = step_result.response
                if step_result.parsed_response and isinstance(step_result.parsed_response, dict):
                    # Check for final_response field in parsed data
                    if "final_response" in step_result.parsed_response:
                        final_content = step_result.parsed_response["final_response"]
                    elif "content" in step_result.parsed_response:
                        final_content = step_result.parsed_response["content"]
                
                # Check if this is a reflexive edge - need to route back to parent
                if self._is_reflexive_branch(branch):
                    logger.info(f"Reflexive branch completing - routing response back to parent")
                    # The branch spawner will handle routing this back
                    return BranchResult(
                        branch_id=branch.id,
                        success=True,
                        final_response=final_content,
                        total_steps=branch.state.current_step,
                        execution_trace=execution_trace,
                        branch_memory=context.branch_memory,
                        metadata={"reflexive_completion": True}
                    )
                
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
                branch.topology.allowed_transitions
            )
            
            if not next_agent:
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
                return StepResult(
                    agent_name="User",
                    success=False,
                    error="User node handler not configured"
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
                        "from_agent": context.metadata.get("from_agent"),
                        "conversation_context": conversation_context,
                        "is_reflexive_call": context.metadata.get("is_reflexive_call", False),
                        "branch_type": branch.type.value
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
                        "tool_continuation": is_tool_continuation
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
                context.add_memory(agent_name, {
                    "role": "assistant",
                    "content": result.response,
                    "name": agent_name,
                    "timestamp": time.time()
                })
            
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
                    
                    # Handle parallel invocation
                    if validation.action_type == ActionType.PARALLEL_INVOKE:
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
                    
                    # CRITICAL: Add retry flag to parsed_response so branch spawner sees it
                    if result.parsed_response is None:
                        result.parsed_response = {}
                    if isinstance(result.parsed_response, dict):
                        result.parsed_response['requires_retry'] = True
                        result.parsed_response['validation_error'] = validation.error_message
                    
                    # FIX: Send validation error back to agent as user message
                    if hasattr(agent, 'memory') and validation.error_message:
                        error_message = f"Invalid response format. {validation.error_message}"
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
                        "from_agent": context.metadata.get("from_agent"),
                        "target_agent": result.next_agent,
                        "is_reflexive_call": context.metadata.get("is_reflexive_call", False),
                        "branch_type": branch.type.value
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
                        
                        # Handle clear reflexive flag
                        if rule_result.modifications.get("clear_reflexive"):
                            context.metadata.pop("is_reflexive_call", None)
                            context.metadata.pop("from_agent", None)
                            logger.info(f"Rule {rule_result.rule_name} cleared reflexive metadata")
                        
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
                        "from_agent": context.metadata.get("from_agent"),
                        "is_reflexive_call": context.metadata.get("is_reflexive_call", False),
                        "branch_type": branch.type.value
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
    
    async def _determine_next_agent(
        self,
        current_agent: str,
        step_result: StepResult,
        allowed_transitions: Dict[str, List[str]]
    ) -> Optional[str]:
        """Determine the next agent based on step result and allowed transitions."""
        # If step result specifies next agent
        if step_result.next_agent:
            # SPECIAL CASE: Self-continuation
            # Allow agent to continue with itself for tool processing or retries
            if step_result.next_agent == current_agent:
                # Check if this is a valid self-continuation scenario
                if hasattr(step_result, 'metadata'):
                    metadata = step_result.metadata
                    if (metadata.get('tool_continuation') or 
                        metadata.get('invalid_response') or
                        metadata.get('has_tool_calls') or
                        metadata.get('has_tool_results')):
                        logger.debug(f"Allowing self-continuation for '{current_agent}' (tools/retry)")
                        return current_agent
                
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
                return step_result.next_agent
            else:
                logger.warning(f"Agent '{current_agent}' tried to invoke '{step_result.next_agent}' "
                             f"but it's not in allowed transitions: {allowed}")
                return None
        
        # Default: take first allowed transition
        allowed = allowed_transitions.get(current_agent, [])
        return allowed[0] if allowed else None
    
    def _prepare_next_request(
        self,
        step_result: StepResult,
        context: BranchExecutionContext
    ) -> Any:
        """Prepare request for next agent including saved context."""
        
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
        
        if (step_result.parsed_response and 
            "action_input" in step_result.parsed_response):
            return step_result.parsed_response["action_input"]
        
        return step_result.response
    
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
        """
        Handle agent-initiated parallel invocation.
        Sets branch to waiting state and returns special StepResult.
        """
        target_agents = validation.next_agents
        logger.info(f"Agent '{agent_name}' initiating parallel invocation of {target_agents}")
        
        # Store branch continuation state in branch metadata instead of self.branch_continuation
        branch.metadata["continuation_state"] = {
            "agent_name": agent_name,
            "context": context,
            "target_agents": target_agents,
            "parsed_response": validation.parsed_response
        }
        
        # Also store in instance variable for compatibility
        self.branch_continuation[branch.id] = {
            "agent_name": agent_name,
            "context": context,
            "branch": branch,
            "target_agents": target_agents,
            "parsed_response": validation.parsed_response
        }
        
        # Set branch to waiting state
        branch.state.status = BranchStatus.WAITING
        self.waiting_for_children[branch.id] = set(target_agents)
        
        # Create step result indicating branch should wait
        return StepResult(
            agent_name=agent_name,
            success=True,
            response=result.response,
            action_type="parallel_invoke",
            parsed_response=validation.parsed_response,
            waiting_for_children=True,
            child_branch_ids=target_agents,  # For now, using agent names as IDs
            should_end_branch=False  # Don't end the branch, just pause it
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
        
        # Create a synthetic request with child results
        resume_request = {
            "original_request": continuation.get("parsed_response", {}),
            "child_results": aggregated_results,
            "resumed_from_parallel": True
        }
        
        # Continue execution from where we left off
        try:
            # Execute the next step after aggregation
            if branch.type == BranchType.SIMPLE:
                # Continue simple branch execution
                result = await self._continue_simple_branch(
                    branch, context, agent_name, resume_request
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
        
        # Continue from current agent with aggregated results
        current_request = resume_request
        
        while True:
            # Check completion condition
            if await self._should_complete(branch, context, execution_trace):
                break
            
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
                # Also update branch metadata
                branch.metadata["continuation_state"] = {
                    "agent_name": current_agent,
                    "context": context,
                    "target_agents": step_result.child_branch_ids,
                    "parsed_response": step_result.parsed_response
                }
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
                branch.topology.allowed_transitions
            )
            
            if not next_agent:
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