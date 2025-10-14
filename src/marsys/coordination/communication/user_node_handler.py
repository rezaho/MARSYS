"""
Handler for User node execution within the coordination system.
"""

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from ..branches.types import StepResult, BranchStatus
from .core import CommunicationMode, UserInteraction
from .manager import CommunicationManager

logger = logging.getLogger(__name__)


class UserNodeHandler:
    """
    Handles execution when control reaches a User node.

    This component manages the pause/resume flow, tracks calling agents,
    and determines where execution should continue after user interaction.
    """

    def __init__(self, communication_manager: CommunicationManager, event_bus=None):
        self.communication_manager = communication_manager
        self.event_bus = event_bus
        self.mode_handlers = {
            CommunicationMode.SYNC: self._handle_sync_mode,
            CommunicationMode.ASYNC_PUBSUB: self._handle_async_pubsub_mode,
            CommunicationMode.ASYNC_QUEUE: self._handle_async_queue_mode
        }
        logger.info("UserNodeHandler initialized")
    
    async def handle_user_node(
        self,
        branch: 'ExecutionBranch',
        incoming_message: Any,
        context: Dict[str, Any]
    ) -> StepResult:
        """
        Handle execution at User node with appropriate communication mode.
        
        Args:
            branch: The execution branch
            incoming_message: Message/request from the calling agent
            context: Execution context
            
        Returns:
            StepResult indicating next action
        """
        # Determine communication mode
        mode = self._determine_communication_mode(context)
        
        # Extract calling agent from execution trace
        calling_agent = self._get_calling_agent(branch)
        
        # Determine where to resume after user response
        resume_agent = self._determine_resume_agent(branch, calling_agent)
        
        # Store context for trace access
        self._current_context = context
        
        logger.info(f"User node handling - Branch: {branch.id}, "
                   f"Calling agent: {calling_agent}, "
                   f"Resume agent: {resume_agent}, "
                   f"Mode: {mode}")
        
        # Check for error recovery or terminal error first
        if isinstance(incoming_message, dict):
            if incoming_message.get('error_recovery'):
                # Fixable error - offer retry options
                return await self._handle_fixable_error(
                    branch,
                    incoming_message.get('error_details', {}),
                    context
                )
            elif incoming_message.get('error_type') == 'terminal':
                # Terminal error - display only
                return await self._handle_terminal_error(
                    branch,
                    incoming_message.get('error_details', {}),
                    context
                )

        # Extract message from incoming_message
        display_message = incoming_message
        
        # Special handling for System/entry point interactions
        if calling_agent == "System":
            # Check for custom user prompt in context
            if "user_prompt" in context:
                display_message = context["user_prompt"]
            elif isinstance(incoming_message, dict):
                display_message = incoming_message.get('message', incoming_message.get('content', str(incoming_message)))
            else:
                # Format the task nicely for user display
                display_message = self._format_task_for_user(incoming_message)
        else:
            # Regular agent-to-user interaction
            # Handle both string and dict formats
            if isinstance(incoming_message, str):
                display_message = incoming_message
            elif isinstance(incoming_message, dict):
                # Check if this is an error response
                if 'error' in incoming_message or 'error_type' in incoming_message:
                    # Don't display raw error messages to user
                    display_message = "I encountered an issue processing your request. Please try rephrasing or provide more specific information."
                else:
                    display_message = incoming_message.get('message', 
                                      incoming_message.get('content',
                                      incoming_message.get('request', str(incoming_message))))
            
            # Final check for empty or invalid messages
            if not display_message or (isinstance(display_message, str) and not display_message.strip()):
                display_message = "Please provide your input or response."
        
        # Emit lightweight status event for monitoring (metadata only, no content)
        if self.event_bus:
            # Import here to avoid circular dependency
            from ..status.events import UserInteractionEvent

            # Status event shows WHAT is happening, not the full content
            status_event = UserInteractionEvent(
                session_id=context.get("session_id", str(uuid.uuid4())),
                branch_id=branch.id,
                agent_name=calling_agent,
                interaction_type="starting",  # Just metadata
                prompt=f"Awaiting user response to {calling_agent}",  # Summary, not full content
                options=None  # Don't duplicate options in status
            )
            await self.event_bus.emit(status_event)

        # Create interaction with full context for actual dialogue
        interaction = UserInteraction(
            interaction_id=str(uuid.uuid4()),
            branch_id=branch.id,
            session_id=context.get("session_id", str(uuid.uuid4())),
            incoming_message=display_message,  # Full content here for dialogue
            interaction_type=self._determine_interaction_type(incoming_message),
            timestamp=time.time(),
            communication_mode=mode,

            # Agent tracing
            calling_agent=calling_agent,
            resume_agent=resume_agent,
            execution_trace=self._get_execution_summary(branch),

            # Context for resumption
            branch_context={
                "memory_snapshot": self._get_memory_snapshot(branch),
                "step_number": getattr(branch.state, 'step_count', 0),
                "topology_info": self._get_topology_info(branch)
            },

            # Additional context
            metadata=context
        )
        
        # Store interaction context in branch for resumption
        self._store_interaction_context(branch, interaction)

        # Handle based on mode
        handler = self.mode_handlers.get(mode, self._handle_sync_mode)
        result = await handler(interaction, context, branch)

        # Check if this is user-first mode initial response that needs task combination
        if (result.success and
            result.response and
            interaction.interaction_type == "initial_query" and
            context.get("pending_task")):
            result = self._combine_pending_task_with_user_response(result, context)

        return result
    
    def _combine_pending_task_with_user_response(self, result: StepResult, context: Dict[str, Any]) -> StepResult:
        """
        Combine the pending task with user response in user-first mode.
        This ensures the first agent receives both the original task and user input.

        Args:
            result: The StepResult from the user interaction
            context: The execution context containing pending_task

        Returns:
            Modified StepResult with combined task and user response
        """
        pending_task = context.get("pending_task")
        user_response = result.response

        # Combine task with user response based on task type
        if isinstance(pending_task, dict):
            # Add user_response as a new key to the existing task dict
            combined_response = {**pending_task, "user_response": user_response}
        else:
            # Create new dict with task and user_response keys
            combined_response = {
                "initial_task": pending_task,
                "user_response": user_response
            }

        logger.info(f"User-first mode: Combined task with user response for next agent")

        # Remove pending_task from context after using it
        context.pop("pending_task", None)

        # Update the result with the combined response
        result.response = combined_response

        return result

    def _determine_communication_mode(self, context: Dict[str, Any]) -> CommunicationMode:
        """Determine communication mode from context."""
        mode_str = context.get("communication_mode", "sync")
        
        try:
            return CommunicationMode(mode_str)
        except ValueError:
            logger.warning(f"Unknown communication mode: {mode_str}, defaulting to SYNC")
            return CommunicationMode.SYNC
    
    def _get_calling_agent(self, branch: 'ExecutionBranch') -> str:
        """Determine which agent called the User node."""
        result = None
        
        try:
            # Look at execution trace from branch or context
            execution_trace = []
            
            # First try branch._execution_trace (set by BranchExecutor)
            if hasattr(branch, '_execution_trace') and branch._execution_trace:
                execution_trace = branch._execution_trace
            # Then try from context
            elif hasattr(self, '_current_context') and self._current_context:
                execution_trace = self._current_context.get('execution_trace', [])
            
            # Find last non-User agent
            for i in range(len(execution_trace) - 1, -1, -1):
                step = execution_trace[i]
                if hasattr(step, 'agent_name') and step.agent_name != "User":
                    result = step.agent_name
                    break
            
            # Fallback: check current agent in branch state
            if not result and hasattr(branch, 'state') and hasattr(branch.state, 'calling_agent') and branch.state.calling_agent:
                result = branch.state.calling_agent
            
            # Another fallback: check topology current agent
            if not result and hasattr(branch, 'topology') and hasattr(branch.topology, 'current_agent'):
                current = branch.topology.current_agent
                if current and current != "User":
                    result = current
        except Exception as e:
            logger.error(f"Error getting calling agent: {e}")
        
        # For entry point User nodes, return "System"
        if not result or result == "Unknown":
            logger.info("User node called with no calling agent - treating as entry point")
            return "System"
        
        return result
    
    def _determine_resume_agent(
        self,
        branch: 'ExecutionBranch',
        calling_agent: str
    ) -> str:
        """Determine where to resume after User response."""
        try:
            # Check topology for User's outgoing edges
            if hasattr(branch, 'topology') and hasattr(branch.topology, 'allowed_transitions'):
                user_transitions = branch.topology.allowed_transitions.get("User", [])
                
                # If User has edge back to calling agent (bidirectional)
                if calling_agent in user_transitions:
                    return calling_agent
                
                # Otherwise, find the next agent in topology
                for next_agent in user_transitions:
                    if next_agent != calling_agent:
                        return next_agent
            
            # Check if there's a specific resume path in branch metadata
            if hasattr(branch, 'metadata'):
                resume_path = branch.metadata.get("user_resume_path", {})
                if calling_agent in resume_path:
                    return resume_path[calling_agent]
        
        except Exception as e:
            logger.error(f"Error determining resume agent: {e}")
        
        # Default: return to calling agent
        return calling_agent
    
    def _determine_interaction_type(self, message: Any) -> str:
        """Determine the type of interaction based on message content."""
        # Handle None message
        if message is None:
            return "input"
            
        # Check context for explicit type (NEW)
        if hasattr(self, '_current_context') and self._current_context:
            if "user_interaction_type" in self._current_context:
                return self._current_context["user_interaction_type"]
        
        if isinstance(message, dict):
            # Check for explicit type
            if "interaction_type" in message:
                return message["interaction_type"]
            
            # Infer from content
            if "options" in message:
                return "choice"
            elif "confirm" in message or "confirmation" in message:
                return "confirmation"
            elif "notify" in message or "notification" in message:
                return "notification"
        
        return "question"
    
    def _get_execution_summary(self, branch: 'ExecutionBranch') -> List[Dict[str, Any]]:
        """Get summary of execution trace for context."""
        summary = []
        try:
            execution_trace = []
            
            # Try to get execution trace from branch or context
            if hasattr(branch, '_execution_trace') and branch._execution_trace:
                execution_trace = branch._execution_trace
            elif hasattr(self, '_current_context') and self._current_context:
                execution_trace = self._current_context.get('execution_trace', [])
            
            # Get last 5 steps
            for step in execution_trace[-5:]:
                summary.append({
                    "agent": getattr(step, 'agent_name', 'Unknown'),
                    "action": getattr(step, 'action_type', 'Unknown'),
                    "success": getattr(step, 'success', False)
                })
        except Exception as e:
            logger.error(f"Error getting execution summary: {e}")
        
        return summary
    
    def _get_memory_snapshot(self, branch: 'ExecutionBranch') -> List[Dict[str, Any]]:
        """Get current memory snapshot from branch."""
        try:
            if hasattr(branch, 'get_memory'):
                return branch.get_memory()
            elif hasattr(branch, 'state') and hasattr(branch.state, 'memory'):
                return branch.state.memory
        except Exception as e:
            logger.error(f"Error getting memory snapshot: {e}")
        
        return []
    
    def _get_topology_info(self, branch: 'ExecutionBranch') -> Dict[str, Any]:
        """Extract relevant topology information."""
        info = {}
        try:
            if hasattr(branch, 'topology'):
                topology = branch.topology
                info['agents'] = getattr(topology, 'agents', [])
                if hasattr(topology, 'allowed_transitions'):
                    info['user_transitions'] = topology.allowed_transitions.get('User', [])
        except Exception as e:
            logger.error(f"Error getting topology info: {e}")
        
        return info
    
    def _store_interaction_context(
        self,
        branch: 'ExecutionBranch',
        interaction: UserInteraction
    ) -> None:
        """Store interaction context in branch for later resumption."""
        try:
            if hasattr(branch, 'state'):
                branch.state.awaiting_user_response = True
                branch.state.interaction_id = interaction.interaction_id
                branch.state.calling_agent = interaction.calling_agent
                branch.state.resume_agent = interaction.resume_agent
                branch.state.user_wait_start_time = time.time()  # Start tracking wait time
                branch.state.interaction_context = {
                    "timestamp": interaction.timestamp,
                    "mode": interaction.communication_mode.value
                }
        except Exception as e:
            logger.error(f"Error storing interaction context: {e}")
    
    async def _handle_sync_mode(
        self,
        interaction: UserInteraction,
        context: Dict[str, Any],
        branch: 'ExecutionBranch'
    ) -> StepResult:
        """Handle synchronous interaction - wait for response."""
        try:
            # Send interaction and wait for response
            response = await self.communication_manager.handle_interaction(interaction)
            
            logger.info(f"Received sync response for interaction {interaction.interaction_id}")
            
            # Clear waiting state and record wait time
            if hasattr(branch, 'state'):
                branch.state.awaiting_user_response = False
                wait_time = 0
                if branch.state.user_wait_start_time:
                    wait_time = time.time() - branch.state.user_wait_start_time
                    branch.state.total_user_wait_time += wait_time
                    branch.state.user_wait_start_time = None
                    logger.debug(f"User wait time: {wait_time:.2f}s, Total: {branch.state.total_user_wait_time:.2f}s")

            # Emit completion status event
            if self.event_bus:
                from ..status.events import UserInteractionEvent
                complete_event = UserInteractionEvent(
                    session_id=interaction.session_id,
                    branch_id=branch.id,
                    agent_name=interaction.calling_agent,
                    interaction_type="completed",  # Just metadata
                    prompt=f"User responded to {interaction.calling_agent}",  # Summary only
                    options=None
                )
                await self.event_bus.emit(complete_event)
            
            return StepResult(
                agent_name="User",
                response=response,
                action_type="user_response",
                parsed_response={
                    "content": response,
                    "resume_agent": interaction.resume_agent,
                    "calling_agent": interaction.calling_agent,
                    "interaction_id": interaction.interaction_id
                },
                next_agent=interaction.resume_agent,
                success=True
            )
            
        except asyncio.TimeoutError:
            logger.warning(f"Timeout in sync interaction {interaction.interaction_id}")
            return StepResult(
                agent_name="User",
                response=None,
                action_type="timeout",
                success=False,
                error="User response timeout"
            )
        except Exception as e:
            logger.error(f"Error in sync interaction: {e}")
            return StepResult(
                agent_name="User",
                response=None,
                action_type="error",
                success=False,
                error=str(e)
            )
    
    async def _handle_async_pubsub_mode(
        self,
        interaction: UserInteraction,
        context: Dict[str, Any],
        branch: 'ExecutionBranch'
    ) -> StepResult:
        """Handle async pub/sub interaction - return immediately."""
        try:
            # Publish interaction (non-blocking)
            await self.communication_manager.handle_interaction(interaction)
            
            logger.info(f"Published async interaction {interaction.interaction_id}")
            
            # Mark branch as waiting for async response
            if hasattr(branch, 'state'):
                branch.state.status = BranchStatus.PAUSED
            
            return StepResult(
                agent_name="User",
                response={"status": "pending", "interaction_id": interaction.interaction_id},
                action_type="async_pending",
                parsed_response={
                    "interaction_id": interaction.interaction_id,
                    "branch_paused": True,
                    "mode": "async_pubsub",
                    "topic": interaction.topic
                },
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error in async pub/sub interaction: {e}")
            return StepResult(
                agent_name="User",
                response=None,
                action_type="error",
                success=False,
                error=str(e)
            )
    
    async def _handle_async_queue_mode(
        self,
        interaction: UserInteraction,
        context: Dict[str, Any],
        branch: 'ExecutionBranch'
    ) -> StepResult:
        """Handle async queue-based interaction."""
        try:
            # Queue interaction
            await self.communication_manager.handle_interaction(interaction)
            
            logger.info(f"Queued async interaction {interaction.interaction_id}")
            
            # Mark branch as waiting
            if hasattr(branch, 'state'):
                branch.state.status = BranchStatus.PAUSED
            
            return StepResult(
                agent_name="User",
                response={"status": "queued", "interaction_id": interaction.interaction_id},
                action_type="async_queued",
                parsed_response={
                    "interaction_id": interaction.interaction_id,
                    "branch_paused": True,
                    "mode": "async_queue",
                    "queue": interaction.queue_name
                },
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error in async queue interaction: {e}")
            return StepResult(
                agent_name="User",
                response=None,
                action_type="error",
                success=False,
                error=str(e)
            )
    
    def _format_task_for_user(self, task: Any) -> str:
        """Format a task description for user-friendly display."""
        if isinstance(task, str):
            # If it's a multi-line task, indent it nicely
            lines = task.strip().split('\n')
            if len(lines) > 1:
                return "Task Description:\n" + "\n".join(f"  {line}" for line in lines) + "\n\nPlease provide your input:"
            else:
                return f"Task: {task}\n\nPlease provide your input:"
        else:
            return f"Task: {str(task)}\n\nPlease provide your input:"

    async def _handle_fixable_error(
        self,
        branch: 'ExecutionBranch',
        error_info: Dict[str, Any],
        context: Dict[str, Any]
    ) -> StepResult:
        """Handle fixable error with retry options."""

        # Format for display with retry options
        formatted_message = self._format_fixable_error(error_info)

        # Create interaction
        interaction = UserInteraction(
            interaction_id=str(uuid.uuid4()),
            branch_id=branch.id,
            session_id=context.get("session_id"),
            incoming_message=formatted_message,
            interaction_type="error_recovery",
            communication_mode=CommunicationMode.SYNC,
            calling_agent=error_info.get('failed_agent', 'Unknown'),
            metadata={
                'error_info': error_info,
                'retry_context': context.get('retry_context')
            }
        )

        # Get user response through CommunicationManager
        if not self.communication_manager:
            logger.error(f"CommunicationManager not available for error recovery. Error details: {error_info}")
            # Return abort result since we can't interact with user
            return StepResult(
                agent_name="User",
                response={"action": "abort"},
                action_type="abort_execution",
                success=False,
                error=f"Cannot handle error recovery - no communication manager. Original error: {error_info.get('message', 'Unknown error')}",
                metadata={"error_aborted": True, "error_details": error_info}
            )

        response = await self.communication_manager.handle_interaction(interaction)

        # Parse the user's choice
        user_choice = 'abort'  # default
        if isinstance(response, dict):
            user_choice = response.get('choice', 'abort').lower()
        elif isinstance(response, str):
            # Try to parse from string response
            response_lower = response.lower()
            if 'retry' in response_lower or response_lower == '1':
                user_choice = 'retry'
            elif 'skip' in response_lower or response_lower == '2':
                user_choice = 'skip'
            else:
                user_choice = 'abort'

        retry_context = context.get('retry_context', {})

        if user_choice == 'retry':
            return StepResult(
                agent_name="User",
                response={"action": "retry"},
                action_type="retry_failed_step",
                success=True,
                metadata={
                    "retry_requested": True,
                    "retry_context": retry_context,
                    "original_agent": retry_context.get('agent_name')
                }
            )
        elif user_choice == 'skip':
            return StepResult(
                agent_name="User",
                response={"action": "skip"},
                action_type="skip_failed_step",
                success=True,
                metadata={"skip_requested": True}
            )
        else:  # abort
            return StepResult(
                agent_name="User",
                response={"action": "abort"},
                action_type="abort_execution",
                success=False,
                error="User chose to abort after error"
            )

    async def _handle_terminal_error(
        self,
        branch: 'ExecutionBranch',
        error_info: Dict[str, Any],
        context: Dict[str, Any]
    ) -> StepResult:
        """Handle terminal error - display only, no retry."""

        # Format for display without retry options
        formatted_message = self._format_terminal_error(error_info)

        # Create interaction for display
        interaction = UserInteraction(
            interaction_id=str(uuid.uuid4()),
            branch_id=branch.id,
            session_id=context.get("session_id"),
            incoming_message=formatted_message,
            interaction_type="terminal_error",  # New type
            communication_mode=CommunicationMode.SYNC,
            calling_agent=error_info.get('failed_agent', 'Unknown'),
            metadata={'error_info': error_info}
        )

        # Display error and wait for acknowledgment
        await self.communication_manager.handle_interaction(interaction)

        # Return failure to terminate execution
        return StepResult(
            agent_name="User",
            response={"action": "terminate"},
            action_type="terminal_error",
            success=False,
            error=f"Terminal error: {error_info.get('message')}"
        )

    def _format_fixable_error(self, error_info: Dict[str, Any]) -> Dict[str, Any]:
        """Format fixable error with retry options."""

        return {
            'type': 'error_recovery',
            'title': self._get_error_title(error_info, fixable=True),
            'content': self._build_fixable_error_content(error_info),
            'options': ['Retry', 'Skip', 'Abort'],
            'default': 'Retry'
        }

    def _format_terminal_error(self, error_info: Dict[str, Any]) -> Dict[str, Any]:
        """Format terminal error for display only."""

        return {
            'type': 'terminal_error',
            'title': self._get_error_title(error_info, fixable=False),
            'content': self._build_terminal_error_content(error_info),
            'options': None,  # No options for terminal errors
            'acknowledge_only': True
        }

    def _get_error_title(self, error_info: Dict[str, Any], fixable: bool) -> str:
        """Get appropriate title based on error type and fixability."""

        category = error_info.get('category', 'unknown')

        if not fixable:
            # Terminal error titles
            titles = {
                'api_error': '❌ API Configuration Error',
                'configuration_error': '❌ Configuration Error',
                'validation_error': '❌ Validation Error',
                'unknown': '❌ Fatal Error'
            }

            # Specific titles for API errors
            if category == 'api_error':
                classification = error_info.get('classification')
                if classification == 'authentication_failed':
                    return '🔐 Authentication Failed - Cannot Continue'
                elif classification == 'invalid_model':
                    return '🚫 Invalid Model - Cannot Continue'
                elif classification == 'permission_denied':
                    return '🚫 Permission Denied - Cannot Continue'
        else:
            # Fixable error titles
            titles = {
                'api_error': '⚠️  API Error - Action Required',
                'resource_error': '📊 Resource Limit - Action Required',
                'timeout_error': '⏰ Timeout - Can Retry',
                'browser_error': '🖥️  Browser Error - Can Fix',
                'unknown': '⚠️  Error Occurred'
            }

            if category == 'api_error':
                classification = error_info.get('classification')
                if classification == 'insufficient_credits':
                    return '💳 Insufficient Credits - Can Fix'
                elif classification == 'rate_limit':
                    return '⏱️  Rate Limit - Can Wait'

        return titles.get(category, titles['unknown'])

    def _build_terminal_error_content(self, error_info: Dict[str, Any]) -> str:
        """Build content for terminal error display."""

        lines = []

        # Error icon and severity
        lines.append("⛔ **EXECUTION CANNOT CONTINUE**\n")

        # Agent that failed
        failed_agent = error_info.get('failed_agent', 'Unknown')
        lines.append(f"**Agent**: {failed_agent}")
        lines.append("")

        # Error message
        message = error_info.get('message', 'An error occurred')
        lines.append(f"**Error**: {message}")

        # Provider for API errors
        if error_info.get('provider'):
            lines.append(f"**Provider**: {error_info['provider']}")

        lines.append("")

        # Termination reason
        reason = error_info.get('termination_reason')
        if reason:
            lines.append("**Why this cannot be fixed:**")
            lines.append(f"{reason}")
        else:
            lines.append("**Reason**: This error requires changes to configuration or code.")

        lines.append("")
        lines.append("**Next Steps:**")
        lines.append("1. Fix the configuration or code issue")
        lines.append("2. Restart the execution")
        lines.append("")
        lines.append("*Press Enter to exit...*")

        return "\n".join(lines)

    def _build_fixable_error_content(self, error_info: Dict[str, Any]) -> str:
        """Build content for fixable error with options."""

        lines = []

        # Agent that failed
        failed_agent = error_info.get('failed_agent', 'Unknown')
        lines.append(f"**Agent**: {failed_agent}")
        lines.append("")

        # Error message
        message = error_info.get('message', 'An error occurred')
        lines.append(f"**Error**: {message}")

        # Provider for API errors
        if error_info.get('provider'):
            lines.append(f"**Provider**: {error_info['provider']}")

        lines.append("")
        lines.append("**Suggested Actions:**")

        # List suggested actions
        for action in error_info.get('suggested_actions', []):
            if action:
                lines.append(f"  • {action}")

        # Add URL if available
        if error_info.get('action_url'):
            lines.append("")
            lines.append(f"  🔗 {error_info['action_url']}")

        lines.append("")
        lines.append("**What would you like to do?**")
        lines.append("  1. **Retry** - Try again after fixing the issue")
        lines.append("  2. **Skip** - Skip this step and continue")
        lines.append("  3. **Abort** - Stop execution")

        return "\n".join(lines)