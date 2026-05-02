"""
Response validation for coordination tool calls and error message handling.

This module provides:
- validate_coordination_action(): Validates native coordination tool calls against topology
- process_error_message(): Classifies API error Messages and routes to recovery/retry
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..branches.types import ExecutionBranch, ExecutionState
from ..topology.graph import TopologyGraph
from .types import AgentInvocation, ValidationErrorCategory

if TYPE_CHECKING:
    from ...agents import BaseAgent

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Action types produced by validation."""
    INVOKE_AGENT = "invoke_agent"
    PARALLEL_INVOKE = "parallel_invoke"
    # REMOVE-IN-V0.4: FINAL_RESPONSE is the legacy enum name; the canonical
    # action is now TERMINATE_WORKFLOW. Tests on tests/coordination/test_response_validator.py
    # still assert FINAL_RESPONSE for the legacy "return_final_response" tool name.
    FINAL_RESPONSE = "final_response"
    TERMINATE_WORKFLOW = "terminate_workflow"
    ASK_USER = "ask_user"
    END_CONVERSATION = "end_conversation"
    ERROR_RECOVERY = "error_recovery"
    TERMINAL_ERROR = "terminal_error"
    AUTO_RETRY = "auto_retry"


@dataclass
class ValidationResult:
    """Result of validation with routing decisions."""
    is_valid: bool
    action_type: Optional[ActionType] = None
    parsed_response: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    retry_suggestion: Optional[str] = None
    error_category: Optional[str] = None
    invocations: List['AgentInvocation'] = None
    tool_calls: List[Dict[str, Any]] = None

    next_agent: Optional[str] = None
    should_end_branch: bool = False
    requires_tool_continuation: bool = False
    final_response: Optional[Any] = None

    def __post_init__(self):
        if self.invocations is None:
            self.invocations = []
        if self.tool_calls is None:
            self.tool_calls = []
        if not self.next_agent and self.invocations and len(self.invocations) == 1:
            self.next_agent = self.invocations[0].agent_name

    @property
    def next_agents(self) -> List[str]:
        return [inv.agent_name for inv in self.invocations] if self.invocations else []


class ValidationProcessor:
    """
    Validates coordination tool calls and classifies error messages.

    Two entry points:
    - validate_coordination_action(): For native tool calls (invoke_agent, return_final_response, end_conversation)
    - process_error_message(): For API error Messages (role="error")
    """

    def __init__(
        self,
        topology_graph: TopologyGraph,
        response_format: str = "json"
    ):
        self.topology_graph = topology_graph
        self._response_format = response_format

    # ── Coordination Tool Validation ─────────────────────────────────

    async def validate_coordination_action(
        self,
        action: str,
        data: Dict[str, Any],
        agent: BaseAgent,
        branch: ExecutionBranch,
        exec_state: ExecutionState,
    ) -> ValidationResult:
        """
        Validate a coordination tool call against topology.

        Called by BranchExecutor with pre-parsed data from
        StepResult.coordination_action and StepResult.coordination_data.
        """
        if action == "invoke_agent":
            return await self._validate_invoke_agent(data, agent)
        elif action == "terminate_workflow":
            return await self._validate_terminate_workflow(data, agent)
        # REMOVE-IN-V0.4: legacy "return_final_response" tool name; renamed to
        # "terminate_workflow". Drop this branch and _validate_return_final_response.
        elif action == "return_final_response":
            return await self._validate_return_final_response(data, agent)
        elif action == "ask_user":
            return await self._validate_ask_user(data, agent)
        elif action == "end_conversation":
            return await self._validate_end_conversation(data, agent, branch)
        else:
            return ValidationResult(
                is_valid=False,
                error_message=f"Unknown coordination action: {action}",
                error_category=ValidationErrorCategory.ACTION_ERROR.value,
            )

    async def _validate_invoke_agent(
        self, data: Dict[str, Any], agent: BaseAgent
    ) -> ValidationResult:
        """Validate invoke_agent: check invocations against topology."""
        raw_invocations = data.get("invocations", [])
        if not raw_invocations:
            return ValidationResult(
                is_valid=False,
                error_message="invoke_agent called with no invocations",
                error_category=ValidationErrorCategory.ACTION_ERROR.value,
            )

        # Build AgentInvocation objects
        invocations = []
        for idx, inv in enumerate(raw_invocations):
            invocations.append(AgentInvocation(
                agent_name=inv.get("agent_name", ""),
                request=inv.get("request", ""),
                instance_id=f"{inv.get('agent_name', 'unknown')}_{idx}_{uuid.uuid4().hex[:8]}",
            ))

        # Check all targets against topology
        next_agents = self.topology_graph.get_next_agents(agent.name)
        invalid = [inv.agent_name for inv in invocations if inv.agent_name not in next_agents]

        if invalid:
            return ValidationResult(
                is_valid=False,
                error_message=f"Agent {agent.name} cannot invoke: {invalid}",
                retry_suggestion=f"You cannot invoke {invalid}. Available agents: {next_agents}",
                error_category=ValidationErrorCategory.PERMISSION_ERROR.value,
            )

        action_type = ActionType.PARALLEL_INVOKE if len(invocations) > 1 else ActionType.INVOKE_AGENT

        return ValidationResult(
            is_valid=True,
            action_type=action_type,
            invocations=invocations,
            parsed_response=data,
        )

    async def _validate_terminate_workflow(
        self, data: Dict[str, Any], agent: BaseAgent
    ) -> ValidationResult:
        """Validate terminate_workflow: agent must have a direct edge to an End det-node.

        Legacy entry/exit/User-edge metadata is translated to det-node edges
        by the topology shim before validation runs, so this is the single
        source of truth."""
        graph = self.topology_graph
        gated = (
            hasattr(graph, "has_edge_to_endnode")
            and graph.has_edge_to_endnode(agent.name)
        )

        if not gated:
            next_agents = graph.get_next_agents(agent.name)
            return ValidationResult(
                is_valid=False,
                error_message=f"Agent '{agent.name}' cannot terminate the workflow (no edge to End)",
                retry_suggestion=f"You must invoke one of: {next_agents}",
                error_category=ValidationErrorCategory.PERMISSION_ERROR.value,
            )

        # Extract content. For output_schema agents the entire data dict carries it.
        answer = data.get("answer", "")
        if not answer:
            answer = data.get("response", "")
        if not answer and data:
            answer = data

        return ValidationResult(
            is_valid=True,
            action_type=ActionType.TERMINATE_WORKFLOW,
            final_response=answer if not isinstance(answer, dict) else None,
            parsed_response={
                "final_response": answer,
                "action_input": {"answer": answer},
            },
        )

    # REMOVE-IN-V0.4: legacy alias for terminate_workflow. Tool name
    # "return_final_response" was renamed to "terminate_workflow"; this
    # dispatcher and ActionType.FINAL_RESPONSE remap stay until v0.4 to
    # keep existing user code and 3 test cases in
    # tests/coordination/test_response_validator.py working.
    async def _validate_return_final_response(
        self, data: Dict[str, Any], agent: BaseAgent
    ) -> ValidationResult:
        """Legacy alias for terminate_workflow (deprecated)."""
        result = await self._validate_terminate_workflow(data, agent)
        if result.is_valid:
            result.action_type = ActionType.FINAL_RESPONSE
        return result

    async def _validate_ask_user(
        self, data: Dict[str, Any], agent: BaseAgent
    ) -> ValidationResult:
        """Validate ask_user: agent has direct edge to User det-node."""
        graph = self.topology_graph
        if not (hasattr(graph, "has_edge_to_usernode") and graph.has_edge_to_usernode(agent.name)):
            next_agents = graph.get_next_agents(agent.name)
            return ValidationResult(
                is_valid=False,
                error_message=f"Agent '{agent.name}' cannot ask the user (no edge to User)",
                retry_suggestion=f"You must invoke one of: {next_agents}",
                error_category=ValidationErrorCategory.PERMISSION_ERROR.value,
            )

        question = data.get("question", "")
        if not question:
            return ValidationResult(
                is_valid=False,
                error_message="ask_user called with empty question",
                error_category=ValidationErrorCategory.ACTION_ERROR.value,
            )

        return ValidationResult(
            is_valid=True,
            action_type=ActionType.ASK_USER,
            parsed_response={
                "question": question,
                "action_input": {"question": question},
            },
        )

    async def _validate_end_conversation(
        self, data: Dict[str, Any], agent: BaseAgent, branch: ExecutionBranch
    ) -> ValidationResult:
        """Validate end_conversation: check branch is a conversation type."""
        if branch.type.value != "conversation":
            next_agents = self.topology_graph.get_next_agents(agent.name)
            if self.topology_graph.has_user_access(agent.name):
                suggestion = "Use return_final_response to complete execution."
            elif next_agents:
                suggestion = f"Invoke one of these agents to continue: {next_agents}"
            else:
                suggestion = "No valid actions available."

            return ValidationResult(
                is_valid=False,
                error_message="Cannot end conversation in non-conversation branch",
                retry_suggestion=suggestion,
                error_category=ValidationErrorCategory.ACTION_ERROR.value,
            )

        return ValidationResult(
            is_valid=True,
            action_type=ActionType.END_CONVERSATION,
            parsed_response={"summary": data.get("summary", "")},
        )

    # ── Error Message Handling ───────────────────────────────────────

    async def process_error_message(
        self,
        message: Any,
        agent: BaseAgent,
        branch: ExecutionBranch,
        exec_state: ExecutionState,
    ) -> ValidationResult:
        """
        Classify an API error Message and route to recovery/retry.

        Called by BranchExecutor when agent returns a Message with role="error".
        Uses ErrorMessageProcessor to classify the error, then dispatches to
        the appropriate handler.
        """
        from ..formats.processors import ErrorMessageProcessor

        processor = ErrorMessageProcessor()

        if not processor.can_process(message):
            return ValidationResult(
                is_valid=False,
                error_message="Expected error Message but got non-error response",
            )

        parsed = processor.process(message)
        if not parsed:
            return ValidationResult(
                is_valid=False,
                error_message="Failed to parse error Message content",
            )

        action_str = parsed.get("next_action", "error_recovery")

        if action_str == "auto_retry":
            return ValidationResult(
                is_valid=True,
                action_type=ActionType.AUTO_RETRY,
                parsed_response=parsed,
            )
        elif action_str == "terminal_error":
            return await self._build_terminal_error_result(parsed, agent, branch)
        else:  # error_recovery
            return await self._build_error_recovery_result(parsed, agent, branch)

    async def _build_error_recovery_result(
        self, parsed: Dict[str, Any], agent: BaseAgent, branch: ExecutionBranch
    ) -> ValidationResult:
        """Build ValidationResult that routes to User for error recovery."""
        error_info = parsed.get("error_info", {})

        suggested_actions = []
        if error_info.get("suggested_action"):
            suggested_actions.append(error_info["suggested_action"])
        if error_info.get("classification") == "insufficient_credits":
            if "Add credits" not in str(error_info.get("suggested_action", "")):
                suggested_actions.append("Add credits to your account")
            suggested_actions.append("Then retry to continue from where you left off")
        if suggested_actions:
            error_info["suggested_actions"] = suggested_actions

        user_invocation = AgentInvocation(
            agent_name="User",
            request={
                "error_details": error_info,
                "retry_context": {"agent_name": agent.name, "branch_id": branch.id}
            }
        )

        return ValidationResult(
            is_valid=True,
            action_type=ActionType.ERROR_RECOVERY,
            invocations=[user_invocation],
            parsed_response={
                **parsed,
                "target_agent": "User",
                "error_details": error_info,
                "retry_context": {"agent_name": agent.name, "branch_id": branch.id}
            },
        )

    async def _build_terminal_error_result(
        self, parsed: Dict[str, Any], agent: BaseAgent, branch: ExecutionBranch
    ) -> ValidationResult:
        """Build ValidationResult that routes to User for terminal error display."""
        error_info = parsed.get("error_info", {})

        if error_info.get("classification") == "authentication_failed":
            error_info["termination_reason"] = (
                "Authentication failed. This requires updating your API key "
                "in the configuration file. Cannot be fixed while running."
            )

        user_invocation = AgentInvocation(
            agent_name="User",
            request={"error_details": error_info}
        )

        return ValidationResult(
            is_valid=True,
            action_type=ActionType.TERMINAL_ERROR,
            invocations=[user_invocation],
            parsed_response={
                **parsed,
                "target_agent": "User",
                "error_details": error_info,
            },
        )
