"""RealRuntime: production runtime adapter for the unified-barrier orchestrator.

Implements the `Runtime` Protocol (from orchestrator_types) by wrapping the
existing MARSYS step-execution stack:

    Orchestrator ↔ RealRuntime.step(branch)
                       │
                       ├── AgentRegistry.get_or_acquire(agent, branch_id)
                       ├── StepExecutor.execute_step(instance, request, [], context)
                       │     → produces MARSYS branches/types.StepResult
                       ├── ValidationProcessor.validate_coordination_action(...)
                       │     → ValidationResult with ActionType + invocations
                       └── _translate(...) → orchestrator_types.StepResult

Step 10 lands the skeleton: instance acquisition, memory rehydration / save,
StepExecutor invocation, validation, and the basic happy-path translation
(SINGLE_INVOKE, PARALLEL_INVOKE, FINAL_RESPONSE, FAIL). Step 11 wires
RulesEngine, SteeringManager, StatusManager event emission, TraceCollector,
and the UserNodeHandler routing for ERROR_RECOVERY / TERMINAL_ERROR.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from ..validation.types import ValidationErrorCategory
from .orchestrator_types import (
    Branch as OrchestratorBranch,
    Invocation,
    StepResult as OrchestratorStepResult,
)

# Default thresholds for content-only-loop detection. The runtime reads
# the actual values from `execution_config` (so workflows can tune them);
# these constants remain as fallbacks for tests that instantiate
# RealRuntime without an ExecutionConfig.
CONTENT_ONLY_STEERING_THRESHOLD = 2
CONTENT_ONLY_HARD_LIMIT = 10

if TYPE_CHECKING:
    from ...agents.registry import AgentRegistry
    from ..topology.graph import TopologyGraph
    from ..validation.response_validator import ValidationProcessor
    from .step_executor import StepExecutor

logger = logging.getLogger(__name__)


class RealRuntime:
    """Production Runtime adapter.

    Constructed once per Orchestra.run() invocation; provides one `step()`
    call per branch tick. The orchestrator owns Branch lifecycle; this
    runtime owns step execution and translation.
    """

    def __init__(
        self,
        registry: "AgentRegistry",
        step_executor: "StepExecutor",
        validator: "ValidationProcessor",
        topology_graph: "TopologyGraph",
        session_id: str,
        execution_config: Any = None,
    ):
        self.registry = registry
        self.step_executor = step_executor
        self.validator = validator
        self.topology_graph = topology_graph
        self.session_id = session_id
        self.execution_config = execution_config

    async def step(self, branch: OrchestratorBranch) -> OrchestratorStepResult:
        """One branch tick: acquire instance, execute, validate, translate.

        Returns an orchestrator StepResult. The orchestrator decides what
        to do with it (deliver, dispatch, fail, etc.).
        """
        # 1. Acquire the agent instance (handles single agents, pools,
        #    pool instances). AgentRegistry.get_or_acquire is sync.
        instance = self.registry.get_or_acquire(
            branch.current_agent, branch_id=branch.id
        )
        if instance is None:
            return OrchestratorStepResult(
                kind="FAIL",
                error=f"agent {branch.current_agent!r} not registered",
            )
        self._current_instance = instance

        steering_threshold = getattr(
            self.execution_config, "content_only_steering_threshold",
            CONTENT_ONLY_STEERING_THRESHOLD,
        )
        hard_limit = getattr(
            self.execution_config, "content_only_hard_limit",
            CONTENT_ONLY_HARD_LIMIT,
        )

        if branch.consecutive_content_only >= hard_limit:
            diagnostic = self._build_content_only_diagnostic(branch, instance)
            return OrchestratorStepResult(kind="FAIL", error=diagnostic)

        is_continuation = branch.last_invoked_agent == branch.current_agent
        context = {
            "session_id": self.session_id,
            "branch_id": branch.id,
            "step_number": branch.step_count,
            "topology_graph": self.topology_graph,
            "execution_config": self.execution_config,
            "tool_continuation": is_continuation,
            "branch": None,
        }
        if branch.consecutive_content_only >= steering_threshold:
            context["metadata"] = {
                "agent_error_context": {
                    "category": ValidationErrorCategory.ACTION_ERROR.value,
                    "error_message": (
                        "Your last response contained no coordination tool call. "
                        "The workflow cannot advance from text content alone — "
                        "you must invoke one of your available coordination tools."
                    ),
                    "retry_count": branch.consecutive_content_only,
                    "failed_action": "content_only",
                },
            }

        try:
            marsys_result = await self.step_executor.execute_step(
                agent=instance,
                request=branch.input,
                memory=branch.memory,
                context=context,
            )
        except Exception as e:
            logger.exception(
                "StepExecutor raised for branch %s (agent=%s)", branch.id, branch.current_agent
            )
            return OrchestratorStepResult(kind="FAIL", error=f"step_executor error: {e}")
        finally:
            branch.last_invoked_agent = branch.current_agent

        # 4. Persist memory back onto the branch. The agent instance
        #    maintains its own memory across calls; we snapshot it here
        #    so the orchestrator's view stays consistent.
        try:
            if hasattr(instance, "memory") and hasattr(instance.memory, "to_messages"):
                branch.memory = list(instance.memory.to_messages())
        except Exception:  # pragma: no cover
            logger.debug("could not snapshot agent memory for branch %s", branch.id)

        # 5. Translate the MARSYS StepResult into an orchestrator StepResult.
        result = await self._translate(marsys_result, branch)
        # Stamp step_span_id so the orchestrator can forward it as
        # ``parent_step_span_id`` for child branches spawned this tick.
        step_span_id = context.get("step_span_id")
        if step_span_id and result.step_span_id is None:
            result.step_span_id = step_span_id
        return result

    async def _translate(
        self, marsys_result: Any, branch: OrchestratorBranch
    ) -> OrchestratorStepResult:
        """Translate a MARSYS branches/types.StepResult into the
        orchestrator's StepResult shape.

        The MARSYS result carries:
          - coordination_action: e.g. "invoke_agent", "parallel_invoke",
            "return_final_response", "end_conversation"
          - coordination_data: the parsed args of that action
          - response: the agent's text response
          - tool_calls / tool_results: native tool execution

        We run ValidationProcessor.validate_coordination_action on the
        coordination call to get a typed ActionType + invocations list,
        then map to one of the five orchestrator step kinds.
        """
        # Step-executor failure path.
        if not getattr(marsys_result, "success", True):
            return OrchestratorStepResult(
                kind="FAIL",
                error=getattr(marsys_result, "error", None) or "step failed",
            )

        coord_action = getattr(marsys_result, "coordination_action", None)
        coord_data = getattr(marsys_result, "coordination_data", None) or {}
        has_tool_calls = bool(getattr(marsys_result, "tool_calls", None))

        if coord_action or has_tool_calls:
            branch.consecutive_content_only = 0

        if not coord_action:
            if not has_tool_calls:
                branch.consecutive_content_only += 1
            return OrchestratorStepResult(kind="NOOP")

        # Validate the coordination action against the topology.
        from ..branches.types import ExecutionState
        from ..validation.response_validator import ActionType

        exec_state = ExecutionState(
            session_id=self.session_id,
            current_step=branch.step_count,
            status="running",
        )
        validation = await self.validator.validate_coordination_action(
            action=coord_action,
            data=coord_data,
            agent=getattr(self, "_current_instance", None),
            branch=None,
            exec_state=exec_state,
        )

        if not validation.is_valid:
            return OrchestratorStepResult(
                kind="FAIL",
                error=validation.error_message or "coordination validation failed",
            )

        action = validation.action_type

        if action == ActionType.PARALLEL_INVOKE:
            invocations = [
                Invocation(agent=inv.agent_name, request=inv.request)
                for inv in (validation.invocations or [])
            ]
            return OrchestratorStepResult(
                kind="PARALLEL_INVOKE",
                invocations=invocations,
            )

        if action == ActionType.INVOKE_AGENT:
            target = validation.next_agent
            if target is None and validation.invocations:
                target = validation.invocations[0].agent_name
            request = (
                validation.invocations[0].request if validation.invocations else None
            )
            return OrchestratorStepResult(
                kind="SINGLE_INVOKE",
                next_agent=target,
                request=request,
            )

        if action in (
            ActionType.FINAL_RESPONSE,
            ActionType.TERMINATE_WORKFLOW,
            ActionType.END_CONVERSATION,
        ):
            value = validation.final_response
            if value is None:
                value = (validation.parsed_response or {}).get("final_response")
            if value is None:
                value = (validation.parsed_response or {}).get("content")
            if value is None:
                value = getattr(marsys_result, "response", None)
            return OrchestratorStepResult(kind="FINAL_RESPONSE", value=value)

        if action == ActionType.ASK_USER:
            question = (validation.parsed_response or {}).get("question")
            return OrchestratorStepResult(
                kind="SINGLE_INVOKE",
                next_agent="User",
                value=question,
            )

        if action == ActionType.TERMINAL_ERROR:
            return OrchestratorStepResult(
                kind="FAIL",
                error=validation.error_message or "terminal error",
            )

        if action == ActionType.AUTO_RETRY:
            # Treat as NOOP so the orchestrator re-queues; the agent will
            # produce a corrected response on the next tick.
            return OrchestratorStepResult(kind="NOOP")

        if action == ActionType.ERROR_RECOVERY:
            # Step 11 wires this to UserNodeHandler. For the skeleton, fail
            # with a clear marker so we know if a real run hits this path.
            return OrchestratorStepResult(
                kind="FAIL",
                error=f"ERROR_RECOVERY not yet wired (step 11): {validation.error_message}",
            )

        return OrchestratorStepResult(
            kind="FAIL",
            error=f"unknown coordination action: {action!r}",
        )


    def _build_content_only_diagnostic(
        self, branch: OrchestratorBranch, instance: Any
    ) -> str:
        """Structured diagnostic emitted when a branch hits CONTENT_ONLY_HARD_LIMIT.

        Names what tools the agent had, what it kept emitting, and points at the
        likely root cause (instruction mismatch or topology gap). Read by tests
        and humans inspecting workflow failures."""
        agent_name = branch.current_agent
        coord_tools: list[str] = []
        next_agents: list[str] = []
        try:
            if self.topology_graph is not None:
                next_agents = list(self.topology_graph.get_next_agents(agent_name) or [])
                if any(a for a in next_agents if a.lower() not in ("user", "start", "end")):
                    coord_tools.append("invoke_agent")
                if hasattr(self.topology_graph, "has_edge_to_endnode") and \
                        self.topology_graph.has_edge_to_endnode(agent_name):
                    coord_tools.append("terminate_workflow")
                if hasattr(self.topology_graph, "has_edge_to_usernode") and \
                        self.topology_graph.has_edge_to_usernode(agent_name):
                    coord_tools.append("ask_user")
        except Exception:
            logger.debug("could not inspect topology for diagnostic", exc_info=True)

        regular_tools: list[str] = []
        try:
            schema = getattr(instance, "tools_schema", None)
            if schema:
                for t in schema:
                    fn = t.get("function") if isinstance(t, dict) else None
                    if fn and fn.get("name"):
                        regular_tools.append(fn["name"])
        except Exception:
            pass

        last_content = ""
        try:
            for msg in reversed(list(branch.memory or [])):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    raw = msg.get("content")
                    if isinstance(raw, str) and raw.strip():
                        last_content = raw[:200]
                        break
        except Exception:
            pass

        return (
            f"Content-only loop detected: agent {agent_name!r} produced "
            f"{branch.consecutive_content_only} consecutive responses with no coordination tool call. "
            f"Available coordination tools: {coord_tools or '(none)'}. "
            f"Available regular tools: {regular_tools or '(none)'}. "
            f"Last assistant content snippet: {last_content!r}. "
            "Likely cause: the agent's instruction asks for an action that doesn't match its available "
            "coordination tools, or the topology has no edge that exposes the right tool. Review the "
            "agent's instruction and the topology gating."
        )


__all__ = ["RealRuntime"]
