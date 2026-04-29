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

from .orchestrator_types import (
    Branch as OrchestratorBranch,
    Invocation,
    StepResult as OrchestratorStepResult,
)

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

        # 2. Build the per-step context expected by StepExecutor.
        context = {
            "session_id": self.session_id,
            "branch_id": branch.id,
            "step_number": branch.step_count,
            "topology_graph": self.topology_graph,
            "execution_config": self.execution_config,
            # branch_executor passes its ExecutionBranch here for dynamic
            # instructions / format building. The orchestrator's Branch
            # is a different shape; pass None for now (system prompt
            # building still works without it).
            "branch": None,
        }

        # 3. Execute the agent step. Returns a MARSYS branches/types.StepResult
        #    with raw response + coordination_action / coordination_data.
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

        # 4. Persist memory back onto the branch. The agent instance
        #    maintains its own memory across calls; we snapshot it here
        #    so the orchestrator's view stays consistent.
        try:
            if hasattr(instance, "memory") and hasattr(instance.memory, "to_messages"):
                branch.memory = list(instance.memory.to_messages())
        except Exception:  # pragma: no cover
            logger.debug("could not snapshot agent memory for branch %s", branch.id)

        # 5. Translate the MARSYS StepResult into an orchestrator StepResult.
        return await self._translate(marsys_result, branch)

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

        # Tool-only / content-only step → NOOP (the step did work but the
        # agent hasn't yet decided coordination; orchestrator re-queues).
        if not coord_action:
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
            agent=None,  # Orchestrator's Branch doesn't carry the instance
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

        if action in (ActionType.FINAL_RESPONSE, ActionType.END_CONVERSATION):
            value = validation.final_response
            if value is None:
                value = (validation.parsed_response or {}).get("content")
            if value is None:
                value = getattr(marsys_result, "response", None)
            return OrchestratorStepResult(kind="FINAL_RESPONSE", value=value)

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


__all__ = ["RealRuntime"]
