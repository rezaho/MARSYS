"""Mock runtime for the simulator.

The orchestrator calls `runtime.step(branch)` when it wants to advance a branch.
Mock: maintains a queue of scripted StepResults per branch_id. step() pops the
next scripted result. Raises if nothing is queued (indicating the trace is
incomplete).
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Optional

from ..orchestrator.types import Branch, StepResult


class MockRuntime:
    """Scripted step responses.

    Two script types:
      - Branch-keyed: `queue(branch_id, result)` — specific branch's script.
      - Agent-keyed: `queue_agent(agent_name, result)` — matches any branch
        whose current_agent equals agent_name (for auto-spawned resolvers).

    Branch-keyed takes priority over agent-keyed.
    """

    def __init__(self) -> None:
        self.scripts: dict[str, collections.deque[StepResult]] = {}
        self.agent_scripts: dict[str, collections.deque[StepResult]] = {}
        self.step_log: list[tuple[str, StepResult]] = []

    def queue(self, branch_id: str, result: StepResult) -> None:
        self.scripts.setdefault(branch_id, collections.deque()).append(result)

    def queue_agent(self, agent_name: str, result: StepResult) -> None:
        self.agent_scripts.setdefault(agent_name, collections.deque()).append(result)

    def step(self, branch: Branch) -> StepResult:
        q = self.scripts.get(branch.id)
        if q is not None and q:
            res = q.popleft()
            self.step_log.append((branch.id, res))
            return res
        q = self.agent_scripts.get(branch.current_agent)
        if q is not None and q:
            res = q.popleft()
            self.step_log.append((branch.id, res))
            return res
        raise RuntimeError(
            f"MockRuntime: no scripted event for branch {branch.id} "
            f"(agent={branch.current_agent}, step_count={branch.step_count})"
        )

    def has_queued(self, branch_id: str) -> bool:
        q = self.scripts.get(branch_id)
        return bool(q)

    def has_agent_script(self, agent_name: str) -> bool:
        q = self.agent_scripts.get(agent_name)
        return bool(q)
