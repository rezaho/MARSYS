"""Deterministic-node primitives for the simulator.

A `SimDeterministicNode` is a non-LLM node in the topology graph. Subclasses
define explicit, single-purpose behavior that runs inline when an agent
invokes the node. Det-nodes never spawn a branch and never appear in
`runtime.step`'s contract.

Three concrete subclasses:
  - `StartNode` — workflow entry point. Receives the task at workflow start
    and dispatches branches to its outgoing edges.
  - `EndNode` — workflow exit point. Invocations deliver value to ROOT.
  - `UserNode` (deferred) — bidirectional human Q&A.

Det-nodes interact with the orchestrator only through a narrow Protocol
(`DetNodeContext`), keeping the coupling explicit and the API surface minimal.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Protocol

if TYPE_CHECKING:
    from ..orchestrator.types import Barrier, Branch
    from .topology import SimTopology


class DetNodeContext(Protocol):
    """Narrow API exposed to deterministic-node handlers.

    Implementations (the Orchestrator) provide these. New methods are added
    here when a new det-node type needs them. Det-nodes don't see internal
    orchestrator state directly.
    """

    @property
    def topology(self) -> "SimTopology": ...

    @property
    def root_barrier_id(self) -> Optional[str]: ...

    @property
    def barriers(self) -> dict[str, "Barrier"]: ...

    def deliver(self, branch: "Branch", target_barrier_id: str, value: Any) -> None:
        """Branch delivers value to a barrier; branch becomes TERMINATED."""

    def dispatch(self, fork: "Barrier", target_barrier_id: str, request: Any) -> None:
        """Fork's parallel-invoke side-dispatches to a barrier."""

    def deliver_to_root(self, branch: "Branch", value: Any) -> None:
        """Convenience for End-style routing."""

    def dispatch_to_root(self, fork: "Barrier", request: Any) -> None:
        """Convenience for End-style routing from a fork."""

    def fail(self, branch: "Branch", error: str) -> None:
        """Branch fails into its delivery_target."""

    def spawn_branch_at(
        self, agent: str, input: Any, delivery_target: str
    ) -> "Branch":
        """Spawn a fresh branch at an agent (used by Start)."""


class SimDeterministicNode(ABC):
    """Base for non-LLM nodes.

    Subclasses define behavior on three lifecycle moments:
      - `on_workflow_start`: called once at workflow start (only StartNode
        overrides; default is no-op).
      - `on_single_invoke`: when an agent does SINGLE_INVOKE(this).
      - `on_dispatch`: when an agent does PARALLEL_INVOKE with this as a
        target (the fork dispatches here).
    """

    name: str

    @abstractmethod
    def on_single_invoke(
        self, ctx: DetNodeContext, branch: "Branch", value: Any
    ) -> None:
        """Handle SINGLE_INVOKE targeting this node."""

    @abstractmethod
    def on_dispatch(
        self, ctx: DetNodeContext, fork: "Barrier", request: Any
    ) -> None:
        """Handle PARALLEL_INVOKE targeting this node from a fork."""

    def on_workflow_start(self, ctx: DetNodeContext, task: Any) -> None:
        """Default: no-op. StartNode overrides."""

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other) -> bool:
        return isinstance(other, SimDeterministicNode) and self.name == other.name


class StartNode(SimDeterministicNode):
    """Workflow entry point. Required, exactly one per topology.

    Receives the initial task at `orchestra.run(task=...)` and dispatches a
    branch to each outgoing agent. Each spawned branch's `delivery_target`
    is ROOT; with Fix 1 the workflow waits until everything settles.
    """

    RESERVED_NAME = "Start"

    def __init__(self, name: str = "Start"):
        self.name = name

    def on_workflow_start(self, ctx, task):
        for target in ctx.topology.successors(self.name):
            ctx.spawn_branch_at(target, task, delivery_target=ctx.root_barrier_id)

    def on_single_invoke(self, ctx, branch, value):
        raise RuntimeError(
            f"StartNode {self.name!r} cannot be invoked from an agent "
            "(use SINGLE_INVOKE on a regular agent instead)"
        )

    def on_dispatch(self, ctx, fork, request):
        raise RuntimeError(
            f"StartNode {self.name!r} cannot appear in parallel_invoke targets"
        )


class EndNode(SimDeterministicNode):
    """Workflow exit point. Optional; zero or more per topology.

    Agents with edges to End can invoke it explicitly to deliver the
    workflow's final answer. Invocations route directly to ROOT.

    SINGLE_INVOKE: branch delivers value to ROOT and terminates.
    PARALLEL_INVOKE target: fork dispatches request to ROOT (fire-and-forget;
    no fork.upstream wiring — ROOT is sink, not a sync gate).
    """

    RESERVED_NAME = "End"

    def __init__(self, name: str = "End"):
        self.name = name

    def on_single_invoke(self, ctx, branch, value):
        ctx.deliver_to_root(branch, value)

    def on_dispatch(self, ctx, fork, request):
        ctx.dispatch_to_root(fork, request)


# Reserved-name registry: maps a reserved string name to the det-node class.
# Used by `build_topology` when resolving string-form node specs like:
#   build_topology(nodes=["Start", "A", "End"], flows=[...])
RESERVED_NAMES: dict[str, type[SimDeterministicNode]] = {
    StartNode.RESERVED_NAME: StartNode,
    EndNode.RESERVED_NAME: EndNode,
    # Future: UserNode.RESERVED_NAME: UserNode,
}
