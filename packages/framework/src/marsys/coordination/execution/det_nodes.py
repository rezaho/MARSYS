"""Deterministic-node primitives for the unified-barrier orchestrator.

A `DeterministicNode` is a non-LLM node in the topology graph. Subclasses
define explicit, single-purpose behavior that runs inline when an agent
invokes the node. Det-nodes never spawn a normal agent step and never
appear in `Runtime.step`'s contract.

Three concrete subclasses:
  - `StartNode` — workflow entry point. Receives the task at workflow start
    and dispatches branches to its outgoing edges.
  - `EndNode` — workflow exit point. Invocations deliver value to ROOT.
  - `UserNode` — bidirectional human Q&A. Stub here; the actual I/O is
    wired through `UserNodeHandler` in step 11 of the migration.

Det-nodes interact with the orchestrator only through the narrow
`DetNodeContext` Protocol (in `orchestrator_types`), keeping the coupling
explicit and the API surface minimal.

Reserved names: a topology referring to the string "Start", "End", or
"User" resolves to the corresponding `NodeKind` via the single authoritative
`NODE_KIND_BEHAVIOUR` registry (the one source of truth for the kind↔class
↔reserved-name relationship). The deterministic behaviour instance is
materialized from `NodeKind` at the analyzer seam.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Type

from ..topology.core import NodeKind

if TYPE_CHECKING:
    from .orchestrator_types import Barrier, Branch, DetNodeContext


class DeterministicNode(ABC):
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
        self, ctx: "DetNodeContext", branch: "Branch", value: Any
    ) -> None:
        """Handle SINGLE_INVOKE targeting this node."""

    @abstractmethod
    def on_dispatch(
        self, ctx: "DetNodeContext", fork: "Barrier", request: Any
    ) -> None:
        """Handle PARALLEL_INVOKE targeting this node from a fork."""

    def on_workflow_start(self, ctx: "DetNodeContext", task: Any) -> None:
        """Default: no-op. StartNode overrides."""

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other) -> bool:
        return isinstance(other, DeterministicNode) and self.name == other.name


class StartNode(DeterministicNode):
    """Workflow entry point. Required, exactly one per topology.

    Receives the initial task at the start of a run and dispatches a branch
    to each outgoing agent. Each spawned branch's `delivery_target` is
    ROOT; the workflow waits until everything settles (Fix 1: ROOT defer).
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
            "(StartNode receives the task only at workflow start)"
        )

    def on_dispatch(self, ctx, fork, request):
        raise RuntimeError(
            f"StartNode {self.name!r} cannot appear in parallel_invoke targets"
        )


class EndNode(DeterministicNode):
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


class UserNode(DeterministicNode):
    """Bidirectional human Q&A node. Differs from EndNode in that the
    invocation expects a response (the user's input becomes the resumed
    branch's input).

    Wiring: an agent invokes this node either via SINGLE_INVOKE("User", question)
    or via PARALLEL_INVOKE targeting User. Both paths route into
    `ctx.enqueue_user_interaction`, which delegates the async I/O to the
    bound `UserNodeHandler`. The orchestrator picks up the user's reply at
    its next tick and resumes the workflow at `resume_agent`.

    Queue discipline (FIFO single-pending) is owned by the orchestrator: if
    a second branch invokes UserNode while a first interaction is still
    awaiting, the second waits in queue until the first is resolved.
    """

    RESERVED_NAME = "User"

    def __init__(self, name: str = "User", handler: Any = None):
        self.name = name
        self.handler = handler  # UserNodeHandler bound at workflow construction

    def _resume_agent_for(self, ctx, branch) -> str:
        """Pick the resume agent: first non-self successor, else fall back to
        the branch's last_invoked_agent (the calling agent)."""
        for target in ctx.topology.successors(self.name):
            if target == self.name:
                continue
            return target
        return branch.last_invoked_agent or branch.current_agent

    def on_single_invoke(self, ctx, branch, value):
        if self.handler is None:
            ctx.fail(branch, f"UserNode {self.name!r} has no handler bound")
            return
        resume_agent = self._resume_agent_for(ctx, branch)
        ctx.enqueue_user_interaction(branch, prompt=value, resume_agent=resume_agent)

    def on_dispatch(self, ctx, fork, request):
        if self.handler is None:
            ctx.fail(fork.resolver_branch_obj, f"UserNode {self.name!r} has no handler bound")
            return
        # For the parallel-fork case, spawn a placeholder branch at User and
        # treat it like a single invoke. The placeholder's delivery_target is
        # the fork barrier, so the user's response flows back into the fork.
        placeholder = ctx.spawn_branch_at(
            agent=self.name, input=request, delivery_target=fork.id,
        )
        resume_agent = self._resume_agent_for(ctx, placeholder)
        ctx.enqueue_user_interaction(placeholder, prompt=request, resume_agent=resume_agent)


# --- Single source of truth -------------------------------------------------
#
# ``NODE_KIND_BEHAVIOUR`` is THE authoritative ``NodeKind → behaviour-class``
# map. Every other lookup (the parse_node reserved-name carve-out, the
# analyzer's kind→instance materialization, the reserved-name string set in
# topology.core) derives from it — adding a new deterministic kind is one
# enum value + one behaviour class + one entry here, with no dispatch-site
# edits (extension-open; ADR-008 Decision 2 / AC-9/10/11).
#
# ``RESERVED_NODE_NAMES`` in ``topology.core`` is derived from ``NodeKind``
# directly (the non-AGENT kinds); this registry binds those kinds to the
# classes that run them and exposes the name↔kind mapping derived from each
# class's ``RESERVED_NAME`` attribute.
NODE_KIND_BEHAVIOUR: Dict[NodeKind, Type[DeterministicNode]] = {
    NodeKind.START: StartNode,
    NodeKind.END: EndNode,
    NodeKind.USER: UserNode,
}

# Derived: reserved string name → NodeKind (e.g. "Start" → NodeKind.START).
# Used by ``parse_node`` to map a string node spec to a uniform Node(kind=).
# Single-sourced from NODE_KIND_BEHAVIOUR + each class's RESERVED_NAME — NOT a
# second hand-maintained dict.
RESERVED_NAME_TO_KIND: Dict[str, NodeKind] = {
    cls.RESERVED_NAME: kind for kind, cls in NODE_KIND_BEHAVIOUR.items()
}


def behaviour_for_kind(kind: NodeKind) -> Type[DeterministicNode]:
    """Return the deterministic behaviour class bound to ``kind``.

    Raises ``KeyError`` for ``NodeKind.AGENT`` (agents are LLM-driven, not
    deterministic) — callers branch on ``kind`` before materializing.
    """
    return NODE_KIND_BEHAVIOUR[kind]


__all__ = [
    "DeterministicNode",
    "StartNode",
    "EndNode",
    "UserNode",
    "NODE_KIND_BEHAVIOUR",
    "RESERVED_NAME_TO_KIND",
    "behaviour_for_kind",
]
