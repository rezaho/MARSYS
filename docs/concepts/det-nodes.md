# Deterministic nodes (Start, End, User)

A **deterministic node** (det-node) is a non-LLM node in the topology graph. Det-nodes have explicit, single-purpose behavior that runs inline when an agent invokes them — they don't make model calls, they don't appear in `Runtime.step`'s contract, and they interact with the orchestrator only through the narrow `DetNodeContext` Protocol.

MARSYS reserves three singleton det-nodes:

- **`StartNode`** — workflow entry point. Required for new topologies (the legacy migration shim auto-creates one for topologies using `entry_point` metadata).
- **`EndNode`** — workflow exit point. Optional but typical: agents with a direct edge to `End` get the `terminate_workflow` tool in their schema.
- **`UserNode`** — bidirectional human Q&A. Agents with a direct edge to `User` get the `ask_user` tool.

Each is a singleton — at most one of each kind per topology.

Source: `src/marsys/coordination/execution/det_nodes.py` (187 LoC).

## Reserved names

`Start`, `End`, `User` are reserved at the topology level. The string converters (`StringNotationConverter`, `ObjectNotationConverter`) auto-resolve these names to det-node instances via `RESERVED_DETNODE_NAMES` (`det_nodes.py:174`):

```python
RESERVED_DETNODE_NAMES = {
    StartNode.RESERVED_NAME: StartNode,   # "Start"
    EndNode.RESERVED_NAME:   EndNode,     # "End"
    UserNode.RESERVED_NAME:  UserNode,    # "User"
}
```

This means a topology like `{"agents": ["Start", "A", "End"], "flows": ["Start -> A", "A -> End"]}` works without explicitly importing the det-node classes — the converter does the resolution.

## Lifecycle hooks

Each det-node implements three methods on the `DeterministicNode` ABC (`det_nodes.py:32`):

### `on_workflow_start(ctx, task) -> None`

Called once when `Orchestrator.init_workflow(task)` runs, immediately after ROOT is created. Default implementation is a no-op; only `StartNode` overrides.

`StartNode.on_workflow_start` (line 80) iterates over the start node's outgoing edges and dispatches a branch to each:

```python
def on_workflow_start(self, ctx, task):
    for target in ctx.topology.successors(self.name):
        ctx.spawn_branch_at(target, task, delivery_target=ctx.root_barrier_id)
```

Each spawned branch's `delivery_target` is the ROOT barrier; the workflow runs until everything settles (per the ROOT-defer fire gate — see [ADR-005](../architecture/framework/decisions/ADR-005-unified-barrier-algorithm.md)).

### `on_single_invoke(ctx, branch, value) -> None`

Called when an agent emits `SINGLE_INVOKE` targeting this det-node — either via `invoke_agent` with the det-node as the target (rare; usually det-nodes are routed via the dedicated coordination tools), or when the agent emits `terminate_workflow` (routes through `EndNode.on_single_invoke`) or `ask_user` (routes through `UserNode.on_single_invoke`).

| Det-node | Behavior |
|---|---|
| `StartNode.on_single_invoke` (line 84) | Raises `RuntimeError`. Start can only receive the task at workflow start, never from an agent. |
| `EndNode.on_single_invoke` (line 112) | `ctx.deliver_to_root(branch, value)` — branch terminates, value goes to ROOT. |
| `UserNode.on_single_invoke` (line 150) | Computes the resume agent (first non-self successor or the calling agent), then `ctx.enqueue_user_interaction(branch, prompt=value, resume_agent=...)`. |

### `on_dispatch(ctx, fork, request) -> None`

Called when a `parallel_invoke` fork includes this det-node as one of its targets. The fork barrier is passed in so the det-node can route the request appropriately.

| Det-node | Behavior |
|---|---|
| `StartNode.on_dispatch` (line 90) | Raises `RuntimeError`. Start can't appear in `parallel_invoke` targets. |
| `EndNode.on_dispatch` (line 115) | `ctx.dispatch_to_root(fork, request)` — side-dispatch to ROOT (fire-and-forget; the fork doesn't wait for ROOT). |
| `UserNode.on_dispatch` (line 157) | Spawns a placeholder branch at User with the fork as `delivery_target`, then `ctx.enqueue_user_interaction(...)`. The user's response flows back into the fork barrier. |

## DetNodeContext Protocol

Det-nodes interact with the orchestrator only through this narrow Protocol (`src/marsys/coordination/execution/orchestrator_types.py`, lines ~220-275):

```python
class DetNodeContext(Protocol):
    @property
    def topology(self) -> TopologyLike: ...

    @property
    def root_barrier_id(self) -> Optional[str]: ...

    def deliver(self, branch: Branch, target_barrier_id: str, value: Any) -> None: ...

    def deliver_to_root(self, branch: Branch, value: Any) -> None: ...

    def dispatch_to_root(self, fork: Barrier, request: Any) -> None: ...

    def spawn_branch_at(
        self, agent: str, input: Any, delivery_target: str
    ) -> Branch: ...

    def enqueue_user_interaction(
        self, branch: Branch, prompt: Any, resume_agent: str
    ) -> None: ...

    def resume_branch_with_user_response(
        self, suspended_branch_id: str, response: Any, resume_agent: str
    ) -> None: ...

    def fail(self, branch: Branch, error: str) -> None: ...
```

The `Orchestrator` class implements this Protocol. The narrow interface keeps det-node coupling explicit — det-nodes can't accidentally reach into branch internals or fire barriers directly.

## UserNode interaction queue

`UserNode` is the most subtle det-node because human I/O is asynchronous. The discipline (FIFO single-pending) is owned by the orchestrator:

- Only one user question may be outstanding at any time.
- If a second branch invokes `UserNode` while a first interaction is still awaiting the user's reply, the second waits in queue.
- Resolution happens at the orchestrator's next tick after the user replies — `Orchestrator._resume_user_responses` is an `asyncio.Queue` (lazily created) that the `UserNodeHandler` writes to.
- The resume agent is computed by `UserNode._resume_agent_for(ctx, branch)` (line 141): the first non-self successor in the topology, falling back to `branch.last_invoked_agent` (the calling agent).

The `UserNode` constructor accepts a `handler` kwarg (`UserNodeHandler` instance, bound at workflow construction time). If `handler` is `None`, `on_single_invoke` and `on_dispatch` call `ctx.fail(...)` immediately.

## Worked example: explicit det-node edges

```python
from marsys.coordination import Orchestra
from marsys.coordination.execution.det_nodes import StartNode, EndNode, UserNode
from marsys.coordination.topology import Topology, Node, Edge
from marsys.agents import Agent
from marsys.agents.registry import AgentRegistry
from marsys.models import ModelConfig

config = ModelConfig(type="api", provider="anthropic", name="claude-sonnet-4-6")

assistant = Agent(
    model_config=config,
    name="Assistant",
    instruction=(
        "Help the user. If you need clarification, call `ask_user(question=...)`. "
        "When you have the answer, call `terminate_workflow(answer=...)`."
    ),
)

topology = Topology(
    nodes=[StartNode(), Node("Assistant"), UserNode(), EndNode()],
    edges=[
        Edge("Start", "Assistant"),
        Edge("Assistant", "User"),
        Edge("User", "Assistant"),
        Edge("Assistant", "End"),
    ],
)

result = await Orchestra.run(
    task="Help me plan a trip.",
    topology=topology,
    agent_registry=AgentRegistry,
)
```

Topology gating that results from these edges:

- **Assistant** has edges to `User` and `End` → its tool schema includes `ask_user` and `terminate_workflow` (no `invoke_agent` because there are no peer agents).
- **`UserNode` → Assistant edge** means when the user replies, the resumed branch runs at `Assistant`.

## Legacy migration

Topologies using legacy `entry_point=A`, `exit_points=[X, Y]`, or a `User(Node)` regular node continue to work via the auto-shim in `Orchestra._apply_legacy_topology_shim` (`orchestra.py:296`). The shim:

- Synthesizes `StartNode` + edge `Start → A` for `entry_point=A` and emits `DeprecationWarning`.
- Synthesizes `EndNode` + edges `X → End`, `Y → End` for `exit_points=[X, Y]` and emits `DeprecationWarning`.
- Registers a `UserNode` det-node alongside any legacy `User(Node)` regular node.

Both forms produce the same runtime graph. The legacy form is kept for one release; removal target v0.4. See [ADR-006](../architecture/framework/decisions/ADR-006-deprecation-timeline.md) for the full migration mapping.

## See also

- [Coordination tools](coordination-tools.md) — the four tools and how det-node edges drive their gating.
- [Messages → AgentInput](messages.md#agentinput-the-orchestrator-agent-boundary) — how the orchestrator packages multi-source arrivals.
- [ADR-005: Unified-barrier algorithm](../architecture/framework/decisions/ADR-005-unified-barrier-algorithm.md) — full Branch / Barrier model and fire gates.
- [ADR-006: Deprecation timeline](../architecture/framework/decisions/ADR-006-deprecation-timeline.md) — `entry_point` / `exit_points` / `User(Node)` migration.
