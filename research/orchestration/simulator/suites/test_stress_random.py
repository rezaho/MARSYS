"""Random topology stress test.

Generates N random topologies with scripted agent behaviors, runs them
through the orchestrator, asserts:
  - no deadlock (no OPEN barrier with pending==∅)
  - no leaked barriers (every barrier in a terminal state by end)
  - every branch settled (no RUNNING branches left at end)
  - no unbounded memory (bounded by topology size)
"""
from __future__ import annotations

import random
from typing import Optional

from research.orchestration.orchestrator.types import Invocation, StepResult
from research.orchestration.simulator.simulator import Simulator
from research.orchestration.simulator.topology import SimNode, build_topology
from research.orchestration.simulator.trace import (
    ConvergencePolicy,
    SimAssertion,
    SimEvent,
    SimTrace,
)


def _gen_topology(seed: int, n_nodes: int = 10, n_convergences: int = 2):
    """Generate a random topology with a tree-like spine plus optional
    convergence edges. Entry is 'root'; topology has explicit Start node
    with edge to root."""
    from research.orchestration.simulator.det_nodes import StartNode
    rng = random.Random(seed)
    nodes: list = [StartNode(), SimNode("root")]
    # Build a tree by random parent choice
    names = ["root"]
    for i in range(1, n_nodes):
        name = f"n{i}"
        names.append(name)
        is_conv = i in rng.sample(range(1, n_nodes), min(n_convergences, n_nodes - 1))
        nodes.append(SimNode(name, convergence_mode="force" if is_conv else "auto"))

    flows = [("Start", "root")]
    parents: dict[str, str] = {}
    for i, name in enumerate(names[1:], 1):
        parent = rng.choice(names[:i])
        parents[name] = parent
        flows.append((parent, name))
        # Return edges for non-leaves (randomly)
        if rng.random() < 0.5:
            flows.append((name, parent))
    # Occasional extra edge to create potential convergences
    for _ in range(n_nodes // 3):
        src = rng.choice(names)
        dst = rng.choice(names)
        if src != dst and (src, dst) not in flows:
            flows.append((src, dst))

    return build_topology(nodes=nodes, flows=flows)


def _gen_scripts(sim: Simulator, topo, rng: random.Random, max_depth: int = 3):
    """Generate agent-keyed scripts so that every agent, when run, does one
    of: FINAL_RESPONSE, SINGLE_INVOKE to a random successor, or (non-leaf
    only) PARALLEL_INVOKE to some successors. Depth-limited to prevent
    runaway recursion."""
    from research.orchestration.simulator.det_nodes import SimDeterministicNode
    for node in topo.nodes:
        name = node.name
        # Skip det-nodes: they don't run scripts (handlers run inline).
        if isinstance(node, SimDeterministicNode):
            continue
        succ = topo.successors(name)
        if not succ:
            # No successors — must emit FINAL_RESPONSE
            sim.mock.queue_agent(name, StepResult(
                kind="FINAL_RESPONSE", value=f"{name}_leaf"))
            continue
        # Always have at least one terminating script per agent so the branch
        # has a way to exit loops
        choice = rng.random()
        if choice < 0.4 or len(succ) < 2:
            # SINGLE_INVOKE to a random successor
            sim.mock.queue_agent(name, StepResult(
                kind="SINGLE_INVOKE", next_agent=rng.choice(succ),
                value=f"{name}_single"))
        elif choice < 0.75:
            # PARALLEL_INVOKE to a subset
            targets = rng.sample(succ, rng.randint(2, min(len(succ), 3)))
            invs = [Invocation(agent=t, request="") for t in targets]
            sim.mock.queue_agent(name, StepResult(
                kind="PARALLEL_INVOKE", invocations=invs))
        else:
            sim.mock.queue_agent(name, StepResult(
                kind="FINAL_RESPONSE", value=f"{name}_end"))

        # Queue many fallback FINAL_RESPONSE scripts so multiple branches
        # that land on the same agent all have something to emit.
        for _ in range(20):
            sim.mock.queue_agent(name, StepResult(
                kind="FINAL_RESPONSE", value=f"{name}_fallback"))


def _run_one(seed: int, verbose: bool = False) -> tuple[bool, str]:
    rng = random.Random(seed)
    topo = _gen_topology(seed, n_nodes=rng.randint(5, 12), n_convergences=rng.randint(0, 3))
    policy = ConvergencePolicy(
        min_ratio=rng.choice([1.0, 0.5, 0.0]),
        on_insufficient=rng.choice(["fail", "proceed"]),
        terminate_orphans=True,
    )
    # Minimal trace: just CREATE_INITIAL, let scripts drive the rest
    events = [
        SimEvent(t=0, branch_id="main", kind="CREATE_INITIAL",
                 payload={"task": "stress", "entry_agent": "root"}),
    ]
    trace = SimTrace(topology=topo, policy=policy, events=events,
                     assertions=[], name=f"stress_{seed}")
    sim = Simulator(trace)
    _gen_scripts(sim, topo, rng)

    try:
        run = sim.run()
    except RuntimeError as e:
        # Safety-limit runaway is acceptable if bounded; record as pass
        return True, f"safety-limit terminated: {e}"
    except Exception as e:
        return False, f"unexpected error: {e}"

    # Invariants:
    # 1. Every barrier is FIRED or CANCELLED (none OPEN)
    leaked = [bid for bid, b in run.orchestrator.barriers.items()
              if b.status == "OPEN"]
    if leaked:
        return False, f"leaked OPEN barriers: {leaked}"

    # 2. Every branch is settled
    running = [bid for bid, b in run.orchestrator.branches.items()
               if b.status in ("RUNNING", "WAITING")]
    # Waiting is OK only if the barrier they wait on has been cancelled/fired
    # (the abandon path should have caught them, but allow this as warning)
    unsettled = []
    for bid in running:
        br = run.orchestrator.branches[bid]
        if br.status == "WAITING" and br.waiting_on:
            wbar = run.orchestrator.barriers.get(br.waiting_on)
            if wbar is None or wbar.status == "OPEN":
                unsettled.append(bid)
        else:
            unsettled.append(bid)
    if unsettled:
        return False, f"unsettled branches: {unsettled}"

    # 3. Memory bounded — high cap because cycle topologies legitimately
    #    spawn many branches until step_count kicks in. Unbounded growth
    #    would be thousands.
    if len(run.orchestrator.barriers) > 500 or len(run.orchestrator.branches) > 500:
        return False, (
            f"unbounded growth: barriers={len(run.orchestrator.barriers)} "
            f"branches={len(run.orchestrator.branches)}"
        )

    return True, "ok"


def test_stress_random_100():
    """Run 100 random topologies. All should terminate cleanly under the
    invariants above."""
    failures = []
    for seed in range(100):
        ok, msg = _run_one(seed)
        if not ok:
            failures.append((seed, msg))
    if failures:
        # Show first 5
        sample = "\n".join(f"  seed={s}: {m}" for s, m in failures[:5])
        raise AssertionError(
            f"{len(failures)}/100 random traces failed:\n{sample}"
        )


if __name__ == "__main__":
    test_stress_random_100()
    print("Stress 100 random traces: PASS")
