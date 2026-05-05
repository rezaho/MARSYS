"""Pytest helpers for the simulator bench.

When a test fails, walk the failing frame's locals for an Orchestrator (or
SimulatorRun wrapping one) and print a structured snapshot of the runtime
state so failures are debuggable from the test output alone — no ad-hoc
`python -c` scripts.
"""
from __future__ import annotations

import pytest


def _find_orchestrator(frame_locals: dict):
    """Look in the failing test frame for an Orchestrator-like object.

    Common variable names: orch, run, sim. SimulatorRun.orchestrator and
    Simulator.orch both expose the orchestrator. Returns None if not found.
    """
    # Direct hit
    for name in ("orchestrator", "orch"):
        obj = frame_locals.get(name)
        if obj is not None and hasattr(obj, "barriers") and hasattr(obj, "branches"):
            return obj
    # SimulatorRun-style wrapper
    for name in ("run", "result"):
        obj = frame_locals.get(name)
        if obj is not None and hasattr(obj, "orchestrator"):
            inner = obj.orchestrator
            if inner is not None and hasattr(inner, "barriers"):
                return inner
    # Simulator wrapper
    for name in ("sim", "simulator"):
        obj = frame_locals.get(name)
        if obj is not None and hasattr(obj, "orch"):
            inner = obj.orch
            if inner is not None and hasattr(inner, "barriers"):
                return inner
    return None


def _format_orchestrator(orch, alias_map: dict | None = None) -> str:
    lines = ["", "─── orchestrator state ──────────────────────────────────────"]

    inv_alias: dict[str, str] = {}
    if alias_map:
        for alias, real in alias_map.items():
            inv_alias.setdefault(real, alias)

    def label(rid: str) -> str:
        a = inv_alias.get(rid)
        return f"{rid}({a})" if a else rid

    root_id = getattr(orch, "root_id", None) or getattr(orch, "root_barrier_id", None)
    lines.append(f"root: {label(root_id) if root_id else '<none>'}")
    if getattr(orch, "_workflow_error", None):
        lines.append(f"workflow_error: {orch._workflow_error}")

    lines.append("")
    lines.append(f"barriers ({len(orch.barriers)}):")
    for bid, bar in orch.barriers.items():
        kind_field = getattr(bar, "kind", None)
        rendezvous = getattr(bar, "rendezvous_node", None) or getattr(bar, "convergence_node", None)
        resolver_branch = getattr(bar, "resolver_branch", None)
        resolver_agent = getattr(bar, "resolver_agent", None)
        descriptor_parts = []
        if kind_field is not None:
            descriptor_parts.append(f"kind={kind_field}")
        if rendezvous:
            descriptor_parts.append(f"rendezvous={rendezvous}")
        if resolver_branch:
            descriptor_parts.append(f"resolver_branch={label(resolver_branch)}")
        if resolver_agent:
            descriptor_parts.append(f"resolver_agent={resolver_agent}")
        descriptor = " ".join(descriptor_parts)
        lines.append(f"  {label(bid)} status={bar.status} {descriptor}")
        if bar.candidates:
            lines.append(f"    candidates: {sorted(label(c) for c in bar.candidates)}")
        if bar.arrived:
            lines.append(f"    arrived:    {sorted(label(c) for c in bar.arrived)}")
        if getattr(bar, 'failed', None):
            lines.append(f"    failed:     {sorted(label(c) for c in bar.failed)}")
        if bar.upstream:
            lines.append(f"    upstream:   {sorted(label(u) for u in bar.upstream)}")
        if bar.downstream:
            lines.append(f"    downstream: {sorted(label(d) for d in bar.downstream)}")

    lines.append("")
    lines.append(f"branches ({len(orch.branches)}):")
    for bid, br in orch.branches.items():
        cof = sorted(label(c) for c in br.candidate_of) if br.candidate_of else []
        wait = label(br.waiting_on) if getattr(br, "waiting_on", None) else None
        deliv = label(br.delivery_target) if br.delivery_target else None
        extras = []
        if wait:
            extras.append(f"waiting_on={wait}")
        if deliv:
            extras.append(f"delivery_target={deliv}")
        if cof:
            extras.append(f"candidate_of={cof}")
        extras.append(f"steps={br.step_count}")
        lines.append(f"  {label(bid)} agent={br.current_agent} status={br.status} {' '.join(extras)}")

    runnable = list(orch.runnable)
    fire_q = list(getattr(orch, "_fire_queue", []))
    lines.append("")
    lines.append(f"runnable: {[label(b) for b in runnable]}")
    lines.append(f"fire_queue: {[label(b) for b in fire_q]}")
    lines.append("─────────────────────────────────────────────────────────────")
    return "\n".join(lines)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()

    if report.when != "call" or report.passed:
        return

    excinfo = call.excinfo
    if excinfo is None:
        return

    # Walk traceback frames from innermost-out to find an orchestrator
    tb = excinfo.tb
    found_orch = None
    found_locals = None
    while tb is not None:
        frame_locals = tb.tb_frame.f_locals
        candidate = _find_orchestrator(frame_locals)
        if candidate is not None:
            found_orch = candidate
            found_locals = frame_locals
            break
        tb = tb.tb_next
    # If innermost didn't have one, scan the rest in case a wrapper helper hides it
    if found_orch is None:
        tb = excinfo.tb
        while tb is not None:
            frame_locals = tb.tb_frame.f_locals
            for v in frame_locals.values():
                if hasattr(v, "barriers") and hasattr(v, "branches") and hasattr(v, "runnable"):
                    found_orch = v
                    found_locals = frame_locals
                    break
                if hasattr(v, "orchestrator") and getattr(v, "orchestrator", None) is not None:
                    inner = v.orchestrator
                    if hasattr(inner, "barriers"):
                        found_orch = inner
                        found_locals = frame_locals
                        break
            if found_orch is not None:
                break
            tb = tb.tb_next

    if found_orch is None:
        return

    alias_map = None
    if found_locals is not None:
        for v in found_locals.values():
            if hasattr(v, "alias_map") and isinstance(getattr(v, "alias_map"), dict):
                alias_map = v.alias_map
                break
        if alias_map is None:
            for v in found_locals.values():
                if hasattr(v, "orch") and hasattr(v, "alias_map"):
                    alias_map = v.alias_map
                    break

    dump = _format_orchestrator(found_orch, alias_map)
    if hasattr(report, "longrepr") and report.longrepr is not None:
        try:
            report.longrepr.addsection("orchestrator state", dump)
        except Exception:
            report.sections.append(("orchestrator state", dump))
    else:
        report.sections.append(("orchestrator state", dump))
