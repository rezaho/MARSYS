"""Pytest helpers for MARSYS orchestrator integration tests.

When a test fails, walk the failing frame's locals for an Orchestrator and
print a structured snapshot of barriers, branches, and queues so failures
are debuggable from the test output alone.

Mirrors the conftest at research/orchestration/simulator/suites/conftest.py.
The orchestrator implementation is the same; only the topology and runtime
adapters differ.
"""
from __future__ import annotations

import pytest


def _find_orchestrator(frame_locals: dict):
    """Look in the failing test frame for an Orchestrator-like object."""
    for name in ("orchestrator", "orch"):
        obj = frame_locals.get(name)
        if obj is not None and hasattr(obj, "barriers") and hasattr(obj, "branches"):
            return obj
    for name in ("run", "result"):
        obj = frame_locals.get(name)
        if obj is not None and hasattr(obj, "orchestrator"):
            inner = obj.orchestrator
            if inner is not None and hasattr(inner, "barriers"):
                return inner
    return None


def _format_orchestrator(orch) -> str:
    lines = ["", "─── orchestrator state ──────────────────────────────────────"]

    root_id = getattr(orch, "root_barrier_id", None)
    lines.append(f"root: {root_id or '<none>'}")
    if getattr(orch, "_workflow_error", None):
        lines.append(f"workflow_error: {orch._workflow_error}")

    lines.append("")
    lines.append(f"barriers ({len(orch.barriers)}):")
    for bid, bar in orch.barriers.items():
        kind_field = getattr(bar, "kind", None)
        rendezvous = getattr(bar, "rendezvous_node", None)
        resolver_branch = getattr(bar, "resolver_branch", None)
        resolver_agent = getattr(bar, "resolver_agent", None)
        descriptor_parts = []
        if kind_field is not None:
            descriptor_parts.append(f"kind={kind_field}")
        if rendezvous:
            descriptor_parts.append(f"rendezvous={rendezvous}")
        if resolver_branch:
            descriptor_parts.append(f"resolver_branch={resolver_branch}")
        if resolver_agent:
            descriptor_parts.append(f"resolver_agent={resolver_agent}")
        lines.append(f"  {bid} status={bar.status} {' '.join(descriptor_parts)}")
        if bar.candidates:
            lines.append(f"    candidates: {sorted(bar.candidates)}")
        if bar.arrived:
            lines.append(f"    arrived:    {sorted(bar.arrived)}")
        if getattr(bar, "failed", None):
            lines.append(f"    failed:     {sorted(bar.failed)}")
        if bar.upstream:
            lines.append(f"    upstream:   {sorted(bar.upstream)}")
        if bar.downstream:
            lines.append(f"    downstream: {sorted(bar.downstream)}")

    lines.append("")
    lines.append(f"branches ({len(orch.branches)}):")
    for bid, br in orch.branches.items():
        cof = sorted(br.candidate_of) if br.candidate_of else []
        wait = br.waiting_on if getattr(br, "waiting_on", None) else None
        deliv = br.delivery_target if br.delivery_target else None
        extras = []
        if wait:
            extras.append(f"waiting_on={wait}")
        if deliv:
            extras.append(f"delivery_target={deliv}")
        if cof:
            extras.append(f"candidate_of={cof}")
        extras.append(f"steps={br.step_count}")
        lines.append(f"  {bid} agent={br.current_agent} status={br.status} {' '.join(extras)}")

    runnable = list(orch.runnable)
    fire_q = list(getattr(orch, "_fire_queue", []))
    lines.append("")
    lines.append(f"runnable: {runnable}")
    lines.append(f"fire_queue: {fire_q}")
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

    tb = excinfo.tb
    found = None
    while tb is not None:
        candidate = _find_orchestrator(tb.tb_frame.f_locals)
        if candidate is not None:
            found = candidate
            break
        tb = tb.tb_next
    if found is None:
        tb = excinfo.tb
        while tb is not None:
            for v in tb.tb_frame.f_locals.values():
                if hasattr(v, "barriers") and hasattr(v, "branches") and hasattr(v, "runnable"):
                    found = v
                    break
            if found is not None:
                break
            tb = tb.tb_next

    if found is None:
        return

    dump = _format_orchestrator(found)
    if hasattr(report, "longrepr") and report.longrepr is not None:
        try:
            report.longrepr.addsection("orchestrator state", dump)
        except Exception:
            report.sections.append(("orchestrator state", dump))
    else:
        report.sections.append(("orchestrator state", dump))
