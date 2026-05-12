"""MarsysRunState + JSON Patch delta round-trip tests."""

from __future__ import annotations

import pytest

pytest.importorskip("ag_ui")

import jsonpatch

from marsys.coordination.aggui.state import (
    BarrierState,
    BranchState,
    MarsysRunState,
    PlanItemState,
    PlanState,
    compute_delta,
)


def test_empty_state_round_trip():
    s = MarsysRunState()
    assert s.schema_version == 1
    assert s.branches == {}
    assert s.barriers == {}
    assert s.plans == {}
    assert s.total_steps == 0
    # Round-trip through JSON
    assert MarsysRunState.model_validate(s.model_dump()) == s


def test_add_branch_produces_add_op():
    s1 = MarsysRunState()
    s2 = s1.model_copy(deep=True)
    s2.branches["br1"] = BranchState(
        branch_id="br1",
        branch_name="Researcher",
        current_agent="Researcher",
        status="RUNNING",
    )
    patch = compute_delta(s1, s2)
    add_ops = [op for op in patch if op["op"] == "add" and "branches" in op["path"]]
    assert add_ops, f"expected an add op on /branches, got {patch}"


def test_change_branch_current_agent_produces_replace_op():
    s1 = MarsysRunState(
        branches={
            "br1": BranchState(
                branch_id="br1",
                branch_name="X",
                current_agent="Researcher",
                status="RUNNING",
            )
        }
    )
    s2 = s1.model_copy(deep=True)
    s2.branches["br1"].current_agent = "Writer"
    patch = compute_delta(s1, s2)
    assert any(
        op["op"] == "replace" and "current_agent" in op["path"] for op in patch
    ), patch


def test_add_plan_item_produces_add_op():
    s1 = MarsysRunState(
        plans={
            "Agent": PlanState(
                agent_name="Agent",
                goal=None,
                items=[],
            )
        }
    )
    s2 = s1.model_copy(deep=True)
    s2.plans["Agent"].items.append(
        PlanItemState(item_id="i1", title="Step 1", status="pending")
    )
    patch = compute_delta(s1, s2)
    assert any(op["op"] == "add" and "items" in op["path"] for op in patch), patch


def test_remove_plan_item_produces_remove_op():
    s1 = MarsysRunState(
        plans={
            "Agent": PlanState(
                agent_name="Agent",
                goal=None,
                items=[
                    PlanItemState(item_id="i1", title="A", status="pending"),
                    PlanItemState(item_id="i2", title="B", status="pending"),
                ],
            )
        }
    )
    s2 = s1.model_copy(deep=True)
    s2.plans["Agent"].items = [s2.plans["Agent"].items[1]]
    patch = compute_delta(s1, s2)
    assert any(op["op"] in {"remove", "replace"} for op in patch), patch


def test_increment_total_steps_produces_replace_op():
    s1 = MarsysRunState()
    s2 = s1.model_copy(deep=True)
    s2.total_steps = 5
    patch = compute_delta(s1, s2)
    assert any(
        op["op"] == "replace" and op["path"] == "/total_steps" and op["value"] == 5
        for op in patch
    ), patch


def test_delta_then_snapshot_round_trip():
    s1 = MarsysRunState()
    s2 = MarsysRunState(
        branches={
            "br1": BranchState(
                branch_id="br1",
                branch_name="Researcher",
                current_agent="Researcher",
                status="RUNNING",
            )
        },
        barriers={
            "g1": BarrierState(
                barrier_id="g1",
                status="OPEN",
                group_id="g1",
                total_count=2,
            )
        },
        total_steps=3,
    )
    patch = compute_delta(s1, s2)
    applied = jsonpatch.apply_patch(s1.model_dump(mode="json"), patch)
    assert applied == s2.model_dump(mode="json")


def test_equal_states_produce_empty_patch():
    s1 = MarsysRunState(total_steps=7)
    s2 = MarsysRunState(total_steps=7)
    assert compute_delta(s1, s2) == []


def test_barrier_trimmed_schema():
    """v0.3 BarrierState must NOT carry arrived_count or resolver_branch."""
    b = BarrierState(barrier_id="g1", status="OPEN")
    fields = set(b.model_fields.keys())
    assert "arrived_count" not in fields
    assert "resolver_branch" not in fields
    # But these are present:
    assert "barrier_id" in fields
    assert "status" in fields
    assert "rendezvous_node" in fields
    assert "group_id" in fields
    assert "successful_count" in fields
    assert "total_count" in fields
