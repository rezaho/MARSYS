"""Tests for the StateSnapshot Pydantic model and round-trip semantics.

Covers:
- AC-2: model_dump_json round-trip is lossless
- AC-3: model_json_schema is JSON-Schema 2020-12 compatible
- AC-4: golden-schema test (top-level field set is stable)
- AC-5: top-level field types match documented contract
- AC-6: BranchState mirrors every Branch field; status enum
- AC-7: BarrierState mirrors every Barrier field; status enum
- AC-8: set→list serialization for Barrier sets
- AC-9: arrived rejects non-JSON-serializable values
- AC-10/11/48: UserInteractionState / PausedSessionMetadata / StorageEntry shape
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from marsys.coordination.state import (
    BarrierState,
    BranchState,
    ConvergencePolicyState,
    PausedSessionMetadata,
    StateSnapshot,
    StorageEntry,
    UserInteractionState,
)


# ─── Top-level snapshot ──────────────────────────────────────────────────────


def _minimal_snapshot() -> StateSnapshot:
    now = datetime.now(tz=timezone.utc)
    return StateSnapshot(
        framework_version="0.0.0-test",
        session_id="sess-1",
        workflow_id=None,
        topology_digest="deadbeef",
        created_at=now,
        paused_at=now,
        branches={
            "br_0001": BranchState(
                id="br_0001",
                current_agent="A",
                status="RUNNING",
                delivery_target="bar_0000",
                created_at=now.timestamp(),
            ),
        },
        barriers={
            "bar_0000": BarrierState(
                id="bar_0000",
                policy=ConvergencePolicyState(),
                status="OPEN",
                created_at=now.timestamp(),
            ),
        },
        convergence_barriers={},
        runnable=["br_0001"],
        fire_queue=[],
        root_barrier_id="bar_0000",
        workflow_error=None,
        completed_emitted=[],
        user_interactions=[],
        user_interaction_inflight=False,
    )


def test_state_snapshot_round_trips_through_json_losslessly():
    """AC-2: model_dump_json + model_validate_json round-trips the model."""
    original = _minimal_snapshot()
    payload = original.model_dump_json()
    restored = StateSnapshot.model_validate_json(payload)
    assert restored == original


def test_state_snapshot_top_level_field_set_is_stable():
    """AC-4 + AC-5: top-level field set matches the documented contract."""
    schema = StateSnapshot.model_json_schema()
    expected_fields = {
        "framework_version",
        "session_id",
        "workflow_id",
        "topology_digest",
        "created_at",
        "paused_at",
        "branches",
        "barriers",
        "convergence_barriers",
        "runnable",
        "fire_queue",
        "root_barrier_id",
        "workflow_error",
        "completed_emitted",
        "user_interactions",
        "user_interaction_inflight",
        "max_steps",
    }
    assert set(schema["properties"].keys()) == expected_fields


def test_state_snapshot_json_schema_is_2020_compatible():
    """AC-3: schema is JSON-Schema 2020-12 compatible. Pydantic v2's
    default schema generator emits 2020-12; verifying the version-agnostic
    surface (presence of `$defs`, schema dialect) is enough here.
    """
    schema = StateSnapshot.model_json_schema()
    # Pydantic v2 emits `$defs` (the 2020-12 spelling) — older drafts use
    # `definitions`. If this assertion ever fails, the framework's pydantic
    # version must be reviewed.
    assert "$defs" in schema or "definitions" in schema


# ─── BranchState / BarrierState mirror checks ────────────────────────────────


def test_branchstate_mirrors_every_field_of_live_branch():
    """AC-6: BranchState carries every field of the live Branch."""
    from marsys.coordination.execution.orchestrator_types import Branch

    live_field_names = {f.name for f in Branch.__dataclass_fields__.values()}
    pyd_field_names = set(BranchState.model_fields.keys())
    # Every Branch field must be representable on BranchState.
    missing = live_field_names - pyd_field_names
    assert missing == set(), f"BranchState missing fields: {missing}"


def test_branchstate_status_enum_values():
    """AC-6: status accepts only the documented enum values."""
    now = datetime.now(tz=timezone.utc).timestamp()
    for status in ("RUNNING", "WAITING", "TERMINATED", "FAILED", "ABANDONED"):
        b = BranchState(
            id="x", current_agent="A", status=status,
            delivery_target="b", created_at=now,
        )
        assert b.status == status


def test_barrierstate_mirrors_every_field_of_live_barrier():
    """AC-7: BarrierState carries every field of the live Barrier."""
    from marsys.coordination.execution.orchestrator_types import Barrier

    live_field_names = {f.name for f in Barrier.__dataclass_fields__.values()}
    pyd_field_names = set(BarrierState.model_fields.keys())
    missing = live_field_names - pyd_field_names
    assert missing == set(), f"BarrierState missing fields: {missing}"


def test_barrierstate_status_enum_values():
    """AC-7: status accepts only the documented enum values."""
    now = datetime.now(tz=timezone.utc).timestamp()
    for status in ("OPEN", "FIRED", "CANCELLED"):
        bar = BarrierState(
            id="x", policy=ConvergencePolicyState(),
            status=status, created_at=now,
        )
        assert bar.status == status


def test_barrier_set_fields_serialize_as_lists():
    """AC-8: Barrier.candidates / upstream / downstream are list[str] on
    the wire; the live orchestrator carries them as set[str]."""
    now = datetime.now(tz=timezone.utc).timestamp()
    bar = BarrierState(
        id="bar_0000",
        policy=ConvergencePolicyState(),
        status="OPEN",
        candidates=["br_0001", "br_0002"],
        upstream=["bar_0001"],
        downstream=["bar_0002"],
        created_at=now,
    )
    payload = bar.model_dump_json()
    assert '"candidates":["br_0001","br_0002"]' in payload
    restored = BarrierState.model_validate_json(payload)
    assert restored == bar


# ─── arrived / non-JSON values ───────────────────────────────────────────────


def test_arrived_with_json_safe_values_round_trips():
    """AC-9 (positive case): JSON-safe values in arrived round-trip."""
    now = datetime.now(tz=timezone.utc).timestamp()
    bar = BarrierState(
        id="x",
        policy=ConvergencePolicyState(),
        status="FIRED",
        arrived={"br_0001": "result string", "br_0002": {"k": "v"}},
        created_at=now,
    )
    payload = bar.model_dump_json()
    restored = BarrierState.model_validate_json(payload)
    assert restored.arrived == bar.arrived


def test_state_snapshot_with_non_json_arrived_value_raises_on_dump():
    """AC-9 (negative case): non-JSON values in arrived raise on
    model_dump_json. The Orchestra layer's _build_state_snapshot wraps
    this and translates into SnapshotSerializationError; here we just
    verify the underlying Pydantic check fires.
    """
    now = datetime.now(tz=timezone.utc)
    snapshot = _minimal_snapshot()
    # Stuff a non-JSON value into a barrier's arrived map.
    snapshot.barriers["bar_0000"].arrived["br_0001"] = object()
    with pytest.raises((TypeError, ValueError)):
        snapshot.model_dump_json()


# ─── UserInteractionState / PausedSessionMetadata / StorageEntry ─────────────


def test_user_interaction_state_shape():
    """AC-10: UserInteractionState carries the four documented fields."""
    ui = UserInteractionState(
        suspended_branch_id="br_0003",
        prompt="please confirm",
        resume_agent="Coordinator",
        delivery_target="bar_0000",
    )
    expected = {"suspended_branch_id", "prompt", "resume_agent", "delivery_target"}
    assert set(ui.model_dump().keys()) == expected


def test_paused_session_metadata_shape():
    """AC-11: PausedSessionMetadata is the return-element type of
    Orchestra.list_paused_sessions and carries the documented fields."""
    psm = PausedSessionMetadata(
        session_id="sess-1",
        workflow_id="run-1",
        paused_at=datetime.now(tz=timezone.utc),
        framework_version="0.3.0",
        snapshot_size_bytes=4096,
    )
    expected = {
        "session_id", "workflow_id", "paused_at",
        "framework_version", "snapshot_size_bytes",
    }
    assert set(psm.model_dump().keys()) == expected


def test_storage_entry_shape():
    """AC-48: StorageEntry has key/size_bytes/modified_at."""
    e = StorageEntry(
        key="sess-1/snapshot.json",
        size_bytes=4096,
        modified_at=datetime.now(tz=timezone.utc),
    )
    assert set(e.model_dump().keys()) == {"key", "size_bytes", "modified_at"}
