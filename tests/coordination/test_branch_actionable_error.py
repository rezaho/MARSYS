"""`_branch_actionable_error` — BranchResult.error carries the CAUSE.

A FAILED branch's underlying error (ModelAPIError, step_executor error,
max_steps, FAIL kind) is recorded in ``barrier.failed[branch_id]`` by the
orchestrator; the branch's own ``.status`` is only the lifecycle state.
``BranchResult.error`` must surface the recorded cause so consumers don't
get a vacuous "branch ended in status FAILED".
"""

from types import SimpleNamespace

from marsys.coordination.orchestra import _branch_actionable_error


def _branch(status="FAILED", branch_id="b1", delivery_target=None):
    return SimpleNamespace(status=status, id=branch_id, delivery_target=delivery_target)


def _barrier(failed=None):
    return SimpleNamespace(failed=failed or {})


def test_terminated_branch_has_no_error():
    assert _branch_actionable_error(_branch(status="TERMINATED"), {}) is None


def test_cause_read_from_delivery_target_barrier():
    barriers = {
        "bar-1": _barrier({"b1": "ModelAPIError: 400 temperature not permitted"}),
    }
    err = _branch_actionable_error(_branch(delivery_target="bar-1"), barriers)
    assert err == "ModelAPIError: 400 temperature not permitted"


def test_cause_found_by_fallback_scan_when_target_missing():
    barriers = {
        "other": _barrier({"b1": "step_executor: max_steps exceeded"}),
    }
    err = _branch_actionable_error(_branch(delivery_target="bar-gone"), barriers)
    assert err == "step_executor: max_steps exceeded"


def test_no_recorded_cause_falls_back_to_status_message():
    err = _branch_actionable_error(_branch(status="ABANDONED"), {"bar": _barrier()})
    assert err == "branch ended in status ABANDONED"
