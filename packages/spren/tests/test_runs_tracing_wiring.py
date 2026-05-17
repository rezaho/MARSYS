"""Tests for the per-run tracing wiring added in Session 05's `materialize.py`.

Verifies:
- ``materialize_run`` with ``data_dir`` + ``run_id`` sets
  ``execution_config.tracing.enabled = True``.
- The per-run ``output_dir`` matches ``<data-dir>/data/runs/{run_id}``.
- ``include_message_content`` defaults to ``True`` (framework default
  preserved per plan §8.17).
- Without ``data_dir`` / ``run_id`` (e.g., legacy test paths),
  tracing stays at the framework default (``enabled=False``).
"""
from __future__ import annotations

from pathlib import Path

from spren.runs.materialize import materialize_run
from spren.models.topology import TopologySpec
from spren.models.workflow import WorkflowDefinition


def _empty_definition() -> WorkflowDefinition:
    return WorkflowDefinition(
        topology=TopologySpec(nodes=[], edges=[]),
        agents={},
    )


def test_materialize_run_enables_tracing_with_per_run_output_dir(data_dir: Path):
    bundle = materialize_run(
        definition=_empty_definition(),
        data_dir=data_dir,
        run_id="run-tracing-1",
    )
    tracing = bundle.execution_config.tracing
    assert tracing.enabled is True
    expected = data_dir / "data" / "runs" / "run-tracing-1"
    assert tracing.output_dir == str(expected)
    assert tracing.include_message_content is True


def test_materialize_run_creates_per_run_output_dir(data_dir: Path):
    """The materializer ensures the directory exists so the framework
    NDJSON writer doesn't trip on missing-parent on first write."""
    materialize_run(
        definition=_empty_definition(),
        data_dir=data_dir,
        run_id="run-mkdir-1",
    )
    expected = data_dir / "data" / "runs" / "run-mkdir-1"
    assert expected.is_dir()


def test_materialize_run_without_data_dir_leaves_tracing_default(data_dir: Path):
    """Backward-compat path: legacy callers without data_dir/run_id
    don't accidentally turn tracing on."""
    bundle = materialize_run(definition=_empty_definition())
    tracing = bundle.execution_config.tracing
    assert tracing.enabled is False
