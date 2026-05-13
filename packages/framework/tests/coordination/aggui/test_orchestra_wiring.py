"""Tests for Orchestra wiring of the AGGUI translator (AC-79..AC-83).

Verifies the translator is constructed when `ExecutionConfig.aggui.enabled=True`
and exposed as `orchestra.aggui_translator`. The translator must be wired in
`_wire_event_bus()` so resume_session re-attaches it on a fresh EventBus.

We construct Orchestra directly (no agent runs) — these tests verify wiring,
not execution.
"""

from __future__ import annotations

import pytest

pytest.importorskip("ag_ui")

from marsys.agents.registry import AgentRegistry
from marsys.coordination.aggui import AGGUIConfig
from marsys.coordination.aggui.translator import AGGUITranslator
from marsys.coordination.config import ExecutionConfig
from marsys.coordination.orchestra import Orchestra


@pytest.fixture(autouse=True)
def clean_registry():
    AgentRegistry.clear()
    yield
    AgentRegistry.clear()


def test_orchestra_aggui_translator_attribute_exists_after_wire():
    """AC-79: Orchestra exposes `aggui_translator` as a plain attribute."""
    cfg = ExecutionConfig()
    cfg.aggui.enabled = True
    orchestra = Orchestra(agent_registry=AgentRegistry(), execution_config=cfg)
    assert hasattr(orchestra, "aggui_translator")


def test_orchestra_aggui_translator_is_instance_when_enabled():
    """AC-80: When `aggui.enabled=True`, the attribute is an AGGUITranslator."""
    cfg = ExecutionConfig()
    cfg.aggui.enabled = True
    orchestra = Orchestra(agent_registry=AgentRegistry(), execution_config=cfg)
    assert isinstance(orchestra.aggui_translator, AGGUITranslator)


def test_orchestra_aggui_translator_is_none_when_disabled():
    """AC-81: When `aggui.enabled=False` (default), the attribute is None."""
    cfg = ExecutionConfig()
    # Default is False
    assert cfg.aggui.enabled is False
    orchestra = Orchestra(agent_registry=AgentRegistry(), execution_config=cfg)
    assert orchestra.aggui_translator is None


def test_orchestra_no_get_aggui_translator_method():
    """AC-4: There is no `get_aggui_translator(run_id)` method on Orchestra."""
    cfg = ExecutionConfig()
    cfg.aggui.enabled = True
    orchestra = Orchestra(agent_registry=AgentRegistry(), execution_config=cfg)
    assert not hasattr(orchestra, "get_aggui_translator")


def test_aggui_translator_wired_in_wire_event_bus():
    """AC-82: Translator must be wired in `_wire_event_bus()` so resume_session
    re-attaches it on a fresh EventBus. Calling `_wire_event_bus()` again should
    produce a fresh translator (proxy for resume parity).
    """
    cfg = ExecutionConfig()
    cfg.aggui.enabled = True
    orchestra = Orchestra(agent_registry=AgentRegistry(), execution_config=cfg)
    first = orchestra.aggui_translator
    assert first is not None
    # Simulate resume: _wire_event_bus() is called from resume_session as well
    # as __init__. Calling it again must construct a fresh translator wired to
    # whatever EventBus is current.
    from marsys.coordination.event_bus import EventBus
    orchestra.event_bus = EventBus()
    orchestra._wire_event_bus()
    second = orchestra.aggui_translator
    assert second is not None
    assert isinstance(second, AGGUITranslator)
    # Fresh instance (not the same translator bound to the old EventBus)
    assert second is not first


def test_aggui_translator_subscribes_to_orchestra_event_bus():
    """The translator subscribes to events on Orchestra.event_bus, not some other bus."""
    cfg = ExecutionConfig()
    cfg.aggui.enabled = True
    orchestra = Orchestra(agent_registry=AgentRegistry(), execution_config=cfg)
    translator = orchestra.aggui_translator
    assert translator.event_bus is orchestra.event_bus
