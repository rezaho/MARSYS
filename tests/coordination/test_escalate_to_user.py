"""Unit / mechanism tests for the `escalate_to_user` control directive (ADR-013).

`escalate_to_user` is `ask_user` with the gate AXIS swapped: gated on the per-agent
`can_escalate` grant (NOT a topology User-node edge), enforced at BOTH the schema
offer and validation. The durable suspend/resume itself is framework 16, exercised
end-to-end in tests/integration/test_escalate_to_user.py.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, Mock

from marsys.coordination.formats.coordination_tools import (
    COORDINATION_TOOL_NAMES,
    CoordinationToolSchemaBuilder,
    is_coordination_tool,
)
from marsys.coordination.formats.context import (
    AgentContext,
    CoordinationContext,
    SystemPromptContext,
)
from marsys.coordination.validation.response_validator import (
    ActionType,
    ValidationProcessor,
)
from marsys.coordination.validation.types import ValidationErrorCategory
from marsys.coordination.topology.graph import TopologyGraph


class _FakeAgent:
    """Minimal agent for validator tests — carries name + the can_escalate grant."""

    def __init__(self, name: str = "A", can_escalate: bool = False):
        self.name = name
        self.can_escalate = can_escalate


# ── Schema offer (gated on can_escalate_user) ──────────────────────────────

class TestEscalateSchema:
    def test_in_coordination_tool_names(self):
        assert "escalate_to_user" in COORDINATION_TOOL_NAMES
        assert is_coordination_tool("escalate_to_user") is True

    def test_offered_when_granted(self):
        schemas = CoordinationToolSchemaBuilder.build_schemas(
            next_agents=[], can_escalate_user=True
        )
        assert "escalate_to_user" in [s["function"]["name"] for s in schemas]

    def test_absent_when_not_granted(self):
        schemas = CoordinationToolSchemaBuilder.build_schemas(
            next_agents=[], can_escalate_user=False
        )
        assert "escalate_to_user" not in [s["function"]["name"] for s in schemas]

    def test_default_off(self):
        # Not passing can_escalate_user at all → absent (default off; AC-4/AC-26).
        schemas = CoordinationToolSchemaBuilder.build_schemas(
            next_agents=["Helper"], can_terminate_workflow=True, can_ask_user=True
        )
        assert "escalate_to_user" not in [s["function"]["name"] for s in schemas]

    def test_uses_prompt_param(self):
        schemas = CoordinationToolSchemaBuilder.build_schemas(
            next_agents=[], can_escalate_user=True
        )
        s = next(x for x in schemas if x["function"]["name"] == "escalate_to_user")
        params = s["function"]["parameters"]
        assert "prompt" in params["properties"]
        assert params["properties"]["prompt"]["type"] == "string"
        assert params["required"] == ["prompt"]

    def test_independent_of_ask_user(self):
        only_escalate = [
            s["function"]["name"]
            for s in CoordinationToolSchemaBuilder.build_schemas(
                next_agents=[], can_ask_user=False, can_escalate_user=True
            )
        ]
        assert "escalate_to_user" in only_escalate and "ask_user" not in only_escalate
        only_ask = [
            s["function"]["name"]
            for s in CoordinationToolSchemaBuilder.build_schemas(
                next_agents=[], can_ask_user=True, can_escalate_user=False
            )
        ]
        assert "ask_user" in only_ask and "escalate_to_user" not in only_ask


# ── Type completeness ──────────────────────────────────────────────────────

def test_actiontype_has_escalate_user():
    assert hasattr(ActionType, "ESCALATE_USER")


def test_stepkind_has_escalate_user():
    from marsys.coordination.execution.orchestrator_types import StepKind

    assert "ESCALATE_USER" in StepKind.__args__


# ── Validation (gated on the per-agent can_escalate grant) ──────────────────

@pytest.fixture
def validator():
    g = Mock(spec=TopologyGraph)
    g.get_next_agents = MagicMock(return_value=["End"])
    g.has_edge_to_usernode = MagicMock(return_value=False)
    return ValidationProcessor(g)


class TestEscalateValidation:
    @pytest.mark.asyncio
    async def test_granted_valid_prompt_ok(self, validator):
        res = await validator.validate_coordination_action(
            action="escalate_to_user",
            data={"prompt": "re-authenticate to x.com"},
            agent=_FakeAgent(can_escalate=True),
            branch=None,
            exec_state=None,
        )
        assert res.is_valid
        assert res.action_type == ActionType.ESCALATE_USER
        assert res.parsed_response["prompt"] == "re-authenticate to x.com"

    @pytest.mark.asyncio
    async def test_ungranted_rejected(self, validator):
        res = await validator.validate_coordination_action(
            action="escalate_to_user",
            data={"prompt": "re-auth"},
            agent=_FakeAgent(can_escalate=False),
            branch=None,
            exec_state=None,
        )
        assert not res.is_valid
        assert res.error_category == ValidationErrorCategory.PERMISSION_ERROR.value

    @pytest.mark.asyncio
    async def test_empty_prompt_rejected(self, validator):
        res = await validator.validate_coordination_action(
            action="escalate_to_user",
            data={"prompt": ""},
            agent=_FakeAgent(can_escalate=True),
            branch=None,
            exec_state=None,
        )
        assert not res.is_valid
        assert res.error_category == ValidationErrorCategory.ACTION_ERROR.value

    @pytest.mark.asyncio
    async def test_ask_user_gate_unaffected_by_grant(self, validator):
        # ask_user stays gated on the topology edge, independent of can_escalate:
        # no User edge → rejected even for a can_escalate-granted agent (AC-23/AC-24).
        res = await validator.validate_coordination_action(
            action="ask_user",
            data={"question": "q"},
            agent=_FakeAgent(can_escalate=True),
            branch=None,
            exec_state=None,
        )
        assert not res.is_valid


# ── Instruction surface (separate, grant-gated block) ───────────────────────

def _ctx(
    *,
    can_escalate_user: bool = False,
    can_terminate_workflow: bool = False,
    can_ask_user: bool = False,
) -> SystemPromptContext:
    return SystemPromptContext(
        agent=AgentContext(
            name="A", goal="g", instruction="Do it.", tools={}, tools_schema=[]
        ),
        coordination=CoordinationContext(
            can_terminate_workflow=can_terminate_workflow,
            can_ask_user=can_ask_user,
            can_escalate_user=can_escalate_user,
        ),
    )


class TestEscalateInstructionSurface:
    def _fmt(self):
        from marsys.coordination.formats.json_format.format import JSONResponseFormat

        return JSONResponseFormat()

    def test_present_when_granted(self):
        out = self._fmt()._build_escalate_instructions(_ctx(can_escalate_user=True))
        assert "escalate_to_user" in out
        assert "ESCALATION" in out

    def test_absent_when_not_granted(self):
        assert self._fmt()._build_escalate_instructions(_ctx(can_escalate_user=False)) == ""

    def test_present_even_with_no_completion_capability(self):
        # AC-10/AC-22: a granted agent on a User-less, End-less topology — the
        # topology-gated completion block is empty, but the escalate contract
        # still appears (separate gate).
        fmt = self._fmt()
        ctx = _ctx(
            can_escalate_user=True,
            can_terminate_workflow=False,
            can_ask_user=False,
        )
        assert fmt._build_workflow_completion_instructions(ctx) == ""
        assert "escalate_to_user" in fmt._build_escalate_instructions(ctx)

    def test_assembled_prompt_includes_escalate_when_granted(self):
        # AC-20: the ASSEMBLED system prompt (not just the sub-builder) carries the
        # escalate contract when granted.
        prompt = self._fmt().build_complete_system_prompt(_ctx(can_escalate_user=True))
        assert "escalate_to_user" in prompt

    def test_assembled_prompt_omits_escalate_when_ungranted(self):
        # AC-21: the assembled prompt omits the escalate contract when ungranted.
        prompt = self._fmt().build_complete_system_prompt(_ctx(can_escalate_user=False))
        assert "escalate_to_user" not in prompt


# ── Translate seam: the REAL validate→_translate path (AC-16 / AC-17) ────────
# The deterministic integration suite injects the ESCALATE_USER StepResult
# directly into DeterministicRuntime, bypassing _translate — so a mis-sourced
# prompt or wrong kind would not turn the suite red. These drive the real seam.

class TestEscalateTranslate:
    def _runtime(self):
        from marsys.coordination.execution.real_runtime import RealRuntime

        g = Mock(spec=TopologyGraph)
        g.get_next_agents = MagicMock(return_value=["End"])
        return RealRuntime(
            registry=Mock(), step_executor=Mock(),
            validator=ValidationProcessor(g), topology_graph=g, session_id="t",
        )

    def _marsys_result(self, prompt: str):
        from types import SimpleNamespace

        return SimpleNamespace(
            success=True, coordination_action="escalate_to_user",
            coordination_data={"prompt": prompt}, tool_calls=None, response=None,
        )

    def _branch(self):
        from marsys.coordination.execution.orchestrator_types import Branch

        return Branch(id="b1", current_agent="A", status="RUNNING", delivery_target="root")

    @pytest.mark.asyncio
    async def test_translate_escalate_action_to_step(self):
        # AC-16: the validated action translates to an ESCALATE_USER step.
        # AC-17: its prompt is sourced from the validator output (not a stray local).
        step = await self._runtime()._translate(
            self._marsys_result("re-auth example.com"),
            self._branch(),
            _FakeAgent(can_escalate=True),
        )
        assert step.kind == "ESCALATE_USER"
        assert step.value == "re-auth example.com"

    @pytest.mark.asyncio
    async def test_translate_ungranted_escalate_fails(self):
        # An ungranted agent's escalate action is rejected by validation → the
        # translate seam returns FAIL, never an ESCALATE_USER step (AC-3/AC-12).
        step = await self._runtime()._translate(
            self._marsys_result("re-auth"),
            self._branch(),
            _FakeAgent(can_escalate=False),
        )
        assert step.kind == "FAIL"


# ── Negative space: the deferred vocabulary was NOT pre-built (AC-29) ─────────

def test_actiontype_gained_exactly_escalate_user():
    # The action-type surface gains EXACTLY ESCALATE_USER — no pause/redirect/fail
    # directive was smuggled in (anti-pattern #11 / AC-29).
    assert set(ActionType.__members__) == {
        "INVOKE_AGENT", "PARALLEL_INVOKE", "FINAL_RESPONSE", "TERMINATE_WORKFLOW",
        "ASK_USER", "ESCALATE_USER", "END_CONVERSATION", "ERROR_RECOVERY",
        "TERMINAL_ERROR", "AUTO_RETRY",
    }


def test_stepkind_gained_exactly_escalate_user():
    # The step-kind surface gains EXACTLY ESCALATE_USER (AC-29 / AC-18).
    from marsys.coordination.execution.orchestrator_types import StepKind

    assert set(StepKind.__args__) == {
        "NOOP", "SINGLE_INVOKE", "PARALLEL_INVOKE", "FINAL_RESPONSE",
        "ESCALATE_USER", "FAIL",
    }


def test_default_agent_coordination_set_unchanged():
    # AC-27: a default (ungranted) agent's coordination tool set is EXACTLY the
    # pre-FW18 set — escalate absent, the standard tools present unchanged.
    names = {
        s["function"]["name"]
        for s in CoordinationToolSchemaBuilder.build_schemas(
            next_agents=["Helper"], can_terminate_workflow=True,
            can_ask_user=True, is_conversation_branch=True,
        )
    }
    assert names == {"invoke_agent", "terminate_workflow", "ask_user", "end_conversation"}


@pytest.mark.asyncio
async def test_ask_user_still_accepted_with_user_edge():
    # AC-24 (positive half): ask_user validation is unchanged — WITH a User edge it
    # still validates, independent of can_escalate.
    g = Mock(spec=TopologyGraph)
    g.has_edge_to_usernode = MagicMock(return_value=True)
    g.get_next_agents = MagicMock(return_value=["User"])
    res = await ValidationProcessor(g).validate_coordination_action(
        action="ask_user", data={"question": "q"},
        agent=_FakeAgent(can_escalate=False), branch=None, exec_state=None,
    )
    assert res.is_valid and res.action_type == ActionType.ASK_USER
