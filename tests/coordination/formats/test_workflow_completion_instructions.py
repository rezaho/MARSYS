"""Tests for ``BaseResponseFormat._build_workflow_completion_instructions``
and its integration into ``build_complete_system_prompt``.

The framework's coordination tools (``terminate_workflow``, ``ask_user``)
are topology-gated — their schemas only appear in the AVAILABLE TOOLS
list when the agent has the corresponding det-node edge. But the tool
descriptions themselves are generic ("Use this when your task is the
final step before returning to the caller") and tell the LLM nothing
about WHEN to call them in the current workflow. Without an analogous
conditional prompt section, a default ``Start → Agent → End`` run
ends ``failed: "insufficient arrivals"`` (or burns to step_limit on
repeated regular tool calls) because the model has no behavioral
contract for invoking ``terminate_workflow``.

The new ``--- WORKFLOW COMPLETION ---`` section in
``build_complete_system_prompt`` closes that gap, conditional on the
same ``CoordinationContext`` flags that gate the schemas.
"""
from __future__ import annotations

from marsys.coordination.formats.context import (
    AgentContext,
    CoordinationContext,
    SystemPromptContext,
)
from marsys.coordination.formats.json_format.format import JSONResponseFormat


def _make_context(
    *,
    can_terminate_workflow: bool = False,
    can_ask_user: bool = False,
    next_agents: list[str] | None = None,
    instruction: str = "Do the thing.",
) -> SystemPromptContext:
    """Build a SystemPromptContext for testing prompt shape."""
    return SystemPromptContext(
        agent=AgentContext(
            name="TestAgent",
            goal="test",
            instruction=instruction,
            tools={},
            tools_schema=[],
        ),
        coordination=CoordinationContext(
            next_agents=next_agents or [],
            can_terminate_workflow=can_terminate_workflow,
            can_ask_user=can_ask_user,
        ),
    )


class TestBuildWorkflowCompletionInstructions:
    """Direct tests of the new ``_build_workflow_completion_instructions``."""

    def test_empty_when_neither_flag_set(self):
        """An agent with no End edge and no User edge gets no completion
        instructions — there are no topology-gated coordination tools to
        document. Returning ``""`` keeps the prompt clean for mid-pipeline
        agents."""
        fmt = JSONResponseFormat()
        ctx = _make_context(
            can_terminate_workflow=False,
            can_ask_user=False,
        )
        result = fmt._build_workflow_completion_instructions(ctx)
        assert result == ""

    def test_terminate_only_emits_termination_section(self):
        """``Start → Agent → End`` (the framework's #1 documented
        failure-mode case): only ``can_terminate_workflow`` is true.
        The section emits the termination contract; ``ask_user`` text
        is absent."""
        fmt = JSONResponseFormat()
        ctx = _make_context(
            can_terminate_workflow=True,
            can_ask_user=False,
        )
        result = fmt._build_workflow_completion_instructions(ctx)
        assert "--- WORKFLOW COMPLETION ---" in result
        assert "--- END WORKFLOW COMPLETION ---" in result
        assert "terminate_workflow" in result
        # Stopping criterion the LLM can self-derive on an open-ended task.
        assert "nothing further to add" in result
        # Both failure modes named — the framework's "insufficient
        # arrivals" error AND the step-limit cancellation that hits
        # when the agent loops on regular tool calls.
        assert "insufficient arrivals" in result
        assert "step limit" in result
        # ask_user section MUST be absent when ``can_ask_user`` is false.
        assert "ask_user" not in result
        assert "User interaction" not in result

    def test_ask_user_only_emits_user_section(self):
        """``Agent → User`` only: the section emits the user-interaction
        contract; termination text is absent."""
        fmt = JSONResponseFormat()
        ctx = _make_context(
            can_terminate_workflow=False,
            can_ask_user=True,
        )
        result = fmt._build_workflow_completion_instructions(ctx)
        assert "--- WORKFLOW COMPLETION ---" in result
        assert "ask_user" in result
        assert "User interaction" in result
        # Termination section MUST be absent when ``can_terminate_workflow``
        # is false.
        assert "terminate_workflow" not in result
        assert "Workflow termination" not in result

    def test_both_flags_emit_both_sections(self):
        """``Agent → User`` + ``Agent → End`` (agent can both ask the
        user mid-task AND terminate the run when done) — both sub-
        sections present in the same WORKFLOW COMPLETION block."""
        fmt = JSONResponseFormat()
        ctx = _make_context(
            can_terminate_workflow=True,
            can_ask_user=True,
        )
        result = fmt._build_workflow_completion_instructions(ctx)
        assert result.count("--- WORKFLOW COMPLETION ---") == 1
        assert result.count("--- END WORKFLOW COMPLETION ---") == 1
        assert "terminate_workflow" in result
        assert "ask_user" in result

    def test_no_stripped_tokens_in_section(self):
        """Regression guard. The framework's ``_strip_schema_hints``
        regex (``base.py``) strips lines containing ``next_action``,
        ``action_input``, or ``Response Structure`` from the agent
        instruction. ``_strip_schema_hints`` only runs on the user-
        authored instruction (not on framework-appended sections), so
        the new section is structurally safe — but a future refactor
        that changes the strip scope would silently delete lines from
        this section. This test guards against that.
        """
        fmt = JSONResponseFormat()
        ctx = _make_context(
            can_terminate_workflow=True,
            can_ask_user=True,
        )
        result = fmt._build_workflow_completion_instructions(ctx)
        assert "next_action" not in result
        assert "action_input" not in result
        assert "Response Structure" not in result


class TestBuildCompleteSystemPromptIntegration:
    """Tests that the new section is wired correctly into
    ``build_complete_system_prompt``."""

    def test_section_included_when_can_terminate(self):
        """The end-to-end prompt assembled by ``build_complete_system_prompt``
        includes the new section when ``can_terminate_workflow`` is set."""
        fmt = JSONResponseFormat()
        ctx = _make_context(can_terminate_workflow=True)
        prompt = fmt.build_complete_system_prompt(ctx)
        assert "--- WORKFLOW COMPLETION ---" in prompt
        assert "terminate_workflow" in prompt

    def test_section_excluded_for_mid_pipeline_agent(self):
        """A mid-pipeline agent (no End edge, no User edge, peers only)
        does NOT see the WORKFLOW COMPLETION section. The peers section
        IS still present because ``next_agents`` is non-empty."""
        fmt = JSONResponseFormat()
        ctx = _make_context(
            can_terminate_workflow=False,
            can_ask_user=False,
            next_agents=["B"],
        )
        prompt = fmt.build_complete_system_prompt(ctx)
        assert "--- WORKFLOW COMPLETION ---" not in prompt

    def test_section_placement_adjacent_to_available_tools(self):
        """Load-bearing: the WORKFLOW COMPLETION section is positioned
        BETWEEN the AVAILABLE TOOLS section and the PEER AGENT DELEGATION
        section. Empirical observation (Session 12 Phase-5): a delivery
        hint placed adjacent to AVAILABLE TOOLS — where the model just
        read the ``terminate_workflow`` tool listing — gets honored more
        reliably than the same text placed further away. PEER AGENT
        DELEGATION sits AFTER our new section so the model reads
        "here is the termination tool, here is how to invoke peers" as a
        coordinated pair."""
        fmt = JSONResponseFormat()
        # Build a context with all three sections active:
        # AVAILABLE TOOLS (always emitted when tools_schema non-empty),
        # WORKFLOW COMPLETION (terminate flag set), and
        # PEER AGENT DELEGATION (next_agents non-empty).
        ctx = SystemPromptContext(
            agent=AgentContext(
                name="MidAgent",
                goal="test",
                instruction="Do the thing.",
                tools={"noop": lambda: None},
                tools_schema=[{
                    "type": "function",
                    "function": {
                        "name": "noop",
                        "description": "no-op",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }],
            ),
            coordination=CoordinationContext(
                next_agents=["DownstreamAgent"],
                can_terminate_workflow=True,
                can_ask_user=False,
            ),
        )
        prompt = fmt.build_complete_system_prompt(ctx)

        tools_pos = prompt.index("--- AVAILABLE TOOLS ---")
        completion_pos = prompt.index("--- WORKFLOW COMPLETION ---")
        peers_pos = prompt.index("--- PEER AGENT DELEGATION ---")

        assert tools_pos < completion_pos, (
            "WORKFLOW COMPLETION must follow AVAILABLE TOOLS — the "
            "completion contract reads as a continuation of the tool list."
        )
        assert completion_pos < peers_pos, (
            "WORKFLOW COMPLETION must precede PEER AGENT DELEGATION — "
            "empirical: termination guidance adjacent to AVAILABLE TOOLS "
            "is honored more reliably than guidance separated from it by "
            "the peer-delegation block."
        )


class TestSpecializedKindUpside:
    """The previous downstream materialize-time helper (now removed)
    skipped specialized agent kinds because they rebuild their own
    instruction from ``params``. Moving the guidance into ``BaseResponseFormat
    .build_complete_system_prompt`` *fixes* that gap — every agent
    regardless of kind goes through the same prompt builder, so
    specialized agents (Browser / FileOperation / DataAnalysis /
    CodeExecution) now get the termination contract too.

    This test asserts the architectural property, not a specialized
    subclass directly: any agent with ``can_terminate_workflow=True``
    in its CoordinationContext gets the section, regardless of what
    its instruction field contains.
    """

    def test_section_present_for_arbitrary_instruction_text(self):
        """The section ships with the framework's prompt builder,
        independent of the agent's instruction. A specialized agent's
        auto-generated instruction (e.g. ``BrowserAgent``'s build) is
        unaffected — it lands in the instruction slot, and the
        framework still appends WORKFLOW COMPLETION when the flag is
        set."""
        fmt = JSONResponseFormat()
        ctx = _make_context(
            can_terminate_workflow=True,
            instruction=(
                "You are a browser-control agent. Use the navigate "
                "tool to move between pages..."
            ),
        )
        prompt = fmt.build_complete_system_prompt(ctx)
        assert "--- WORKFLOW COMPLETION ---" in prompt
        assert "terminate_workflow" in prompt
