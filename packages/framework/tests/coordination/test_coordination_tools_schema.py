"""Tests for CoordinationToolSchemaBuilder — new terminate_workflow / ask_user schemas."""
from __future__ import annotations

import pytest

from marsys.coordination.formats.coordination_tools import (
    COORDINATION_TOOL_NAMES,
    CoordinationToolSchemaBuilder,
    is_coordination_tool,
)


class TestSchemaBuilder:

    def test_terminate_workflow_emitted_when_gated(self):
        schemas = CoordinationToolSchemaBuilder.build_schemas(
            next_agents=["Coordinator"],
            can_terminate_workflow=True,
            can_ask_user=False,
        )
        names = [s["function"]["name"] for s in schemas]
        assert "invoke_agent" in names
        assert "terminate_workflow" in names
        assert "ask_user" not in names

    def test_terminate_workflow_skipped_when_not_gated(self):
        schemas = CoordinationToolSchemaBuilder.build_schemas(
            next_agents=["Coordinator"],
            can_terminate_workflow=False,
            can_ask_user=False,
        )
        names = [s["function"]["name"] for s in schemas]
        assert "terminate_workflow" not in names
        assert "invoke_agent" in names

    def test_ask_user_emitted_when_gated(self):
        schemas = CoordinationToolSchemaBuilder.build_schemas(
            next_agents=["Helper"],
            can_terminate_workflow=False,
            can_ask_user=True,
        )
        names = [s["function"]["name"] for s in schemas]
        assert "ask_user" in names
        assert "terminate_workflow" not in names

    def test_invoke_agent_excludes_det_node_names(self):
        schemas = CoordinationToolSchemaBuilder.build_schemas(
            next_agents=["Researcher", "User", "End", "Start"],
            can_terminate_workflow=False,
            can_ask_user=False,
        )
        names = [s["function"]["name"] for s in schemas]
        assert "invoke_agent" in names
        # The invoke_agent enum should only contain Researcher
        invoke_schema = next(s for s in schemas if s["function"]["name"] == "invoke_agent")
        enum = invoke_schema["function"]["parameters"]["properties"]["invocations"]["items"]["properties"]["agent_name"]["enum"]
        assert enum == ["Researcher"]

    def test_invoke_agent_skipped_when_only_det_node_targets(self):
        schemas = CoordinationToolSchemaBuilder.build_schemas(
            next_agents=["End"],
            can_terminate_workflow=True,
            can_ask_user=False,
        )
        names = [s["function"]["name"] for s in schemas]
        assert "invoke_agent" not in names
        assert "terminate_workflow" in names

    def test_terminate_workflow_uses_answer_param(self):
        schemas = CoordinationToolSchemaBuilder.build_schemas(
            next_agents=[],
            can_terminate_workflow=True,
        )
        s = next(x for x in schemas if x["function"]["name"] == "terminate_workflow")
        params = s["function"]["parameters"]["properties"]
        assert "answer" in params
        assert params["answer"]["type"] == "string"

    def test_ask_user_uses_question_param(self):
        schemas = CoordinationToolSchemaBuilder.build_schemas(
            next_agents=[],
            can_ask_user=True,
        )
        s = next(x for x in schemas if x["function"]["name"] == "ask_user")
        params = s["function"]["parameters"]["properties"]
        assert "question" in params
        assert params["question"]["type"] == "string"

    def test_terminate_workflow_with_output_schema(self):
        output_schema = {
            "properties": {
                "title": {"type": "string"},
                "findings": {"type": "array"},
            },
            "required": ["title"],
        }
        schemas = CoordinationToolSchemaBuilder.build_schemas(
            next_agents=[],
            can_terminate_workflow=True,
            output_schema=output_schema,
        )
        s = next(x for x in schemas if x["function"]["name"] == "terminate_workflow")
        params = s["function"]["parameters"]
        assert params["properties"]["title"]["type"] == "string"
        assert "title" in params["required"]


class TestCoordinationToolNames:

    def test_new_names_recognized(self):
        assert "terminate_workflow" in COORDINATION_TOOL_NAMES
        assert "ask_user" in COORDINATION_TOOL_NAMES
        assert "invoke_agent" in COORDINATION_TOOL_NAMES
        assert "end_conversation" in COORDINATION_TOOL_NAMES

    def test_legacy_alias_still_recognized(self):
        assert "return_final_response" in COORDINATION_TOOL_NAMES
        assert is_coordination_tool("return_final_response") is True

    def test_non_coord_tool(self):
        assert is_coordination_tool("plan_create") is False
