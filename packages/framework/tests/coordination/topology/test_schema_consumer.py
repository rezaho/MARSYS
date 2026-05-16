"""Multi-consumer JSON Schema test.

Validates that the generated JSON Schema works with the ``jsonschema`` library
(not just Pydantic's own validators). This proves a non-Python consumer can
validate a wire payload against the canonical schema without round-tripping
through Python.
"""

from __future__ import annotations

import json

import jsonschema
import pytest

from marsys.coordination.topology.serialize import (
    JSON_SCHEMA_DIALECT_2020_12,
    workflow_definition_schema,
)


def _valid_payload() -> dict:
    return {
        "topology": {
            "nodes": [
                {"name": "A", "node_type": "agent", "agent_ref": "Worker"},
                {"name": "B", "node_type": "user"},
            ],
            "edges": [
                {"source": "A", "target": "B", "edge_type": "invoke"},
            ],
            "metadata": {},
            "rules": [],
        },
        "agents": {
            "Worker": {
                "name": "Worker",
                "goal": "do work",
                "instruction": "follow the plan",
                "agent_model": {
                    "type": "api",
                    "name": "gpt-4o",
                    "provider": "openai",
                },
                "tools": [],
            }
        },
        "execution_config": {},
    }


def test_schema_dialect_uri_is_2020_12():
    schema = workflow_definition_schema()
    assert schema["$schema"] == JSON_SCHEMA_DIALECT_2020_12


def test_valid_payload_passes_jsonschema_validation():
    schema = workflow_definition_schema()
    payload = _valid_payload()
    # Raises jsonschema.ValidationError on mismatch.
    jsonschema.validate(instance=payload, schema=schema)


def test_intentionally_broken_payload_fails_jsonschema_validation():
    schema = workflow_definition_schema()

    # Missing required `topology` field.
    bad_payload = {"agents": {}}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=bad_payload, schema=schema)


def test_unknown_node_type_value_fails_jsonschema_validation():
    schema = workflow_definition_schema()
    payload = _valid_payload()
    payload["topology"]["nodes"][0]["node_type"] = "unknown_value"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=payload, schema=schema)


def test_extra_field_fails_jsonschema_validation():
    schema = workflow_definition_schema()
    payload = _valid_payload()
    payload["topology"]["nodes"][0]["arbitrary_unknown"] = "boom"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=payload, schema=schema)


def test_schema_is_json_round_trippable():
    """A non-Python tool would dump the schema to JSON and re-load it."""
    schema = workflow_definition_schema()
    payload = json.dumps(schema)
    re_parsed = json.loads(payload)
    assert re_parsed["$schema"] == JSON_SCHEMA_DIALECT_2020_12
