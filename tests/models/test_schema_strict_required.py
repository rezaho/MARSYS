"""Regression: the OpenAI Responses API strict-mode schema massaging.

OpenAI's strict ``text.format`` json_schema rejects any schema whose ``required`` omits a property
(``invalid_json_schema``: "'required' ... must include every key in properties"). Pydantic emits
optional/defaulted fields outside ``required``, so structured-output calls 400'd. The OpenAI and
OpenAI-OAuth adapters now compose ``_ensure_all_properties_required`` so ``required`` == every
property on each object node. This is OpenAI-strict-specific — Anthropic's native schema mode does
not impose it, so the transform is composed only in the OpenAI adapters, never globally.
"""
from marsys.models.adapters.base import APIProviderAdapter

req = APIProviderAdapter._ensure_all_properties_required


def test_sets_required_to_every_property():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
        "required": ["a"],  # Pydantic-style: only the no-default field
    }
    assert set(req(schema)["required"]) == {"a", "b"}


def test_recurses_into_nested_objects_and_defs():
    schema = {
        "type": "object",
        "properties": {
            "outer": {"type": "string"},
            "child": {"type": "object", "properties": {"x": {"type": "string"}, "y": {"type": "string"}}, "required": []},
        },
        "$defs": {"Thing": {"type": "object", "properties": {"p": {"type": "string"}, "q": {"type": "string"}}, "required": ["p"]}},
        "required": [],
    }
    out = req(schema)
    assert set(out["required"]) == {"outer", "child"}
    assert set(out["properties"]["child"]["required"]) == {"x", "y"}
    assert set(out["$defs"]["Thing"]["required"]) == {"p", "q"}


def test_does_not_mutate_the_input():
    schema = {"type": "object", "properties": {"a": {"type": "string"}}, "required": []}
    req(schema)
    assert schema["required"] == []  # original untouched (deep copy returned)


def test_object_without_properties_is_left_alone():
    schema = {"type": "object", "additionalProperties": {"type": "string"}}  # a map, no properties
    assert "required" not in req(schema)


def test_composes_with_additional_properties_false():
    # the exact composition the OpenAI adapters apply
    base = {"type": "object", "properties": {"v": {"type": "string"}, "u": {"type": "array", "items": {"type": "string"}}}, "required": ["v"]}
    out = req(APIProviderAdapter._ensure_additional_properties_false(base))
    assert out["additionalProperties"] is False
    assert set(out["required"]) == {"v", "u"}
