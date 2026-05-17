"""Python AST importer tests."""
from __future__ import annotations

import io
from pathlib import Path

import pytest

from spren.importers.python_workflow import (
    MAX_IMPORT_BYTES,
    PythonImportError,
    parse_python_workflow,
)

FIXTURES = Path(__file__).parent / "fixtures" / "python_workflows"


def _read(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_valid_minimal_round_trip():
    result = parse_python_workflow(_read("valid_minimal.py"))
    defn, name, description = result.definition, result.name, result.description
    assert name == "Minimal Research Pipeline."
    assert description and "AgentSpec" in description
    assert result.warnings == []
    assert set(defn.agents.keys()) == {"Researcher", "Writer"}

    researcher = defn.agents["Researcher"]
    assert researcher.agent_model.provider == "openai"
    assert researcher.agent_model.name == "gpt-4o"
    assert researcher.tools == ["search_web"]
    assert researcher.memory_retention == "session"
    assert researcher.allowed_peers == ["Writer"]

    writer = defn.agents["Writer"]
    assert writer.agent_model.provider == "anthropic"
    assert writer.agent_model.name == "claude-opus-4-7"
    assert writer.tools == ["write_doc"]

    assert {n.name for n in defn.topology.nodes} == {"Researcher", "Writer"}
    writer_node = next(n for n in defn.topology.nodes if n.name == "Writer")
    assert writer_node.is_convergence_point is True
    assert writer_node.agent_ref == "Writer"

    edge = defn.topology.edges[0]
    assert edge.source == "Researcher"
    assert edge.target == "Writer"
    assert edge.edge_type == "invoke"

    assert defn.execution_config.convergence_timeout == 120.0
    assert defn.execution_config.user_interaction == "none"


def test_valid_with_constants_resolves_name_refs():
    result = parse_python_workflow(_read("valid_with_constants.py"))
    defn = result.definition
    assert "Reza" in defn.agents
    assert defn.agents["Reza"].agent_model.name == "claude-opus-4-7"
    assert defn.agents["Reza"].agent_model.provider == "anthropic"


def test_rejects_dict_dsl():
    with pytest.raises(PythonImportError) as exc:
        parse_python_workflow(_read("invalid_dict_dsl.py"))
    assert exc.value.details["reason"] == "dict_dsl_topology"


def test_rejects_dynamic_topology():
    with pytest.raises(PythonImportError) as exc:
        parse_python_workflow(_read("invalid_dynamic_topology.py"))
    # Either the comprehension itself is rejected or the resolved nodes/edges
    # list isn't a literal — both are valid rejection paths.
    assert exc.value.details["reason"] in {"list_comp", "nodes_not_list"}


def test_rejects_exec_call():
    with pytest.raises(PythonImportError) as exc:
        parse_python_workflow(_read("invalid_exec.py"))
    assert exc.value.details["reason"] == "forbidden_call"


@pytest.mark.parametrize("fixture", ["invalid_eval.py", "invalid_compile.py", "invalid_dunder_import.py"])
def test_rejects_eval_compile_and_dunder_import(fixture):
    """eval / compile / __import__ are rejected (not just exec)."""
    with pytest.raises(PythonImportError) as exc:
        parse_python_workflow(_read(fixture))
    assert exc.value.details["reason"] == "forbidden_call"


def test_rejects_fstring_in_user_field():
    """f-strings in user-facing fields are rejected."""
    with pytest.raises(PythonImportError) as exc:
        parse_python_workflow(_read("invalid_fstring.py"))
    assert exc.value.details["reason"] == "fstring"


def test_rejects_non_utf8_at_endpoint(client, auth_headers):
    """non-UTF-8 file uploaded to the endpoint → 422 + PYTHON_IMPORT_REJECTED."""
    bad = b"\xff\xfe\xfd # not utf-8 prefix"
    files = {"file": ("bad.py", io.BytesIO(bad), "text/x-python")}
    r = client.post("/v1/workflows/import-python", files=files, headers=auth_headers)
    assert r.status_code == 422
    body = r.json()
    assert body["error"]["code"] == "PYTHON_IMPORT_REJECTED"
    assert body["error"]["details"]["reason"] == "non_utf8"


def test_rejects_subscript_in_tools_dict_value():
    """Subscript expressions inside tools= dict values are rejected."""
    src = """
\"\"\"x\"\"\"
from marsys.agents import Agent
from marsys.models import ModelConfig
from marsys.coordination.topology.core import Node, NodeKind, Topology
TOOLS = {"a": 1, "b": 2}
def stub(): pass
agent = Agent(
    name="R",
    goal="g",
    instruction="i",
    model_config=ModelConfig(type="api", name="gpt-4o", provider="openai"),
    tools={"x": TOOLS["a"]},
)
topology = Topology(nodes=[Node(name="R", kind=NodeKind.AGENT, agent_ref="R")], edges=[])
"""
    with pytest.raises(PythonImportError) as exc:
        parse_python_workflow(src)
    # tools= reaches a non-Name value via dict-access; the parser only accepts
    # name references or constants. Either tools_key_not_string or a more
    # specific reason — check it's a structured rejection.
    assert exc.value.details.get("reason"), exc.value.details


def test_importer_does_not_execute_source(tmp_path: Path):
    """Side-effect-only assertion: parsing must not run the file's code."""
    marker = Path("/tmp/spren-importer-must-not-execute.txt")
    if marker.exists():
        marker.unlink()
    try:
        with pytest.raises(PythonImportError):
            parse_python_workflow(_read("invalid_exec.py"))
        assert not marker.exists(), "importer executed the source file"
    finally:
        if marker.exists():
            marker.unlink()


def test_rejects_no_topology():
    src = """
\"\"\"empty\"\"\"
from marsys.agents import Agent
from marsys.models import ModelConfig
def stub(): pass
agent = Agent(name="R", goal="g", instruction="i", model_config=ModelConfig(type="api", name="x", provider="openai"), tools={"x": stub})
"""
    with pytest.raises(PythonImportError) as exc:
        parse_python_workflow(src)
    assert exc.value.details["reason"] == "no_topology"


def test_rejects_no_agents():
    with pytest.raises(PythonImportError) as exc:
        parse_python_workflow('"""empty"""\n')
    assert exc.value.details["reason"] == "no_agents"


def test_rejects_class_def():
    src = '"""x"""\nclass Foo: pass\n'
    with pytest.raises(PythonImportError) as exc:
        parse_python_workflow(src)
    assert exc.value.details["reason"] == "class_def"


def test_rejects_module_level_list_comp():
    src = '"""x"""\nx = [i for i in range(3)]\n'
    with pytest.raises(PythonImportError) as exc:
        parse_python_workflow(src)
    assert exc.value.details["reason"] == "list_comp"


def test_drops_api_key_silently():
    """Imported ModelConfig(api_key="...") must not propagate to ModelConfigSpec."""
    src = """
\"\"\"x\"\"\"
from marsys.agents import Agent
from marsys.models import ModelConfig
from marsys.coordination.topology.core import Node, NodeKind, Topology
def stub(): pass
agent = Agent(
    name="R",
    goal="g",
    instruction="i",
    model_config=ModelConfig(type="api", name="gpt-4o", provider="openai", api_key="sk-secret"),
    tools={"stub": stub},
)
topology = Topology(
    nodes=[Node(name="R", kind=NodeKind.AGENT, agent_ref="R")],
    edges=[],
)
"""
    result = parse_python_workflow(src)
    spec = result.definition.agents["R"].agent_model
    # api_key isn't a field on ModelConfigSpec — model_dump must not surface it.
    dumped = spec.model_dump()
    assert "api_key" not in dumped


# --- HTTP-boundary checks via the import-python endpoint ---


def test_endpoint_rejects_oversized_upload(client, auth_headers):
    """oversize file → 422 + PYTHON_IMPORT_REJECTED, fast-fail before AST parse."""
    payload = b"#" * (MAX_IMPORT_BYTES + 100)
    files = {"file": ("big.py", io.BytesIO(payload), "text/x-python")}
    r = client.post("/v1/workflows/import-python", files=files, headers=auth_headers)
    assert r.status_code == 422
    body = r.json()
    assert body["error"]["code"] == "PYTHON_IMPORT_REJECTED"
    assert body["error"]["details"]["reason"] == "too_large"


def test_endpoint_imports_valid_minimal(client, auth_headers):
    src = _read("valid_minimal.py").encode("utf-8")
    files = {"file": ("valid_minimal.py", io.BytesIO(src), "text/x-python")}
    r = client.post("/v1/workflows/import-python", files=files, headers=auth_headers)
    assert r.status_code == 201, r.text
    envelope = r.json()
    assert envelope["warnings"] == []
    body = envelope["workflow"]
    assert body["provenance"] == "code_import"
    assert body["provenance_metadata"]["source_filename"] == "valid_minimal.py"
    assert "sha256" in body["provenance_metadata"]
    assert body["name"] == "Minimal Research Pipeline."
    assert set(body["definition"]["agents"].keys()) == {"Researcher", "Writer"}


def test_endpoint_surfaces_pattern_conversion_warning(client, auth_headers):
    """alternating/symmetric edges auto-convert to bidirectional
    with a non-blocking warning emitted in the import response.
    """
    src = _read("valid_with_alt_pattern.py").encode("utf-8")
    files = {"file": ("valid_with_alt_pattern.py", io.BytesIO(src), "text/x-python")}
    r = client.post("/v1/workflows/import-python", files=files, headers=auth_headers)
    assert r.status_code == 201, r.text
    envelope = r.json()
    assert len(envelope["warnings"]) == 1
    warning = envelope["warnings"][0]
    assert warning["code"] == "pattern_auto_converted"
    assert warning["original_pattern"] == "alternating"
    assert warning["source"] == "A"
    assert warning["target"] == "B"

    workflow = envelope["workflow"]
    edges = workflow["definition"]["topology"]["edges"]
    assert len(edges) == 1
    assert edges[0]["bidirectional"] is True
    assert edges[0]["pattern"] is None
    assert edges[0]["metadata"]["spren_converted_from"] == "alternating"


def test_endpoint_rejects_invalid_dict_dsl(client, auth_headers):
    src = _read("invalid_dict_dsl.py").encode("utf-8")
    files = {"file": ("invalid_dict_dsl.py", io.BytesIO(src), "text/x-python")}
    r = client.post("/v1/workflows/import-python", files=files, headers=auth_headers)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "PYTHON_IMPORT_REJECTED"


def test_endpoint_rejects_invalid_exec(client, auth_headers):
    """endpoint-level rejection of exec() returns 422 + PYTHON_IMPORT_REJECTED."""
    src = _read("invalid_exec.py").encode("utf-8")
    files = {"file": ("invalid_exec.py", io.BytesIO(src), "text/x-python")}
    r = client.post("/v1/workflows/import-python", files=files, headers=auth_headers)
    assert r.status_code == 422
    body = r.json()
    assert body["error"]["code"] == "PYTHON_IMPORT_REJECTED"


def test_rejected_import_inserts_no_row(client, auth_headers):
    """a rejected file must not insert a row into `workflows`."""
    pre = client.get("/v1/workflows", headers=auth_headers).json()
    pre_count = len(pre["items"])

    src = _read("invalid_dict_dsl.py").encode("utf-8")
    files = {"file": ("invalid_dict_dsl.py", io.BytesIO(src), "text/x-python")}
    r = client.post("/v1/workflows/import-python", files=files, headers=auth_headers)
    assert r.status_code == 422

    post = client.get("/v1/workflows", headers=auth_headers).json()
    assert len(post["items"]) == pre_count


def test_endpoint_includes_line_in_rejection_details(client, auth_headers):
    """rejection responses include details = {reason, line}."""
    src = _read("invalid_dict_dsl.py").encode("utf-8")
    files = {"file": ("invalid_dict_dsl.py", io.BytesIO(src), "text/x-python")}
    r = client.post("/v1/workflows/import-python", files=files, headers=auth_headers)
    assert r.status_code == 422
    details = r.json()["error"]["details"]
    assert "reason" in details
    assert "line" in details
