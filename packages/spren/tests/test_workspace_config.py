"""Workspace + tooling config audits.

These checks live with the spren package because the spren tests are the
only suite that runs against the workspace root in CI today; they audit
files that don't have a natural test home.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_python_ulid_pin_in_spren_pyproject():
    """spren's pyproject pins ``python-ulid >= 3.0,<4.0`` (NOT `ulid-py`)."""
    pyproject = (REPO_ROOT / "packages" / "spren" / "pyproject.toml").read_text()
    assert re.search(r'python-ulid\s*>=\s*3\.0', pyproject), "python-ulid must be pinned"
    assert "ulid-py" not in pyproject, "ulid-py is the wrong package; use python-ulid"


def test_apps_web_does_not_install_json_schema_to_typescript():
    """json-schema-to-typescript dropped — single emitter approach."""
    package_json = json.loads((REPO_ROOT / "apps" / "web" / "package.json").read_text())
    deps = {**package_json.get("dependencies", {}), **package_json.get("devDependencies", {})}
    assert "json-schema-to-typescript" not in deps


def test_apps_web_installs_openapi_typescript():
    """openapi-typescript is the only emitter; declared as devDep."""
    package_json = json.loads((REPO_ROOT / "apps" / "web" / "package.json").read_text())
    dev = package_json.get("devDependencies", {})
    assert "openapi-typescript" in dev


def test_apps_web_prebuild_runs_type_generator():
    """package.json `prebuild` calls scripts/generate-types.mjs (no `predev`)."""
    package_json = json.loads((REPO_ROOT / "apps" / "web" / "package.json").read_text())
    scripts = package_json.get("scripts", {})
    assert "scripts/generate-types.mjs" in scripts.get("prebuild", "")
    # NOT predev — type-gen would block every dev server start
    assert "predev" not in scripts


def test_apps_web_does_not_have_inline_pnpm_block():
    """pnpm.peerDependencyRules belongs in pnpm-workspace.yaml, NOT package.json."""
    package_json = json.loads((REPO_ROOT / "apps" / "web" / "package.json").read_text())
    assert "pnpm" not in package_json


def test_workspace_peer_dep_rules_for_openapi_typescript():
    """workspace-level peerDependencyRules.allowedVersions allows openapi-typescript ^7 to coexist with TS 6."""
    workspace = (REPO_ROOT / "pnpm-workspace.yaml").read_text()
    assert "peerDependencyRules" in workspace
    assert "openapi-typescript>typescript" in workspace


def test_gitignore_includes_openapi_snapshot():
    """openapi-snapshot.json is gitignored — only the .ts is tracked."""
    gitignore = (REPO_ROOT / ".gitignore").read_text()
    assert "apps/web/openapi-snapshot.json" in gitignore


def test_generate_types_script_exists_as_mjs():
    """type-gen script lives at apps/web/scripts/generate-types.mjs (Node ESM, no tsx)."""
    script = REPO_ROOT / "apps" / "web" / "scripts" / "generate-types.mjs"
    assert script.exists(), f"missing: {script}"
    # Sanity: not a sneaky .ts twin
    twin = REPO_ROOT / "apps" / "web" / "scripts" / "generate-types.ts"
    assert not twin.exists(), "scripts/generate-types.ts must not exist (we use .mjs)"


def test_apps_web_lib_does_not_redeclare_pydantic_mirrors():
    """no apps/web/src/lib/*.ts other than api-types.generated declares
    interfaces/types named after Pydantic models."""
    lib_dir = REPO_ROOT / "apps" / "web" / "src" / "lib"
    forbidden_decls = re.compile(
        r"(interface|type)\s+(Workflow|WorkflowDefinition|WorkflowCreateRequest|"
        r"WorkflowUpdateRequest|WorkflowListResponse|TopologySpec|NodeSpec|EdgeSpec|"
        r"AgentSpec|ModelConfigSpec|ExecutionConfigSpec)\b"
    )
    for ts_file in lib_dir.iterdir():
        if not ts_file.is_file() or ts_file.name == "api-types.generated.ts":
            continue
        content = ts_file.read_text()
        # `export type X = ...` is OK if the RHS is `components["schemas"]["X"]` —
        # i.e., a re-export of the generated type. Forbid stand-alone `type X = {`
        # or `interface X {` in non-generated files.
        for m in forbidden_decls.finditer(content):
            kind, name = m.group(1), m.group(2)
            line_start = content.rfind("\n", 0, m.start()) + 1
            line_end = content.find("\n", m.end())
            if line_end == -1:
                line_end = len(content)
            line = content[line_start:line_end]
            if kind == "type" and 'components["schemas"]' in line:
                continue  # re-export of generated type — fine
            pytest.fail(f"{ts_file.name} declares hand-rolled `{kind} {name}`: {line.strip()}")


def test_no_stale_datamodel_codegen_comments():
    """stale `datamodel-code-generator` comments removed."""
    for path in [
        REPO_ROOT / "apps" / "web" / "src" / "lib" / "api.ts",
    ]:
        if path.exists():
            content = path.read_text()
            assert "datamodel-code-generator" not in content, f"{path}: stale comment"


def test_types_api_placeholder_removed():
    """apps/web/src/types/api.ts had a placeholder note; the file should
    either be deleted or no longer carry the stale Session-02 promise."""
    path = REPO_ROOT / "apps" / "web" / "src" / "types" / "api.ts"
    if path.exists():
        content = path.read_text()
        assert "Placeholder" not in content
        assert "datamodel-code-generator" not in content


# ---- SP-rules audits ----


def test_no_mocks_in_product_code():
    """grep returns 0 hits for mock patterns in product source."""
    pattern = re.compile(r"\b(MagicMock|mock\.patch|vi\.mock|jest\.mock|monkeypatch)\b")
    for src_dir in [
        REPO_ROOT / "packages" / "spren" / "src",
        REPO_ROOT / "apps" / "web" / "src",
    ]:
        for f in src_dir.rglob("*.py"):
            content = f.read_text()
            assert not pattern.search(content), f"{f}: mock pattern in product code"
        for f in src_dir.rglob("*.ts"):
            content = f.read_text()
            assert not pattern.search(content), f"{f}: mock pattern in product code"
        for f in src_dir.rglob("*.tsx"):
            content = f.read_text()
            assert not pattern.search(content), f"{f}: mock pattern in product code"


def test_no_legacy_branches_in_product_code():
    """no `if version`, `# legacy`, `# TODO: remove` patterns in product."""
    forbidden = [
        re.compile(r"#\s*legacy\b", re.IGNORECASE),
        re.compile(r"#\s*TODO:\s*remove\b", re.IGNORECASE),
        re.compile(r"REMOVE-IN-V\d+", re.IGNORECASE),
        re.compile(r"if\s+legacy_format\b"),
    ]
    for src_dir in [REPO_ROOT / "packages" / "spren" / "src"]:
        for f in src_dir.rglob("*.py"):
            content = f.read_text()
            for pat in forbidden:
                assert not pat.search(content), f"{f}: {pat.pattern}"


def test_framework_does_not_import_spren():
    """SP-018: framework must not import Spren at runtime.

    Docstring references to Spren as one of several adapter consumers are
    fine — the rule is about CODE imports + runtime branches, not prose.
    """
    fw_src = REPO_ROOT / "packages" / "framework" / "src" / "marsys"
    import_pattern = re.compile(
        r"^\s*(from\s+spren(\.|\s+import)|import\s+spren\b)", re.MULTILINE
    )
    for f in fw_src.rglob("*.py"):
        content = f.read_text()
        match = import_pattern.search(content)
        if match:
            pytest.fail(f"{f}:{content[:match.start()].count(chr(10)) + 1}: imports spren — SP-018 violation")


def test_just_recipes_use_canonical_invocations():
    """Justfile uses `pnpm --filter @marsys/spren-web ...` and `uv run --package <pkg> ...`."""
    justfile = (REPO_ROOT / "Justfile").read_text()
    # No bare `pnpm --filter web` (the wrong scoped name)
    assert re.search(r"pnpm\s+--filter\s+web\b", justfile) is None
    # The scoped name appears
    assert "@marsys/spren-web" in justfile
    # uv invocations use --package
    assert "uv run --package spren" in justfile
    assert "uv run --package marsys" in justfile
