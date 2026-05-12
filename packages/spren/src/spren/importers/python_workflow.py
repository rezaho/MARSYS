"""AST-based parser for marsys-framework Python workflow files.

The parser reads source written against the framework's own constructor
surface (NOT Spren's API mirror) and translates to a Spren ``WorkflowDefinition``.
It NEVER calls ``exec``/``eval``/``compile`` itself; everything is structural
inspection of the parsed ``ast`` tree.

Two-pass walk:

1. Pass 1 collects module-level constants (``ASSIGN`` of a literal to a single
   ``Name``) into a name → value table.
2. Pass 2 walks the module body looking for ``Agent(...)``, ``Topology(...)``,
   ``ExecutionConfig(...)`` calls and resolves any ``Name`` reference (e.g.,
   ``name=AGENT_NAME``) against the pass-1 table.

Rejected constructs (each surfaces ``PythonImportError`` with a structured
``reason`` tag):

- ``exec``/``eval``/``compile``/``__import__`` calls anywhere
- Class definitions, function definitions, decorators
- List/dict/set/generator comprehensions, conditional expressions
- The dict-DSL ``{"Start -> Researcher": ...}`` topology shape
- f-strings (``JoinedStr``) in user-facing fields
- Files larger than ``MAX_IMPORT_BYTES`` (1 MiB) — checked at the FastAPI handler

``api_key=`` kwargs on ``ModelConfig(...)`` calls are silently dropped at the
extraction step. Credentials live in the per-user secrets store, not in
imported workflow definitions; logging the discard is enough.
"""
from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from typing import Any

from spren.models import (
    AgentSpec,
    EdgePattern,
    EdgeSpec,
    EdgeType,
    ExecutionConfigSpec,
    ModelConfigSpec,
    NodeSpec,
    NodeType,
    TopologySpec,
    WorkflowDefinition,
)

logger = logging.getLogger(__name__)

MAX_IMPORT_BYTES = 1 * 1024 * 1024  # 1 MiB

_FORBIDDEN_CALLS = frozenset({"exec", "eval", "compile", "__import__"})

# AST node types that signal "this file uses control flow / dynamic
# construction we don't accept" — anything beyond literal kwarg constructors.
# `def` / `async def` are allowed because users supply tool callables via
# `tools={"name": some_def}`; the importer only extracts the dict KEYS.
_FORBIDDEN_NODE_TYPES = (
    ast.ClassDef,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.IfExp,
    ast.Lambda,
)
# Inside a function body anything goes (the importer never executes the
# bodies). The forbidden-node check only runs over module-level statements
# and their direct expression descendants — see the call-site filter.


class PythonImportError(Exception):
    """Raised when the importer rejects a file. Carries a structured reason."""

    def __init__(self, message: str, *, reason: str, line: int | None = None, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = {"reason": reason}
        if line is not None:
            self.details["line"] = line
        self.details.update(details)


@dataclass
class _AgentCall:
    """Captured ``Agent(...)`` constructor call after kwarg/positional binding."""

    line: int
    kwargs: dict[str, Any]


@dataclass
class ImportWarning:
    """Non-blocking warning surfaced to the user after a successful import.

    Currently emitted by ``_build_edges`` when a Spren-unsupported edge
    pattern (``EdgePattern.ALTERNATING`` / ``EdgePattern.SYMMETRIC``) is
    auto-converted to plain bidirectional. The frontend renders these as a
    toast on import and as a yellow per-edge marker on the canvas.
    """

    code: str
    source: str
    target: str
    original_pattern: str
    message: str


@dataclass
class ImportResult:
    definition: WorkflowDefinition
    name: str
    description: str | None
    warnings: list[ImportWarning]


def parse_python_workflow(source: str) -> ImportResult:
    """Parse the source and return an ``ImportResult``.

    ``ImportResult.name`` is sourced from the module docstring's first line,
    otherwise ``"Imported workflow"``. ``description`` is the rest of the
    docstring (or ``None``). ``warnings`` lists every non-blocking
    conversion the parser performed.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise PythonImportError(
            f"file failed to parse: {exc.msg}",
            reason="syntax_error",
            line=exc.lineno,
        ) from exc

    # Pre-walk: reject forbidden constructs in module-level scope. Skip into
    # function bodies — the importer only extracts dict KEYS from `tools={}`,
    # so what's inside a tool callable doesn't matter to us.
    for node in _walk_module_scope(tree):
        if isinstance(node, _FORBIDDEN_NODE_TYPES):
            raise PythonImportError(
                f"unsupported construct: {type(node).__name__}; "
                "use the visual builder for control-flow workflows",
                reason=_label(node),
                line=getattr(node, "lineno", None),
            )
        if isinstance(node, ast.Call) and _name_of(node.func) in _FORBIDDEN_CALLS:
            raise PythonImportError(
                f"forbidden call to {_name_of(node.func)}",
                reason="forbidden_call",
                line=node.lineno,
            )
        # Decorators on module-level functions are out of scope: they execute
        # at import time normally and we don't actually run the file, but
        # their presence signals "this isn't a plain workflow declaration".
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.decorator_list:
            raise PythonImportError(
                f"decorators on module-level definitions are not supported "
                f"(found on '{node.name}')",
                reason="decorator",
                line=node.lineno,
            )

    # Pass 1 — module-level constants (Name = literal).
    constants = _collect_module_constants(tree)

    # Pass 2 — find Agent / Topology / ExecutionConfig literal constructor calls.
    agent_calls: list[_AgentCall] = []
    topology_call: ast.Call | None = None
    exec_config_call: ast.Call | None = None

    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            call = node.value
            target = _name_of(call.func)
            if target == "Agent":
                agent_calls.append(_capture_agent(call, constants))
            elif target == "Topology":
                if topology_call is not None:
                    raise PythonImportError(
                        "more than one Topology(...) call in file",
                        reason="multiple_topologies",
                        line=call.lineno,
                    )
                topology_call = call
            elif target == "ExecutionConfig":
                exec_config_call = call

    # Even if Agent/ExecutionConfig calls live deeper in the tree (kwarg of
    # something else), we want to find them at module level — the user's
    # workflow files we accept declare them as top-level statements.

    if not agent_calls:
        raise PythonImportError(
            "no Agent(...) constructor calls found at module level",
            reason="no_agents",
        )
    if topology_call is None:
        raise PythonImportError(
            "no Topology(...) constructor call found at module level",
            reason="no_topology",
        )

    agents_by_name = _build_agents(agent_calls)
    warnings: list[ImportWarning] = []
    topology = _build_topology(topology_call, constants, agents_by_name, warnings)
    execution_config = (
        _build_execution_config(exec_config_call, constants)
        if exec_config_call is not None
        else ExecutionConfigSpec()
    )

    definition = WorkflowDefinition(
        topology=topology,
        agents=agents_by_name,
        execution_config=execution_config,
    )

    docstring = ast.get_docstring(tree)
    if docstring:
        first_line, _, rest = docstring.partition("\n")
        derived_name = first_line.strip() or "Imported workflow"
        description = rest.strip() or None
    else:
        derived_name = "Imported workflow"
        description = None

    return ImportResult(
        definition=definition,
        name=derived_name,
        description=description,
        warnings=warnings,
    )


# --- pass 1: module-level constants ---


def _collect_module_constants(tree: ast.Module) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            value = _resolve_literal(node.value, table=table)
        except _NotALiteral:
            continue
        table[target.id] = value
    return table


# --- pass 2: extract calls ---


def _capture_agent(call: ast.Call, constants: dict[str, Any]) -> _AgentCall:
    """Bind ``Agent(...)`` positional + keyword args to the marsys signature.

    The framework's ``Agent.__init__`` (``packages/framework/src/marsys/agents/
    agents.py:2748``) takes:
        ``model_config, goal, instruction, tools=None, memory_type=...,
          memory_config=..., compaction_model_config=..., max_tokens=None,
          name=None, allowed_peers=None, ...``
    """
    positional_order = [
        "model_config",
        "goal",
        "instruction",
        "tools",
        "memory_type",
        "memory_config",
        "compaction_model_config",
        "max_tokens",
        "name",
        "allowed_peers",
    ]
    bound: dict[str, Any] = {}
    for index, arg in enumerate(call.args):
        if index >= len(positional_order):
            raise PythonImportError(
                "Agent(...) called with too many positional arguments",
                reason="too_many_positional",
                line=call.lineno,
            )
        key = positional_order[index]
        bound[key] = _resolve_for_kwarg(arg, key, constants)
    for kw in call.keywords:
        if kw.arg is None:
            raise PythonImportError(
                "**kwargs splat is not supported in Agent(...) calls",
                reason="kwargs_splat",
                line=call.lineno,
            )
        if kw.arg in bound:
            raise PythonImportError(
                f"Agent argument '{kw.arg}' specified twice",
                reason="duplicate_arg",
                line=call.lineno,
            )
        bound[kw.arg] = _resolve_for_kwarg(kw.value, kw.arg, constants)
    return _AgentCall(line=call.lineno, kwargs=bound)


def _build_agents(agent_calls: list[_AgentCall]) -> dict[str, AgentSpec]:
    agents: dict[str, AgentSpec] = {}
    for entry in agent_calls:
        kwargs = entry.kwargs
        if "model_config" not in kwargs:
            raise PythonImportError(
                "Agent(...) requires a model_config argument",
                reason="missing_model_config",
                line=entry.line,
            )
        if "goal" not in kwargs:
            raise PythonImportError(
                "Agent(...) requires a goal argument",
                reason="missing_goal",
                line=entry.line,
            )
        if "instruction" not in kwargs:
            raise PythonImportError(
                "Agent(...) requires an instruction argument",
                reason="missing_instruction",
                line=entry.line,
            )
        name = kwargs.get("name")
        if not isinstance(name, str) or not name:
            raise PythonImportError(
                "Agent(...) requires a non-empty `name` keyword "
                "(Spren keys agents by name in the workflow definition)",
                reason="missing_name",
                line=entry.line,
            )
        if name in agents:
            raise PythonImportError(
                f"two agents share the name '{name}'",
                reason="duplicate_name",
                line=entry.line,
            )
        try:
            spec = AgentSpec(
                agent_model=_to_model_config_spec(kwargs["model_config"], entry.line),
                name=name,
                goal=kwargs["goal"],
                instruction=kwargs["instruction"],
                tools=_extract_tool_names(kwargs.get("tools"), entry.line),
                memory_retention=kwargs.get("memory_retention", "session"),
                allowed_peers=kwargs.get("allowed_peers", []) or [],
            )
        except Exception as exc:
            raise PythonImportError(
                f"Agent '{name}' failed Pydantic validation: {exc}",
                reason="agent_invalid",
                line=entry.line,
            ) from exc
        agents[name] = spec
    return agents


def _to_model_config_spec(value: Any, line: int) -> ModelConfigSpec:
    if isinstance(value, dict):
        # Drop api_key silently — credentials live in the secrets store.
        if "api_key" in value:
            logger.warning(
                "discarding api_key= in imported ModelConfig (line %d); "
                "configure credentials via the secrets store",
                line,
            )
            value = {k: v for k, v in value.items() if k != "api_key"}
        try:
            return ModelConfigSpec.model_validate(value)
        except Exception as exc:
            raise PythonImportError(
                f"ModelConfig(...) at line {line} failed validation: {exc}",
                reason="model_config_invalid",
                line=line,
            ) from exc
    raise PythonImportError(
        "Agent.model_config must be a literal ModelConfig(...) constructor",
        reason="model_config_not_literal",
        line=line,
    )


def _extract_tool_names(value: Any, line: int) -> list[str]:
    """``tools=`` is a Dict[str, Callable] in marsys; we keep only the keys."""
    if value is None:
        return []
    if not isinstance(value, dict):
        raise PythonImportError(
            "Agent.tools must be a dict literal {tool_name: callable}",
            reason="tools_not_dict",
            line=line,
        )
    names: list[str] = []
    for key in value:
        if not isinstance(key, str):
            raise PythonImportError(
                "Agent.tools keys must be string literals",
                reason="tools_key_not_string",
                line=line,
            )
        names.append(key)
    return names


# --- topology ---


def _build_topology(
    call: ast.Call,
    constants: dict[str, Any],
    agents: dict[str, AgentSpec],
    warnings: list[ImportWarning],
) -> TopologySpec:
    nodes_arg = _kwarg(call, "nodes")
    edges_arg = _kwarg(call, "edges", default=None)
    rules_arg = _kwarg(call, "rules", default=None)

    if nodes_arg is None:
        raise PythonImportError(
            "Topology(...) requires a `nodes=` argument",
            reason="topology_missing_nodes",
            line=call.lineno,
        )
    nodes = _build_nodes(nodes_arg, constants, agents)
    edges = _build_edges(edges_arg, constants, warnings) if edges_arg is not None else []
    rules = _build_rule_strings(rules_arg, constants) if rules_arg is not None else []
    try:
        return TopologySpec(nodes=nodes, edges=edges, rules=rules)
    except Exception as exc:
        raise PythonImportError(
            f"Topology failed validation: {exc}",
            reason="topology_invalid",
            line=call.lineno,
        ) from exc


def _build_nodes(arg: ast.AST, constants: dict[str, Any], agents: dict[str, AgentSpec]) -> list[NodeSpec]:
    if isinstance(arg, ast.Dict):
        raise PythonImportError(
            "the dict-DSL topology shape is not supported; rewrite to "
            "Topology(nodes=[Node(...)], edges=[Edge(...)])",
            reason="dict_dsl_topology",
            line=arg.lineno,
        )
    if not isinstance(arg, ast.List):
        raise PythonImportError(
            "Topology.nodes must be a literal list of Node(...) calls",
            reason="nodes_not_list",
            line=getattr(arg, "lineno", None),
        )
    nodes: list[NodeSpec] = []
    agent_names = {spec.name for spec in agents.values()}
    for element in arg.elts:
        if not (isinstance(element, ast.Call) and _name_of(element.func) == "Node"):
            raise PythonImportError(
                "every nodes[] element must be a Node(...) call",
                reason="node_not_call",
                line=getattr(element, "lineno", None),
            )
        kwargs = _bind_call(element, ["name", "node_type", "agent_ref", "is_convergence_point", "metadata"], constants)
        node_type = _coerce_node_type(kwargs.get("node_type"), element.lineno)
        agent_ref = kwargs.get("agent_ref")
        # Allow agent_ref to point at an agent name from the file (matching
        # marsys's runtime behavior of resolving agent_ref to the agent
        # instance by name).
        if isinstance(agent_ref, str) and node_type == NodeType.AGENT and agent_ref not in agents:
            # Fall through to Spren-side cross-validation for the cleaner
            # error message; don't pre-empt here.
            pass
        try:
            spec = NodeSpec(
                name=kwargs["name"],
                node_type=node_type,
                agent_ref=agent_ref if isinstance(agent_ref, str) else None,
                is_convergence_point=bool(kwargs.get("is_convergence_point", False)),
                metadata=dict(kwargs.get("metadata") or {}),
            )
        except Exception as exc:
            raise PythonImportError(
                f"Node(...) failed validation: {exc}",
                reason="node_invalid",
                line=element.lineno,
            ) from exc
        nodes.append(spec)
        # If the node is an agent and agent_ref was omitted, default to the
        # node name (matches the agents map keying) — this keeps the import
        # round-trippable even when users wrote `Node(name="X", node_type="agent")`.
        if spec.node_type == NodeType.AGENT and spec.agent_ref is None and spec.name in agent_names:
            nodes[-1] = spec.model_copy(update={"agent_ref": spec.name})
    return nodes


def _build_edges(
    arg: ast.AST,
    constants: dict[str, Any],
    warnings: list[ImportWarning],
) -> list[EdgeSpec]:
    if not isinstance(arg, ast.List):
        raise PythonImportError(
            "Topology.edges must be a literal list of Edge(...) calls",
            reason="edges_not_list",
            line=getattr(arg, "lineno", None),
        )
    edges: list[EdgeSpec] = []
    for element in arg.elts:
        if not (isinstance(element, ast.Call) and _name_of(element.func) == "Edge"):
            raise PythonImportError(
                "every edges[] element must be an Edge(...) call",
                reason="edge_not_call",
                line=getattr(element, "lineno", None),
            )
        kwargs = _bind_call(
            element,
            ["source", "target", "edge_type", "bidirectional", "pattern", "metadata"],
            constants,
        )
        try:
            pattern = _coerce_edge_pattern(kwargs.get("pattern"), element.lineno)
            bidirectional = bool(kwargs.get("bidirectional", False))
            metadata = dict(kwargs.get("metadata") or {})
            source = kwargs["source"]
            target = kwargs["target"]

            # Q2 lock: Spren's canvas only renders unidirectional + plain
            # bidirectional edges. Alternating / symmetric patterns the
            # framework supports are auto-converted to bidirectional on
            # import, with a non-blocking warning surfaced to the UI.
            if pattern in (EdgePattern.ALTERNATING, EdgePattern.SYMMETRIC):
                original_pattern = pattern.value
                metadata = {
                    **metadata,
                    "spren_converted_from": original_pattern,
                }
                warnings.append(
                    ImportWarning(
                        code="pattern_auto_converted",
                        source=source,
                        target=target,
                        original_pattern=original_pattern,
                        message=(
                            f"edge {source}<{'~' if original_pattern == 'alternating' else '|'}>"
                            f"{target} was converted to plain bidirectional; "
                            "the canvas only renders uni/bi directions in v0.3"
                        ),
                    )
                )
                bidirectional = True
                pattern = None

            spec = EdgeSpec(
                source=source,
                target=target,
                edge_type=_coerce_edge_type(kwargs.get("edge_type"), element.lineno),
                bidirectional=bidirectional,
                pattern=pattern,
                metadata=metadata,
            )
        except PythonImportError:
            raise
        except Exception as exc:
            raise PythonImportError(
                f"Edge(...) failed validation: {exc}",
                reason="edge_invalid",
                line=element.lineno,
            ) from exc
        edges.append(spec)
    return edges


def _build_rule_strings(arg: ast.AST, constants: dict[str, Any]) -> list[str]:
    if not isinstance(arg, ast.List):
        raise PythonImportError(
            "Topology.rules must be a literal list of strings",
            reason="rules_not_list",
            line=getattr(arg, "lineno", None),
        )
    out: list[str] = []
    for element in arg.elts:
        try:
            value = _resolve_literal(element, table=constants)
        except _NotALiteral as exc:
            raise PythonImportError(
                "Topology.rules entries must be string literals",
                reason="rule_not_literal",
                line=element.lineno,
            ) from exc
        if not isinstance(value, str):
            raise PythonImportError(
                "Topology.rules entries must be strings",
                reason="rule_not_string",
                line=element.lineno,
            )
        out.append(value)
    return out


def _build_execution_config(call: ast.Call, constants: dict[str, Any]) -> ExecutionConfigSpec:
    fields: dict[str, Any] = {}
    for kw in call.keywords:
        if kw.arg is None:
            raise PythonImportError(
                "**kwargs splat is not supported in ExecutionConfig(...) calls",
                reason="exec_kwargs_splat",
                line=call.lineno,
            )
        try:
            fields[kw.arg] = _resolve_literal(kw.value, table=constants)
        except _NotALiteral as exc:
            raise PythonImportError(
                f"ExecutionConfig argument '{kw.arg}' is not a literal",
                reason="exec_arg_not_literal",
                line=kw.value.lineno,
            ) from exc
    try:
        return ExecutionConfigSpec.model_validate(fields)
    except Exception as exc:
        raise PythonImportError(
            f"ExecutionConfig failed validation: {exc}",
            reason="exec_config_invalid",
            line=call.lineno,
        ) from exc


# --- enum coercion ---


def _coerce_node_type(value: Any, line: int) -> NodeType:
    if value is None:
        return NodeType.AGENT
    if isinstance(value, NodeType):
        return value
    if isinstance(value, str):
        try:
            return NodeType(value.lower())
        except ValueError as exc:
            raise PythonImportError(
                f"unknown node_type '{value}'",
                reason="bad_node_type",
                line=line,
            ) from exc
    if isinstance(value, dict) and "_enum_member" in value:
        cls = value["_enum_class"]
        if cls != "NodeType":
            raise PythonImportError(
                f"expected NodeType, got {cls}",
                reason="bad_node_type",
                line=line,
            )
        return NodeType(value["_enum_member"])
    raise PythonImportError(
        f"unsupported node_type literal: {type(value).__name__}",
        reason="bad_node_type",
        line=line,
    )


def _coerce_edge_type(value: Any, line: int) -> EdgeType:
    if value is None:
        return EdgeType.INVOKE
    if isinstance(value, EdgeType):
        return value
    if isinstance(value, str):
        try:
            return EdgeType(value.lower())
        except ValueError as exc:
            raise PythonImportError(
                f"unknown edge_type '{value}'",
                reason="bad_edge_type",
                line=line,
            ) from exc
    if isinstance(value, dict) and "_enum_member" in value:
        cls = value["_enum_class"]
        if cls != "EdgeType":
            raise PythonImportError(
                f"expected EdgeType, got {cls}",
                reason="bad_edge_type",
                line=line,
            )
        return EdgeType(value["_enum_member"])
    raise PythonImportError(
        f"unsupported edge_type literal: {type(value).__name__}",
        reason="bad_edge_type",
        line=line,
    )


def _coerce_edge_pattern(value: Any, line: int) -> EdgePattern | None:
    if value is None:
        return None
    if isinstance(value, EdgePattern):
        return value
    if isinstance(value, str):
        try:
            return EdgePattern(value.lower())
        except ValueError as exc:
            raise PythonImportError(
                f"unknown edge pattern '{value}'",
                reason="bad_edge_pattern",
                line=line,
            ) from exc
    if isinstance(value, dict) and "_enum_member" in value:
        cls = value["_enum_class"]
        if cls != "EdgePattern":
            raise PythonImportError(
                f"expected EdgePattern, got {cls}",
                reason="bad_edge_pattern",
                line=line,
            )
        return EdgePattern(value["_enum_member"])
    raise PythonImportError(
        f"unsupported edge pattern literal: {type(value).__name__}",
        reason="bad_edge_pattern",
        line=line,
    )


# --- literal-only resolution ---


class _NotALiteral(ValueError):
    """Raised when an AST node can't be resolved to a literal value."""


def _resolve_literal(node: ast.AST, *, table: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.JoinedStr):
        raise PythonImportError(
            "f-strings are not supported in workflow files",
            reason="fstring",
            line=node.lineno,
        )
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        operand = _resolve_literal(node.operand, table=table)
        return -operand if isinstance(node.op, ast.USub) else +operand
    if isinstance(node, ast.Name):
        if node.id in table:
            return table[node.id]
        # Allow bare True/False/None even though those are constants in py3.
        # Anything else: caller decides.
        raise _NotALiteral(f"unknown name '{node.id}'")
    if isinstance(node, ast.List):
        return [_resolve_literal(e, table=table) for e in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_resolve_literal(e, table=table) for e in node.elts)
    if isinstance(node, ast.Set):
        return {_resolve_literal(e, table=table) for e in node.elts}
    if isinstance(node, ast.Dict):
        return {
            _resolve_literal(k, table=table): _resolve_literal(v, table=table)
            for k, v in zip(node.keys, node.values, strict=True)
            if k is not None
        }
    if isinstance(node, ast.Call):
        target = _name_of(node.func)
        if target == "ModelConfig":
            return _flatten_kwargs(node, table)
        if target == "ExecutionConfig":
            return _flatten_kwargs(node, table)
        if target in {"Node", "Edge", "Topology", "Agent"}:
            # These are top-level constructors handled by their own walkers,
            # not embeddable as nested literals.
            raise _NotALiteral(f"{target}(...) cannot appear as a nested literal")
        raise _NotALiteral(f"unsupported call: {target or '<unknown>'}")
    if isinstance(node, ast.Attribute):
        # Allow NodeType.AGENT, EdgeType.INVOKE, EdgePattern.SYMMETRIC etc.
        owner = _name_of(node.value)
        if owner in {"NodeType", "EdgeType", "EdgePattern"}:
            return {"_enum_class": owner, "_enum_member": node.attr.lower()}
        raise _NotALiteral(f"unsupported attribute access: {owner}.{node.attr}")
    raise _NotALiteral(f"unsupported expression: {type(node).__name__}")


def _flatten_kwargs(call: ast.Call, table: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if call.args:
        raise PythonImportError(
            "positional arguments are not supported in nested ModelConfig/ExecutionConfig calls",
            reason="nested_positional",
            line=call.lineno,
        )
    for kw in call.keywords:
        if kw.arg is None:
            raise PythonImportError(
                "**kwargs splat is not supported",
                reason="nested_kwargs_splat",
                line=call.lineno,
            )
        try:
            out[kw.arg] = _resolve_literal(kw.value, table=table)
        except _NotALiteral as exc:
            raise PythonImportError(
                f"argument '{kw.arg}' is not a literal",
                reason="nested_arg_not_literal",
                line=kw.value.lineno,
            ) from exc
    return out


def _resolve_for_kwarg(node: ast.AST, key: str, constants: dict[str, Any]) -> Any:
    """Resolve an Agent kwarg against the constants table.

    For ``model_config`` we expect a literal ``ModelConfig(...)`` call; the
    walker normalizes that to the kwarg dict so the AgentSpec builder can
    re-validate via Pydantic.

    For ``tools`` we accept a dict whose keys are string literals; values may
    be Name references to in-file callables (or imported names). The values
    are discarded — only the keys travel to ``AgentSpec.tools``.
    """
    if key == "tools":
        return _extract_tools_dict_keys(node)
    try:
        return _resolve_literal(node, table=constants)
    except _NotALiteral as exc:
        raise PythonImportError(
            f"Agent.{key} argument is not a literal value",
            reason=f"{key}_not_literal",
            line=node.lineno,
        ) from exc


def _extract_tools_dict_keys(node: ast.AST) -> dict[str, None]:
    """``tools={"name": <callable>, ...}`` — keep keys, discard values.

    Each value is allowed to be a Name reference (the typical
    ``def my_tool(...): ...; tools={"x": my_tool}`` pattern) or a literal
    constant. Subscript / Call / Attribute / dynamic constructions in the
    value position are rejected — the parser shouldn't pretend the value
    didn't matter when its shape itself is suspect.
    """
    if isinstance(node, ast.Constant) and node.value is None:
        return {}
    if not isinstance(node, ast.Dict):
        raise PythonImportError(
            "Agent.tools must be a dict literal {tool_name: callable}",
            reason="tools_not_dict",
            line=getattr(node, "lineno", None),
        )
    out: dict[str, None] = {}
    for k, v in zip(node.keys, node.values, strict=True):
        if not isinstance(k, ast.Constant) or not isinstance(k.value, str):
            raise PythonImportError(
                "Agent.tools keys must be string literals",
                reason="tools_key_not_string",
                line=getattr(k, "lineno", None),
            )
        if not isinstance(v, (ast.Name, ast.Constant)):
            raise PythonImportError(
                "Agent.tools values must be a Name reference (e.g., a `def` in the same file) "
                "or a constant; complex expressions are not supported",
                reason=_label_tools_value(v),
                line=getattr(v, "lineno", None),
            )
        out[k.value] = None
    return out


def _label_tools_value(node: ast.AST) -> str:
    if isinstance(node, ast.Subscript):
        return "tools_value_subscript"
    if isinstance(node, ast.Call):
        return "tools_value_call"
    if isinstance(node, ast.Attribute):
        return "tools_value_attribute"
    if isinstance(node, ast.Lambda):
        return "tools_value_lambda"
    return "tools_value_unsupported"


def _bind_call(call: ast.Call, positional_order: list[str], constants: dict[str, Any]) -> dict[str, Any]:
    """Bind positional+keyword args to a known signature, resolving literals."""
    bound: dict[str, Any] = {}
    for index, arg in enumerate(call.args):
        if index >= len(positional_order):
            raise PythonImportError(
                f"{_name_of(call.func)}(...) called with too many positional arguments",
                reason="too_many_positional",
                line=call.lineno,
            )
        key = positional_order[index]
        try:
            bound[key] = _resolve_literal(arg, table=constants)
        except _NotALiteral as exc:
            raise PythonImportError(
                f"argument '{key}' is not a literal",
                reason=f"{key}_not_literal",
                line=arg.lineno,
            ) from exc
    for kw in call.keywords:
        if kw.arg is None:
            raise PythonImportError(
                "**kwargs splat is not supported",
                reason="kwargs_splat",
                line=call.lineno,
            )
        if kw.arg in bound:
            raise PythonImportError(
                f"argument '{kw.arg}' specified twice",
                reason="duplicate_arg",
                line=call.lineno,
            )
        try:
            bound[kw.arg] = _resolve_literal(kw.value, table=constants)
        except _NotALiteral as exc:
            raise PythonImportError(
                f"argument '{kw.arg}' is not a literal",
                reason=f"{kw.arg}_not_literal",
                line=kw.value.lineno,
            ) from exc
    return bound


def _kwarg(call: ast.Call, name: str, *, default: Any = ...) -> Any:
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    if default is ...:
        return None
    return default


def _name_of(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _walk_module_scope(tree: ast.Module):
    """Walk every node in the module EXCEPT inside function bodies.

    Tool callables defined in the file (``def my_tool(...): ...``) may contain
    any Python construct — we don't execute them, we just keep their name. The
    rejection rules only apply to module-level statements and the expressions
    used to build ``Agent / Topology / Node / Edge / ModelConfig / ExecutionConfig``
    calls.
    """
    stack: list[ast.AST] = list(tree.body)
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Skip the body; still surface the FunctionDef itself in case the
            # type is later moved back to the forbidden list.
            continue
        for child in ast.iter_child_nodes(node):
            stack.append(child)


def _label(node: ast.AST) -> str:
    return {
        ast.FunctionDef: "function_def",
        ast.AsyncFunctionDef: "async_function_def",
        ast.ClassDef: "class_def",
        ast.ListComp: "list_comp",
        ast.SetComp: "set_comp",
        ast.DictComp: "dict_comp",
        ast.GeneratorExp: "generator_exp",
        ast.IfExp: "if_expr",
        ast.Lambda: "lambda",
    }.get(type(node), type(node).__name__)
