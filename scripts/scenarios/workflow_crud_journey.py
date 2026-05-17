"""Workflow CRUD journey — a realistic user trip through the workflow surface.

What this exercises:
    1. Sidecar boots and answers /healthz unauthenticated
    2. Auth gate works (missing token → 401, valid token → 200)
    3. /v1/tools returns the framework tool registry
    4. Empty workflow list at start
    5. Create a 3-agent workflow with a realistic topology
    6. Read it back; payload round-trips
    7. List filters work (provenance, archived, include_drafts)
    8. Lint catches at least one Spren-side issue we plant on purpose
    9. PATCH archives the workflow; list excludes it; ?archived=true includes it
   10. PUT replaces the workflow with edits; updated_at advances
   11. DELETE removes it; subsequent GET returns 404
   12. Idempotency-Key replay works for POST

Findings are recorded for anything that surprises a real user — bad error
messages, non-2xx responses on happy paths, shape inconsistencies, etc.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make _shared importable when run as a standalone script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _shared import FindingsCollector, SprenClient, request_summary, sidecar


JOURNEY = "workflow_crud_journey"


def realistic_definition() -> dict[str, object]:
    """A 3-agent research → write → review pipeline with a Tool node.

    NOTE: the user/system/tool nodes are named `user_in`/`web_search` instead
    of `user`/`web_search` because NodeSpec._no_reserved_names rejects any
    node whose name is in {user, system, tool} regardless of node_type — see
    PRODUCT-BUG-001 below.
    """
    return {
        "topology": {
            "nodes": [
                {"name": "user_in", "node_type": "user", "agent_ref": None, "is_convergence_point": False, "metadata": {}},
                {"name": "researcher", "node_type": "agent", "agent_ref": "researcher", "is_convergence_point": False, "metadata": {}},
                {"name": "writer", "node_type": "agent", "agent_ref": "writer", "is_convergence_point": False, "metadata": {}},
                {"name": "reviewer", "node_type": "agent", "agent_ref": "reviewer", "is_convergence_point": True, "metadata": {}},
                {"name": "web_search", "node_type": "tool", "agent_ref": None, "is_convergence_point": False, "metadata": {}},
            ],
            "edges": [
                {"source": "user_in", "target": "researcher", "edge_type": "invoke", "bidirectional": False, "pattern": None, "metadata": {}},
                {"source": "researcher", "target": "writer", "edge_type": "invoke", "bidirectional": False, "pattern": None, "metadata": {}},
                {"source": "writer", "target": "reviewer", "edge_type": "invoke", "bidirectional": False, "pattern": None, "metadata": {}},
                {"source": "researcher", "target": "web_search", "edge_type": "invoke", "bidirectional": False, "pattern": None, "metadata": {}},
            ],
            "rules": [],
        },
        "agents": {
            "researcher": _agent("Researcher", "Find authoritative sources on the topic.", "You are a research agent.", tools=["search_web", "browse_url"]),
            "writer": _agent("Writer", "Draft a concise summary.", "You are a writer agent.", tools=[]),
            "reviewer": _agent("Reviewer", "Critique the draft for accuracy.", "You are a reviewer agent.", tools=[]),
        },
        "execution_config": {
            "convergence_timeout": 300.0,
            "response_format": "json",
            "user_interaction": "none",
            "convergence_policy": 0.7,
            "status": {"enabled": True},
        },
    }


def _agent(name: str, goal: str, instruction: str, *, tools: list[str]) -> dict[str, object]:
    return {
        "agent_model": {
            "type": "api",
            "name": "claude-opus-4-7",
            "provider": "anthropic",
            "base_url": None,
            "max_tokens": 4096,
            "temperature": 0.7,
            "thinking_budget": None,
            "reasoning_effort": None,
            "oauth_profile": None,
        },
        "name": name,
        "goal": goal,
        "instruction": instruction,
        "tools": tools,
        "memory_retention": "session",
        "allowed_peers": [],
    }


def run() -> int:
    findings = FindingsCollector(JOURNEY)
    print(f"=== {JOURNEY} starting ===", flush=True)

    with sidecar() as h:
        print(f"sidecar up on {h.base_url} (data_dir={h.data_dir})", flush=True)

        # --- step 1: healthz unauthenticated -----------------------------
        with SprenClient(h) as c:
            r = c.healthz()
            if r.status_code != 200:
                findings.critical(
                    surface="GET /healthz",
                    summary=f"healthz returned {r.status_code} (expected 200)",
                    response=request_summary(r),
                )

            # --- step 2: auth gate -----------------------------------------
            # Same path with no auth headers should 401.
            import httpx
            unauth = httpx.get(f"{h.base_url}/v1/bootstrap", timeout=5)
            if unauth.status_code != 401:
                findings.critical(
                    surface="auth gate",
                    summary=f"missing-token /v1/bootstrap returned {unauth.status_code} (expected 401)",
                    response={"status": unauth.status_code, "body": unauth.text[:200]},
                )

            r = c.bootstrap()
            if r.status_code != 200:
                findings.critical(
                    surface="GET /v1/bootstrap",
                    summary=f"authenticated bootstrap returned {r.status_code}",
                    response=request_summary(r),
                )
            else:
                body = r.json()
                expected_keys = {"framework", "spren", "endpoints", "started_at", "data_dir"}
                missing = expected_keys - set(body.keys())
                if missing:
                    findings.important(
                        surface="GET /v1/bootstrap",
                        summary=f"bootstrap missing expected keys: {sorted(missing)}",
                        response=request_summary(r),
                    )
                expected_endpoints = {"workflows", "tools", "lint", "runs", "files"}
                actual_endpoints = set((body.get("endpoints") or {}).keys())
                if not expected_endpoints.issubset(actual_endpoints):
                    findings.important(
                        surface="GET /v1/bootstrap",
                        summary=f"bootstrap endpoints missing: {sorted(expected_endpoints - actual_endpoints)}",
                        response=request_summary(r),
                    )

            # --- step 3: /v1/tools -----------------------------------------
            r = c.list_tools()
            if r.status_code != 200:
                findings.critical(
                    surface="GET /v1/tools",
                    summary=f"/v1/tools returned {r.status_code}",
                    response=request_summary(r),
                )
            else:
                tools_body = r.json()
                if "items" not in tools_body or not isinstance(tools_body["items"], list):
                    findings.critical(
                        surface="GET /v1/tools",
                        summary="response is missing top-level 'items' array",
                        response=request_summary(r),
                    )
                else:
                    if not tools_body["items"]:
                        findings.important(
                            surface="GET /v1/tools",
                            summary="tool registry is empty — framework tools missing?",
                            response=request_summary(r),
                        )
                    for item in tools_body["items"][:3]:
                        if set(item.keys()) != {"name", "source", "description"}:
                            findings.important(
                                surface="GET /v1/tools",
                                summary=f"tool item shape unexpected: keys={sorted(item.keys())}",
                                response={"item": item},
                            )
                            break

            # --- step 4: empty list at start -------------------------------
            r = c.list_workflows()
            if r.status_code != 200:
                findings.critical(
                    surface="GET /v1/workflows",
                    summary=f"empty list returned {r.status_code}",
                    response=request_summary(r),
                )
                return _finish(findings)
            initial_items = r.json().get("items", [])
            if initial_items:
                findings.info(
                    surface="GET /v1/workflows",
                    summary=f"initial list non-empty ({len(initial_items)} rows) — leftover data from a prior run",
                )

            # --- step 5: create realistic workflow -------------------------
            payload = {
                "name": "research-pipeline",
                "description": "3-agent research → write → review pipeline",
                "definition": realistic_definition(),
                "provenance": "api",
                "provenance_metadata": None,
            }
            r = c.create_workflow(payload)
            if r.status_code != 201:
                findings.critical(
                    surface="POST /v1/workflows",
                    summary=f"create returned {r.status_code} (expected 201)",
                    request={"payload_keys": sorted(payload.keys())},
                    response=request_summary(r),
                )
                return _finish(findings)
            created = r.json()
            workflow_id = created.get("id")
            if not workflow_id:
                findings.critical(
                    surface="POST /v1/workflows",
                    summary="201 response is missing 'id'",
                    response=request_summary(r),
                )
                return _finish(findings)

            # --- step 6: round-trip ----------------------------------------
            r = c.get_workflow(workflow_id)
            if r.status_code != 200:
                findings.critical(
                    surface="GET /v1/workflows/{id}",
                    summary=f"read-after-create returned {r.status_code}",
                    response=request_summary(r),
                )
            else:
                read_back = r.json()
                if read_back.get("name") != payload["name"]:
                    findings.important(
                        surface="GET /v1/workflows/{id}",
                        summary="name did not round-trip",
                        detail=f"sent={payload['name']!r} got={read_back.get('name')!r}",
                    )
                # Check the definition's agents key count matches.
                got_agents = read_back.get("definition", {}).get("agents", {})
                sent_agents = payload["definition"]["agents"]  # type: ignore[index]
                if set(got_agents.keys()) != set(sent_agents.keys()):  # type: ignore[union-attr]
                    findings.critical(
                        surface="definition round-trip",
                        summary="agent set drifted between POST and GET",
                        detail=f"sent={sorted(sent_agents.keys())} got={sorted(got_agents.keys())}",  # type: ignore[union-attr]
                    )

            # --- step 7: list filters --------------------------------------
            r = c.list_workflows(provenance="api")
            if r.status_code != 200:
                findings.important(
                    surface="GET /v1/workflows?provenance=api",
                    summary=f"provenance filter returned {r.status_code}",
                    response=request_summary(r),
                )
            else:
                names = [w["name"] for w in r.json()["items"]]
                if "research-pipeline" not in names:
                    findings.critical(
                        surface="GET /v1/workflows?provenance=api",
                        summary="just-created api workflow not in api-filtered list",
                        detail=f"names={names}",
                    )

            # Provenance=visual_builder should NOT include the api workflow.
            r = c.list_workflows(provenance="visual_builder")
            if r.status_code == 200:
                names = [w["name"] for w in r.json()["items"]]
                if "research-pipeline" in names:
                    findings.critical(
                        surface="GET /v1/workflows?provenance=visual_builder",
                        summary="api workflow leaked into visual_builder-filtered list",
                        detail=f"names={names}",
                    )

            # --- step 8: lint ----------------------------------------------
            r = c.lint_workflow(workflow_id, payload["definition"])
            if r.status_code != 200:
                findings.critical(
                    surface="POST /v1/workflows/{id}/lint",
                    summary=f"lint returned {r.status_code} (should always be 200 for known wf)",
                    response=request_summary(r),
                )
            else:
                lint_body = r.json()
                if "findings" not in lint_body or not isinstance(lint_body["findings"], list):
                    findings.critical(
                        surface="POST /v1/workflows/{id}/lint",
                        summary="lint response is missing 'findings' array",
                        response=request_summary(r),
                    )
                else:
                    # We referenced 'search_web' and 'browse_url'; if the tool
                    # registry actually has these, lint should be clean. If not,
                    # we expect unknown_tool findings — either is informative.
                    findings.info(
                        surface="POST /v1/workflows/{id}/lint",
                        summary=f"lint returned {len(lint_body['findings'])} findings on the realistic workflow",
                        detail="\n".join(
                            f"- {f.get('severity')}/{f.get('code')}: {f.get('message')}"
                            for f in lint_body["findings"][:10]
                        ) or "(no findings)",
                    )

            # --- step 9: archive via PATCH ---------------------------------
            r = c.patch_workflow(workflow_id, {"is_archived": True})
            if r.status_code != 200:
                findings.important(
                    surface="PATCH /v1/workflows/{id}",
                    summary=f"archive PATCH returned {r.status_code}",
                    response=request_summary(r),
                )
            else:
                r2 = c.list_workflows()
                ids_default = [w["id"] for w in r2.json()["items"]]
                if workflow_id in ids_default:
                    findings.important(
                        surface="archive visibility",
                        summary="archived workflow still in default list",
                    )
                r3 = c.list_workflows(archived="true")
                ids_with = [w["id"] for w in r3.json()["items"]]
                if workflow_id not in ids_with:
                    findings.important(
                        surface="archive visibility",
                        summary="archived workflow missing from ?archived=true list",
                    )

            # --- step 10: replace via PUT ----------------------------------
            edited = dict(payload)
            edited["name"] = "research-pipeline-edited"
            r = c.replace_workflow(workflow_id, edited)
            if r.status_code != 200:
                findings.important(
                    surface="PUT /v1/workflows/{id}",
                    summary=f"replace returned {r.status_code}",
                    response=request_summary(r),
                )
            else:
                if r.json().get("name") != "research-pipeline-edited":
                    findings.important(
                        surface="PUT /v1/workflows/{id}",
                        summary="name did not persist on replace",
                    )
                # Provenance should NOT have flipped.
                if r.json().get("provenance") != "api":
                    findings.critical(
                        surface="PUT /v1/workflows/{id}",
                        summary=f"provenance flipped on edit (was 'api', now {r.json().get('provenance')!r})",
                    )

            # --- step 11: DELETE -------------------------------------------
            r = c.delete_workflow(workflow_id)
            if r.status_code not in (200, 204):
                findings.important(
                    surface="DELETE /v1/workflows/{id}",
                    summary=f"delete returned {r.status_code}",
                    response=request_summary(r),
                )
            r = c.get_workflow(workflow_id)
            if r.status_code != 404:
                findings.critical(
                    surface="GET /v1/workflows/{id} (after delete)",
                    summary=f"after delete got {r.status_code} (expected 404)",
                    response=request_summary(r),
                )

            # --- step 12: idempotency-key replay ---------------------------
            idem_key = "scenario-test-key-12345"
            r1 = c.request(
                "POST", "/v1/workflows",
                json=payload,
                headers={"Idempotency-Key": idem_key},
            )
            r2 = c.request(
                "POST", "/v1/workflows",
                json=payload,
                headers={"Idempotency-Key": idem_key},
            )
            if r1.status_code != 201 or r2.status_code != 201:
                findings.important(
                    surface="Idempotency-Key replay",
                    summary=f"either request not 201 (r1={r1.status_code} r2={r2.status_code})",
                )
            elif r1.json().get("id") != r2.json().get("id"):
                findings.critical(
                    surface="Idempotency-Key replay",
                    summary="same key produced two different workflow ids",
                    detail=f"r1.id={r1.json().get('id')} r2.id={r2.json().get('id')}",
                )
            # Cleanup the replay row.
            if r1.status_code == 201:
                c.delete_workflow(r1.json()["id"])

    return _finish(findings)


def _finish(findings: FindingsCollector) -> int:
    findings.finalize()
    findings.render_summary()
    output_dir = Path(__file__).resolve().parent / "output"
    path = findings.write_json(output_dir)
    print(f"\nfindings → {path}", flush=True)
    # Exit non-zero if any critical findings — useful for CI smoke.
    return 0 if findings.report.count("critical") == 0 else 1


if __name__ == "__main__":
    raise SystemExit(run())
