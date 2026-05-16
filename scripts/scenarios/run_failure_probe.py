"""Run-failure probe — reproduce the post-validation execution error trail.

WF-BUG-RUN-3: the user reports "a lot of errors in the backend" when running a
workflow. A no-key environment fails at the clean 400 validation gate (RUN-1),
producing no backend trace. This probe assumes the sidecar was started with a
*deliberately-invalid* API key (e.g.
``SPREN_ANTHROPIC_API_KEY=sk-ant-FAKE-...``) so run-create passes the
secrets-store presence check and execution proceeds far enough to fail at the
real provider call — surfacing whatever orchestration error trail the user is
seeing.

This connects to an ALREADY-RUNNING sidecar (it does not spawn one) so the
caller controls the env. Pass the port + token as argv or env.

Usage:
    uv run --package spren python scripts/scenarios/run_failure_probe.py \
        <port> <token>
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx  # noqa: E402

from _shared import FindingsCollector, request_summary  # noqa: E402

JOURNEY = "run_failure_probe"


def valid_definition() -> dict[str, object]:
    """Minimal v0.3 runnable workflow: Start → Agent → End.

    Session 07 node model: Core det-nodes (start/end) materialize to framework
    StartNode/EndNode instances. NOT Start→User→Agent→End — v0.3 has no
    executable User path (no UserNodeHandler wired; see Session 07 "Known
    v0.3 limitation"), so the RUN-3d regression gate uses Start→Agent→End to
    isolate det-node materialization from the User-handler concern.
    """
    return {
        "topology": {
            "nodes": [
                {"name": "Start", "node_type": "start", "agent_ref": None, "is_convergence_point": False, "metadata": {}},
                {"name": "assistant", "node_type": "agent", "agent_ref": "assistant", "is_convergence_point": True, "metadata": {}},
                {"name": "End", "node_type": "end", "agent_ref": None, "is_convergence_point": False, "metadata": {}},
            ],
            "edges": [
                {"source": "Start", "target": "assistant", "edge_type": "invoke", "bidirectional": False, "pattern": None, "metadata": {}},
                {"source": "assistant", "target": "End", "edge_type": "invoke", "bidirectional": False, "pattern": None, "metadata": {}},
            ],
            "rules": [],
        },
        "agents": {
            "assistant": {
                "agent_model": {
                    "type": "api",
                    "name": "claude-sonnet-4-6",
                    "provider": "anthropic",
                    "base_url": None,
                    "max_tokens": 1024,
                    "temperature": 0.7,
                    "thinking_budget": None,
                    "reasoning_effort": None,
                    "oauth_profile": None,
                },
                "name": "Assistant",
                "goal": "Answer the user's question.",
                "instruction": "You are a helpful assistant. Answer concisely.",
                "tools": [],
                "memory_retention": "session",
                "allowed_peers": [],
            },
        },
        "execution_config": {
            "convergence_timeout": 60.0,
            "response_format": "json",
            "user_interaction": "none",
            "convergence_policy": 0.7,
            "status": {"enabled": True},
        },
    }


def run(port: int, token: str) -> int:
    findings = FindingsCollector(JOURNEY)
    base = f"http://127.0.0.1:{port}"
    headers = {"Authorization": f"Bearer {token}"}
    print(f"=== {JOURNEY} against {base} ===", flush=True)

    with httpx.Client(base_url=base, headers=headers, timeout=30.0) as c:
        # 1. Create a runnable workflow.
        r = c.post(
            "/v1/workflows",
            json={
                "name": "run-probe",
                "description": "probe for WF-BUG-RUN-3",
                "definition": valid_definition(),
                "provenance": "api",
                "provenance_metadata": None,
            },
        )
        if r.status_code != 201:
            findings.critical(
                surface="POST /v1/workflows",
                summary=f"could not create probe workflow: {r.status_code}",
                response=request_summary(r),
            )
            return _finish(findings)
        wf_id = r.json()["id"]
        print(f"created workflow {wf_id}", flush=True)

        # 2. Trigger a run. With a fake key this passes the presence check
        #    and fails at the real Anthropic call — capturing the trail.
        r = c.post(
            "/v1/runs",
            json={
                "workflow_id": wf_id,
                "task_input": {"text": "Say hello.", "attachments": []},
                "trigger": "manual",
            },
        )
        print(f"POST /v1/runs -> {r.status_code}", flush=True)
        print(f"body: {r.text[:600]}", flush=True)
        findings.info(
            surface="POST /v1/runs",
            summary=f"run-create returned {r.status_code}",
            detail=r.text[:800],
            response=request_summary(r),
        )

        if r.status_code in (200, 201):
            run_id = r.json().get("run_id") or r.json().get("id")
            print(f"run_id={run_id} — letting it execute for 12s", flush=True)
            # Let the framework attempt execution; the trail lands in the
            # sidecar's stderr (read separately from its log file).
            time.sleep(12)
            rr = c.get(f"/v1/runs/{run_id}")
            print(f"GET /v1/runs/{run_id} -> {rr.status_code}: {rr.text[:600]}", flush=True)
            findings.info(
                surface="GET /v1/runs/{id}",
                summary=f"post-execution run state: {rr.status_code}",
                detail=rr.text[:800],
            )

    return _finish(findings)


def _finish(findings: FindingsCollector) -> int:
    findings.finalize()
    findings.render_summary()
    out = Path(__file__).resolve().parent / "output"
    p = findings.write_json(out)
    print(f"\nfindings -> {p}", flush=True)
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 2:
        print("usage: run_failure_probe.py <port> <token>", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(run(int(args[0]), args[1]))
