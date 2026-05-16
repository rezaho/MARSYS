"""
B2 demonstration — surface the REAL Anthropic 4xx error body, with the
standard Anthropic API, in the same family as the parallel-tracing scripts.

WHY this script is different from the other tracing scripts: those test the
SUCCESS path (they reach `Success: True`). B2 is an ERROR-path fix — the
async adapter used to discard the provider's 4xx JSON body and report the
opaque aiohttp reason phrase "Bad Request". A normal successful run never
exercises that. So this script deliberately forces a real Anthropic HTTP
400 (via `temperature=1.5` — our ModelConfigSpec accepts 0..2, but the
Anthropic Messages API requires temperature <= 1) and shows what the user
now sees.

  BEFORE B2:  error == "Bad Request"            (provider body discarded)
  AFTER  B2:  error  mentions "temperature ..." (the real Anthropic message)
              classification category == "invalid_request"

Run it like the other GAIA tracing scripts:

    python packages/framework/benchmarks/GAIA/test_anthropic_4xx_error_body.py

Costs one cheap, immediately-rejected Anthropic call. Requires
ANTHROPIC_API_KEY (loaded from the repo-root .env, same as the canonical
Anthropic tracing script).
"""

import asyncio
import logging
import os
import sys
from pathlib import Path


def _load_repo_dotenv() -> None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / ".env"
        if candidate.is_file():
            for raw in candidate.read_text().splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
            return


_load_repo_dotenv()

if not os.environ.get("ANTHROPIC_API_KEY"):
    sys.exit(
        "ANTHROPIC_API_KEY is not set and was not found in a repo-root .env. "
        "Export it or add it to .env, then re-run."
    )

from marsys.agents.registry import AgentRegistry
from marsys.agents.utils import init_agent_logging
from marsys.coordination import Orchestra
from marsys.coordination.config import ExecutionConfig, StatusConfig
from marsys.coordination.tracing import TracingConfig
from marsys.agents.serialize import AgentSpec
from marsys.models.serialize import ModelConfigSpec
from marsys.coordination.topology.serialize import (
    EdgeSpec,
    NodeSpec,
    TopologySpec,
    WorkflowDefinition,
    pydantic_to_topology,
)

TRACE_DIR = str(Path(__file__).parent / "results" / "traces_anthropic_4xx_error_body")
_TMP_LOG_FILE = str(Path(TRACE_DIR) / "_run.log")
Path(TRACE_DIR).mkdir(parents=True, exist_ok=True)

init_agent_logging(level=logging.ERROR, log_file=_TMP_LOG_FILE, file_level=logging.DEBUG)

print(f"Traces will be written to: {TRACE_DIR}")

# provider="anthropic" (standard API key), valid model id, but temperature
# 1.5: accepted by ModelConfigSpec (le=2.0), rejected by the Anthropic
# Messages API with a 400 invalid_request_error. This drives the async
# adapter's 4xx path — exactly the B2 code.
bad_temp_spec = ModelConfigSpec(
    type="api",
    provider="anthropic",
    name="claude-haiku-4-5-20251001",
    temperature=1.5,
    max_tokens=512,
)

solo = AgentSpec(
    name="Solo",
    goal="Answer the question.",
    instruction=(
        "Answer the user's question in one sentence, then call "
        "`terminate_workflow` with your answer."
    ),
    agent_model=bad_temp_spec,
    memory_retention="single_run",
)

workflow = WorkflowDefinition(
    topology=TopologySpec(
        nodes=[NodeSpec(name="Solo", node_type="agent", agent_ref="Solo")],
        edges=[],
        metadata={"entry_point": "Solo", "exit_points": ["Solo"]},
    ),
    agents={"Solo": solo},
)

topology = pydantic_to_topology(workflow, tool_registry={})


def _error_lines_from_run_log() -> list:
    """The model/API error is logged (deterministically) to the run log:
    `... response: error='<msg>' error_type=... classification={...}` and
    `... Model/API call failed: ... API Error: <msg>`. Read those lines —
    the trace JSON nests this differently per writer; the log is stable."""
    log_dir = Path(TRACE_DIR)
    lines = []
    for log in list(log_dir.glob("*.log")) + [Path(_TMP_LOG_FILE)]:
        if log.is_file():
            for ln in log.read_text(errors="replace").splitlines():
                if "Model/API call failed:" in ln or "response: error=" in ln:
                    lines.append(ln.split("| ")[-1].strip())
    seen = set()
    return [x for x in lines if not (x in seen or seen.add(x))]


async def main():
    try:
        result = await Orchestra.run(
            task="What is the capital of France?",
            topology=topology,
            agent_registry=AgentRegistry,
            execution_config=ExecutionConfig(
                status=StatusConfig.from_verbosity(1),
                step_timeout=60.0,
                tracing=TracingConfig(
                    enabled=True,
                    output_dir=TRACE_DIR,
                    include_message_content=True,
                ),
            ),
            max_steps=3,
        )

        print(f"\n{'='*64}")
        print(f"Workflow success : {result.success}   (expected: False — the call 400s)")
        print(f"Final response   : {result.final_response}")

        errors = _error_lines_from_run_log()

        print(f"\n{'='*64}")
        print("MODEL/API ERROR AS THE FRAMEWORK NOW REPORTS IT:")
        for e in errors:
            print(f"  - {e[:320]}")

        joined = " ".join(errors).lower()
        mentions_real = "temperature" in joined and "invalid_request" in joined
        only_bad_request = ("bad request" in joined) and not mentions_real

        print(f"\n{'='*64}")
        if mentions_real:
            print("B2 WORKING: the trace carries Anthropic's real "
                  "invalid_request_error message (mentions 'temperature'),")
            print("not the opaque 'Bad Request'. The 4xx body survived the "
                  "async path.")
        elif only_bad_request:
            print("B2 NOT in effect: only the opaque 'Bad Request' reason "
                  "phrase is present — the provider body was discarded "
                  "(pre-B2 behavior).")
        else:
            print("Inconclusive — inspect the trace + run log under:")
            print(f"  {TRACE_DIR}")
        print(f"{'='*64}")

    finally:
        AgentRegistry.clear()


if __name__ == "__main__":
    asyncio.run(main())
