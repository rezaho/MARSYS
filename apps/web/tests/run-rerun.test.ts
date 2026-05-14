/**
 * run-rerun helper tests.
 *
 * Covers AC-408..414: rerunRun copies workflow_id + task_input.text +
 * task_input.attachments verbatim. rerunRunWithAttachments lets the
 * caller substitute a filtered attachments list (the "remaining
 * attachments" path after a deleted-file confirm).
 */
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../src/lib/api", async () => {
  const actual = await vi.importActual<typeof import("../src/lib/api")>("../src/lib/api");
  return { ...actual, createRun: vi.fn() };
});

import { createRun as mockCreateRun, type RunRead } from "../src/lib/api";
import { rerunRun, rerunRunWithAttachments } from "../src/lib/run-rerun";

beforeEach(() => {
  vi.clearAllMocks();
});

const baseRun: RunRead = {
  schema_version: 1,
  id: "run-A",
  workflow_id: "wf-research",
  status: "succeeded",
  task_input: { text: "summarize this", attachments: ["file-1", "file-2"] },
  trigger: "manual",
  started_at: null,
  finished_at: null,
  total_steps: null,
  total_duration_ms: null,
  total_tokens_input: 0,
  total_tokens_output: 0,
  total_cost_usd: 0.0,
  final_response: null,
  error: null,
  created_at: "2026-05-13T00:00:00Z",
  updated_at: "2026-05-13T00:00:00Z",
};

describe("rerunRun", () => {
  it("fires POST /v1/runs with workflow_id + task_input copied verbatim", async () => {
    (mockCreateRun as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      schema_version: 1,
      run_id: "run-B",
      status: "queued",
    });
    await rerunRun("token", baseRun);
    expect(mockCreateRun).toHaveBeenCalledTimes(1);
    const payload = (mockCreateRun as ReturnType<typeof vi.fn>).mock.calls[0][1];
    expect(payload.workflow_id).toBe("wf-research");
    expect(payload.task_input.text).toBe("summarize this");
    expect(payload.task_input.attachments).toEqual(["file-1", "file-2"]);
    expect(payload.trigger).toBe("manual");
  });
});

describe("rerunRunWithAttachments", () => {
  it("substitutes the attachments list while preserving text + workflow", async () => {
    (mockCreateRun as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      schema_version: 1,
      run_id: "run-C",
      status: "queued",
    });
    await rerunRunWithAttachments("token", baseRun, ["file-1"]);
    expect(mockCreateRun).toHaveBeenCalledTimes(1);
    const payload = (mockCreateRun as ReturnType<typeof vi.fn>).mock.calls[0][1];
    expect(payload.workflow_id).toBe("wf-research");
    expect(payload.task_input.text).toBe("summarize this");
    expect(payload.task_input.attachments).toEqual(["file-1"]);
  });
});
