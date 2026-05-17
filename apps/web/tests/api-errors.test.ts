/**
 * ApiError + failResponse contract (WF-BUG-RUN-1).
 *
 * createRun is the first adopter of the structured-envelope parser.
 * Callers branch on `.code`/`.status`; `.message` is human-facing only.
 * Mocks fetch at the boundary (repo convention) rather than relying on a
 * jsdom global Response.
 */
import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, createRun } from "../src/lib/api";

function fakeResponse(status: number, body: string): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 400 ? "Bad Request" : "Internal Server Error",
    text: async () => body,
    json: async () => JSON.parse(body),
  } as unknown as Response;
}

const REQ = {
  workflow_id: "wf",
  task_input: { text: "", attachments: [] },
  trigger: "manual" as const,
};

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("createRun structured error envelope (WF-BUG-RUN-1)", () => {
  it("parses {error:{code,message}} into a typed ApiError", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        fakeResponse(
          400,
          JSON.stringify({
            error: {
              code: "VALIDATION_FAILED",
              message: "No api_key in secrets store for provider 'anthropic'",
              details: {},
            },
          }),
        ),
      ),
    );
    const err = await createRun("tok", REQ).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).status).toBe(400);
    expect((err as ApiError).code).toBe("VALIDATION_FAILED");
    expect((err as ApiError).message).toBe(
      "No api_key in secrets store for provider 'anthropic'",
    );
  });

  it("falls back to the raw body when not the structured envelope", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => fakeResponse(500, "Internal Server Error")),
    );
    const err = await createRun("tok", REQ).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).status).toBe(500);
    expect((err as ApiError).code).toBeNull();
    expect((err as ApiError).message).toBe("Internal Server Error");
  });
});
