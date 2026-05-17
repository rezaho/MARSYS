// Generated TS types are produced from /openapi.json via openapi-typescript at
// build time (apps/web/scripts/generate-types.mjs). Hand-writing TS types that
// mirror Pydantic models is forbidden (SP-005); add new types by adding new
// FastAPI routes instead.
import type { components, paths } from "./api-types.generated";

export type FrameworkInfo = components["schemas"]["FrameworkInfo"];
export type SprenInfo = components["schemas"]["SprenInfo"];
export type BootstrapResponse = components["schemas"]["BootstrapResponse"];

export type Workflow = components["schemas"]["Workflow"];
export type WorkflowDefinition = components["schemas"]["WorkflowDefinition"];
export type WorkflowCreateRequest = components["schemas"]["WorkflowCreateRequest"];
export type WorkflowUpdateRequest = components["schemas"]["WorkflowUpdateRequest"];
export type WorkflowListResponse = components["schemas"]["WorkflowListResponse"];
export type WorkflowProvenance = Workflow["provenance"];

export type AgentSpec = components["schemas"]["AgentSpec"];
export type ModelConfigSpec = components["schemas"]["ModelConfigSpec"];
export type ExecutionConfigSpec = components["schemas"]["ExecutionConfigSpec"];
export type TopologySpec = components["schemas"]["TopologySpec"];
export type NodeSpec = components["schemas"]["NodeSpec"];
export type EdgeSpec = components["schemas"]["EdgeSpec"];
export type NodeKind = components["schemas"]["NodeKind"];

export type ToolInfo = components["schemas"]["ToolInfo"];
export type ToolListResponse = components["schemas"]["ToolListResponse"];

export type ImportWarningPayload = components["schemas"]["ImportWarningPayload"];
export type WorkflowImportResponse = components["schemas"]["WorkflowImportResponse"];

export type LintFinding = components["schemas"]["LintFinding"];
export type LintResponse = components["schemas"]["LintResponse"];
export type LintSeverity = LintFinding["severity"];
export type LintCode = LintFinding["code"];

export type ErrorEnvelope = components["schemas"]["ErrorEnvelope"];

export type RunStatus = components["schemas"]["RunStatus"];
export type TaskInput = components["schemas"]["TaskInput"];
export type RunCreate = components["schemas"]["RunCreate"];
export type RunCreateResponse = components["schemas"]["RunCreateResponse"];
export type RunRead = components["schemas"]["RunRead"];
export type RunListItem = components["schemas"]["RunListItem"];
export type RunListResponse = components["schemas"]["RunListResponse"];
export type RunCreatedEvent = components["schemas"]["RunCreatedEvent"];
export type RunUpdatedEvent = components["schemas"]["RunUpdatedEvent"];
export type RunFinishedEvent = components["schemas"]["RunFinishedEvent"];
export type RunCancelledEvent = components["schemas"]["RunCancelledEvent"];
export type RunsListEvent =
  | RunCreatedEvent
  | RunUpdatedEvent
  | RunFinishedEvent
  | RunCancelledEvent;

// Session 05 — trace + files + artifacts types
export type RunTrace = components["schemas"]["RunTrace"];
export type SpanNode = components["schemas"]["SpanNode"];
export type SpanKind = SpanNode["kind"];
export type RunTraceCompletionStatus = components["schemas"]["RunTraceCompletionStatus"];
export type FileMetadata = components["schemas"]["FileMetadata"];
export type FileUploadResponse = components["schemas"]["FileUploadResponse"];
export type ArtifactInfo = components["schemas"]["ArtifactInfo"];
export type ArtifactListResponse = components["schemas"]["ArtifactListResponse"];

export type { paths };

export function isTerminalStatus(status: RunStatus): boolean {
  return status === "succeeded" || status === "failed" || status === "cancelled";
}

export function resolveBaseUrl(): string {
  if (typeof window !== "undefined" && window.__SPREN_PORT__) {
    return `http://127.0.0.1:${window.__SPREN_PORT__}`;
  }
  const envUrl = import.meta.env.VITE_SPREN_API_URL as string | undefined;
  if (envUrl) return envUrl;
  return ""; // same-origin (production: sidecar serves the bundle on /)
}

function authHeader(token: string): HeadersInit {
  return { Authorization: `Bearer ${token}` };
}

/**
 * A failed API call. `code` is the backend envelope's machine code
 * (`{"error":{"code":...}}`) and is the contract callers branch on;
 * `message` is the human-readable text to show the user; `raw` is the
 * untouched body for diagnostics. Branch on `code`/`status`, never on
 * `message` substrings.
 */
export class ApiError extends Error {
  readonly status: number;
  readonly code: string | null;
  readonly raw: string;

  constructor(args: { status: number; code: string | null; message: string; raw: string }) {
    super(args.message);
    this.name = "ApiError";
    this.status = args.status;
    this.code = args.code;
    this.raw = args.raw;
  }
}

/**
 * Parse the backend's `{"error":{"code","message","details"}}` envelope
 * from a non-OK response and throw a typed {@link ApiError}. Falls back to
 * the raw body / status line when the body is not the structured envelope.
 *
 * Adopted by `createRun` (the run-create path — WF-BUG-RUN-1). The other
 * client functions still throw plain `Error`; migrating them is a separate
 * mechanical change deliberately kept out of the run-create bug fix to
 * avoid altering unrelated call sites' error-display behaviour.
 */
async function failResponse(res: Response, action: string): Promise<never> {
  const raw = await res.text();
  let code: string | null = null;
  let message = "";
  try {
    const parsed = JSON.parse(raw) as { error?: { code?: string; message?: string } };
    if (parsed?.error) {
      code = parsed.error.code ?? null;
      message = parsed.error.message ?? "";
    }
  } catch {
    // body is not JSON — fall through to the raw-text message
  }
  if (!message) {
    message = raw.trim() || `${action} failed: ${res.status} ${res.statusText}`;
  }
  throw new ApiError({ status: res.status, code, message, raw });
}

export async function fetchBootstrap(token: string): Promise<BootstrapResponse> {
  const res = await fetch(`${resolveBaseUrl()}/v1/bootstrap`, {
    headers: authHeader(token),
  });
  if (!res.ok) {
    throw new Error(`bootstrap failed: ${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<BootstrapResponse>;
}

export interface ListWorkflowsOptions {
  archived?: boolean;
  include_drafts?: boolean;
  provenance?: WorkflowProvenance;
}

export async function listWorkflows(
  token: string,
  options: ListWorkflowsOptions = {},
): Promise<WorkflowListResponse> {
  const url = new URL("/v1/workflows", resolveBaseUrl() || window.location.origin);
  if (options.archived) url.searchParams.set("archived", "true");
  if (options.include_drafts) url.searchParams.set("include_drafts", "true");
  if (options.provenance) url.searchParams.set("provenance", options.provenance);
  const res = await fetch(url.toString(), { headers: authHeader(token) });
  if (!res.ok) throw new Error(`list workflows failed: ${res.status} ${res.statusText}`);
  return res.json() as Promise<WorkflowListResponse>;
}

export async function getWorkflow(token: string, id: string): Promise<Workflow> {
  const res = await fetch(`${resolveBaseUrl()}/v1/workflows/${id}`, {
    headers: authHeader(token),
  });
  if (!res.ok) throw new Error(`get workflow failed: ${res.status} ${res.statusText}`);
  return res.json() as Promise<Workflow>;
}

export async function replaceWorkflow(
  token: string,
  id: string,
  payload: WorkflowCreateRequest,
): Promise<Workflow> {
  const res = await fetch(`${resolveBaseUrl()}/v1/workflows/${id}`, {
    method: "PUT",
    headers: { ...authHeader(token), "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`replace workflow failed: ${res.status} ${body}`);
  }
  return res.json() as Promise<Workflow>;
}

export async function listTools(token: string): Promise<ToolListResponse> {
  const res = await fetch(`${resolveBaseUrl()}/v1/tools`, { headers: authHeader(token) });
  if (!res.ok) throw new Error(`list tools failed: ${res.status} ${res.statusText}`);
  return res.json() as Promise<ToolListResponse>;
}

export async function lintWorkflowById(token: string, id: string): Promise<LintResponse> {
  const res = await fetch(`${resolveBaseUrl()}/v1/workflows/${id}/lint`, {
    method: "POST",
    headers: authHeader(token),
  });
  if (!res.ok) throw new Error(`lint workflow failed: ${res.status} ${res.statusText}`);
  return res.json() as Promise<LintResponse>;
}

export async function createWorkflow(
  token: string,
  payload: WorkflowCreateRequest,
): Promise<Workflow> {
  const res = await fetch(`${resolveBaseUrl()}/v1/workflows`, {
    method: "POST",
    headers: { ...authHeader(token), "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`create workflow failed: ${res.status} ${body}`);
  }
  return res.json() as Promise<Workflow>;
}

export async function deleteWorkflow(token: string, id: string): Promise<void> {
  const res = await fetch(`${resolveBaseUrl()}/v1/workflows/${id}`, {
    method: "DELETE",
    headers: authHeader(token),
  });
  if (!res.ok && res.status !== 204) {
    const body = await res.text();
    throw new Error(`delete workflow failed: ${res.status} ${body}`);
  }
}

export async function importPythonWorkflow(
  token: string,
  file: File,
): Promise<WorkflowImportResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${resolveBaseUrl()}/v1/workflows/import-python`, {
    method: "POST",
    headers: authHeader(token),
    body: form,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`python import failed: ${res.status} ${body}`);
  }
  return res.json() as Promise<WorkflowImportResponse>;
}

// ---- Runs ----

export async function createRun(
  token: string,
  payload: RunCreate,
): Promise<RunCreateResponse> {
  const res = await fetch(`${resolveBaseUrl()}/v1/runs`, {
    method: "POST",
    headers: { ...authHeader(token), "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    await failResponse(res, "create run");
  }
  return res.json() as Promise<RunCreateResponse>;
}

export async function getRun(token: string, id: string): Promise<RunRead> {
  const res = await fetch(`${resolveBaseUrl()}/v1/runs/${id}`, {
    headers: authHeader(token),
  });
  if (!res.ok) throw new Error(`get run failed: ${res.status} ${res.statusText}`);
  return res.json() as Promise<RunRead>;
}

export interface ListRunsOptions {
  /** Single workflow id; mutually exclusive with ``workflow_ids``. */
  workflow_id?: string;
  /** Multi workflow id (comma-joined on the wire). */
  workflow_ids?: string[];
  /** Single status; mutually exclusive with ``statuses``. */
  status?: RunStatus;
  /** Multi status (comma-joined on the wire). */
  statuses?: RunStatus[];
  /** ISO 8601 absolute or Session 04's relative shorthand (e.g., ``"24h"``). */
  since?: string;
  /** ISO 8601 absolute upper bound. */
  until?: string;
  cursor?: string;
  limit?: number;
}

export async function listRuns(
  token: string,
  options: ListRunsOptions = {},
): Promise<RunListResponse> {
  const url = new URL("/v1/runs", resolveBaseUrl() || window.location.origin);
  if (options.workflow_ids && options.workflow_ids.length > 0) {
    url.searchParams.set("workflow_id", options.workflow_ids.join(","));
  } else if (options.workflow_id) {
    url.searchParams.set("workflow_id", options.workflow_id);
  }
  if (options.statuses && options.statuses.length > 0) {
    url.searchParams.set("status", options.statuses.join(","));
  } else if (options.status) {
    url.searchParams.set("status", options.status);
  }
  if (options.since) url.searchParams.set("since", options.since);
  if (options.until) url.searchParams.set("until", options.until);
  if (options.cursor) url.searchParams.set("cursor", options.cursor);
  if (options.limit) url.searchParams.set("limit", String(options.limit));
  const res = await fetch(url.toString(), { headers: authHeader(token) });
  if (!res.ok) throw new Error(`list runs failed: ${res.status} ${res.statusText}`);
  return res.json() as Promise<RunListResponse>;
}

export async function cancelRun(token: string, id: string): Promise<RunRead> {
  const res = await fetch(`${resolveBaseUrl()}/v1/runs/${id}/cancel`, {
    method: "POST",
    headers: authHeader(token),
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`cancel run failed: ${res.status} ${body}`);
  }
  return res.json() as Promise<RunRead>;
}
