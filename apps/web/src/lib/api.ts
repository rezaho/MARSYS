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
export type NodeType = components["schemas"]["NodeType"];
export type EdgeType = components["schemas"]["EdgeType"];
export type EdgePattern = components["schemas"]["EdgePattern"];

export type ToolInfo = components["schemas"]["ToolInfo"];
export type ToolListResponse = components["schemas"]["ToolListResponse"];

export type ImportWarningPayload = components["schemas"]["ImportWarningPayload"];
export type WorkflowImportResponse = components["schemas"]["WorkflowImportResponse"];

export type LintFinding = components["schemas"]["LintFinding"];
export type LintResponse = components["schemas"]["LintResponse"];
export type LintSeverity = LintFinding["severity"];
export type LintCode = LintFinding["code"];

export type ErrorEnvelope = components["schemas"]["ErrorEnvelope"];

export type { paths };

function resolveBaseUrl(): string {
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
