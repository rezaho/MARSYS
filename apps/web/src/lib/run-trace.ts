/**
 * Run-trace fetch + 2s polling helper.
 *
 * For completed runs, ``getRunTrace`` is called once. For ``status=running``
 * runs, the inspector subscribes to ``pollRunTrace`` which re-fetches every
 * 2 seconds and replaces the tree wholesale on each poll (plan §8.3 — the
 * v0.3 polling approach; SSE-derived per-event-delta is v0.4 polish).
 */
import type { RunTrace } from "./api";
import { resolveBaseUrl } from "./api";
import type { ArtifactInfo, WorkflowDefinition } from "./api";

export async function getRunTrace(token: string, runId: string): Promise<RunTrace> {
  const res = await fetch(`${resolveBaseUrl()}/v1/runs/${runId}/trace`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) {
    if (res.status === 404) {
      throw new TraceNotAvailableError(`no trace for run ${runId} yet`);
    }
    throw new Error(`get trace failed: ${res.status}`);
  }
  return (await res.json()) as RunTrace;
}

export async function getRunWorkflow(
  token: string,
  runId: string,
): Promise<WorkflowDefinition> {
  const res = await fetch(`${resolveBaseUrl()}/v1/runs/${runId}/workflow`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) {
    throw new Error(`get workflow snapshot failed: ${res.status}`);
  }
  return (await res.json()) as WorkflowDefinition;
}

export interface ArtifactListResponse {
  items: ArtifactInfo[];
}

export async function getRunArtifacts(
  token: string,
  runId: string,
): Promise<ArtifactListResponse> {
  const res = await fetch(`${resolveBaseUrl()}/v1/runs/${runId}/artifacts`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) {
    throw new Error(`get artifacts failed: ${res.status}`);
  }
  return (await res.json()) as ArtifactListResponse;
}

export function artifactDownloadUrl(runId: string, name: string): string {
  return `${resolveBaseUrl()}/v1/runs/${runId}/artifacts/${encodeURIComponent(name)}`;
}

export async function downloadArtifact(
  token: string,
  runId: string,
  name: string,
): Promise<void> {
  const res = await fetch(artifactDownloadUrl(runId, name), {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) {
    throw new Error(`download artifact failed: ${res.status}`);
  }
  const blob = await res.blob();
  const objectUrl = URL.createObjectURL(blob);
  try {
    const a = document.createElement("a");
    a.href = objectUrl;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

export class TraceNotAvailableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "TraceNotAvailableError";
  }
}
