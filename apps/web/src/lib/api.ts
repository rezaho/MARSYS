// v0.3 placeholder types — Session 02 generates these from Pydantic via datamodel-code-generator.
export interface FrameworkInfo {
  version: string;
}
export interface SprenInfo {
  active: boolean;
  version: string;
}
export interface BootstrapResponse {
  framework: FrameworkInfo;
  spren: SprenInfo;
  surfaces: string[];
  capabilities: Record<string, boolean>;
  endpoints: Record<string, string>;
  started_at: string;
  data_dir: string;
}

function resolveBaseUrl(): string {
  if (typeof window !== "undefined" && window.__SPREN_PORT__) {
    return `http://127.0.0.1:${window.__SPREN_PORT__}`;
  }
  const envUrl = import.meta.env.VITE_SPREN_API_URL as string | undefined;
  if (envUrl) return envUrl;
  return ""; // same-origin (production: sidecar serves the bundle on /)
}

export async function fetchBootstrap(token: string): Promise<BootstrapResponse> {
  const base = resolveBaseUrl();
  const res = await fetch(`${base}/v1/bootstrap`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) {
    throw new Error(`bootstrap failed: ${res.status} ${res.statusText}`);
  }
  return res.json();
}
