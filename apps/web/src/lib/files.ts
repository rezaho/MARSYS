/**
 * File upload + download client helpers.
 *
 * ``uploadFile`` uses ``XMLHttpRequest`` (not fetch) because fetch+ReadableStream
 * does NOT expose ``upload.onprogress`` reliably across browsers (plan §10.3).
 * The other helpers use fetch.
 */
import type { FileMetadata, FileUploadResponse } from "./api";
import { resolveBaseUrl } from "./api";

export interface UploadProgress {
  /** 0–100 percentage as integers. */
  percent: number;
  /** Bytes uploaded so far. */
  loaded: number;
  /** Total bytes (-1 if not computable). */
  total: number;
}

export interface UploadOptions {
  onProgress?: (event: UploadProgress) => void;
  /** AbortSignal for canceling an in-flight upload. */
  signal?: AbortSignal;
}

export class UploadError extends Error {
  status: number;
  /** ErrorCode literal from the response envelope, when available. */
  code?: string;
  /** Optional details payload from the error envelope. */
  details?: Record<string, unknown>;

  constructor(message: string, status: number, code?: string, details?: Record<string, unknown>) {
    super(message);
    this.name = "UploadError";
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

export async function uploadFile(
  token: string,
  file: File,
  options: UploadOptions = {},
): Promise<FileUploadResponse> {
  const url = `${resolveBaseUrl()}/v1/files`;
  const form = new FormData();
  form.append("file", file, file.name);

  return new Promise<FileUploadResponse>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", url);
    xhr.setRequestHeader("Authorization", `Bearer ${token}`);
    xhr.responseType = "json";

    if (options.onProgress) {
      xhr.upload.addEventListener("progress", (event) => {
        if (event.lengthComputable) {
          options.onProgress?.({
            percent: Math.round((event.loaded / event.total) * 100),
            loaded: event.loaded,
            total: event.total,
          });
        } else {
          options.onProgress?.({
            percent: 0,
            loaded: event.loaded,
            total: -1,
          });
        }
      });
    }

    if (options.signal) {
      const onAbort = (): void => xhr.abort();
      if (options.signal.aborted) {
        xhr.abort();
      } else {
        options.signal.addEventListener("abort", onAbort, { once: true });
      }
    }

    xhr.addEventListener("load", () => {
      const status = xhr.status;
      if (status >= 200 && status < 300) {
        resolve(xhr.response as FileUploadResponse);
        return;
      }
      const body = (xhr.response ?? {}) as {
        error?: { code?: string; message?: string; details?: Record<string, unknown> };
      };
      reject(
        new UploadError(
          body?.error?.message ?? `upload failed: ${status}`,
          status,
          body?.error?.code,
          body?.error?.details,
        ),
      );
    });

    xhr.addEventListener("error", () => {
      reject(new UploadError("network error during upload", 0));
    });
    xhr.addEventListener("abort", () => {
      reject(new UploadError("upload aborted", 0));
    });

    xhr.send(form);
  });
}

export async function getFileMetadata(token: string, fileId: string): Promise<FileMetadata> {
  const res = await fetch(`${resolveBaseUrl()}/v1/files/${fileId}`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) {
    throw new Error(`get file metadata failed: ${res.status}`);
  }
  return (await res.json()) as FileMetadata;
}

export function fileDownloadUrl(fileId: string): string {
  return `${resolveBaseUrl()}/v1/files/${fileId}/download`;
}

/** Triggers a browser download. Auth header is sent via fetch + Blob URL —
 *  required because ``<a href>`` won't carry the Authorization header. */
export async function downloadFile(token: string, fileId: string, filename?: string): Promise<void> {
  const res = await fetch(fileDownloadUrl(fileId), {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) {
    throw new Error(`download failed: ${res.status}`);
  }
  const blob = await res.blob();
  const objectUrl = URL.createObjectURL(blob);
  try {
    const a = document.createElement("a");
    a.href = objectUrl;
    if (filename) a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

export async function deleteFile(token: string, fileId: string): Promise<void> {
  const res = await fetch(`${resolveBaseUrl()}/v1/files/${fileId}`, {
    method: "DELETE",
    headers: { Authorization: `Bearer ${token}` },
  });
  if (res.status === 204) return;
  let code: string | undefined;
  let details: Record<string, unknown> | undefined;
  try {
    const body = (await res.json()) as {
      error?: { code?: string; details?: Record<string, unknown> };
    };
    code = body?.error?.code;
    details = body?.error?.details;
  } catch {
    // Non-JSON body; fall through with default message.
  }
  throw new UploadError(`delete failed: ${res.status}`, res.status, code, details);
}
