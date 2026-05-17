/**
 * Aggregate /v1/runs/events SSE consumer for the /runs list page.
 *
 * Emits ``RunCreated`` / ``RunUpdated`` / ``RunFinished`` / ``RunCancelled``
 * payloads as they fire on the server's RunsBroker. Same fetch + reader
 * pattern as ``run-sse.ts``.
 */
import type { RunListItem, RunsListEvent } from "./api";

export interface RunsListSseHandlers {
  onCreated?: (run: RunListItem) => void;
  onUpdated?: (run: RunListItem) => void;
  onFinished?: (run: RunListItem) => void;
  onCancelled?: (run: RunListItem) => void;
  onLagged?: (droppedCount: number) => void;
  onOpen?: () => void;
  onError?: (err: unknown) => void;
  onClose?: () => void;
}

export interface RunsListSseHandle {
  close: () => void;
}

const BACKOFF_MS = [1000, 2000, 4000, 8000];

export function openRunsListSse(
  baseUrl: string,
  token: string,
  handlers: RunsListSseHandlers = {},
): RunsListSseHandle {
  const controller = new AbortController();
  let closed = false;
  let backoffIdx = 0;

  async function loop(): Promise<void> {
    while (!closed) {
      try {
        const res = await fetch(`${baseUrl || ""}/v1/runs/events`, {
          headers: {
            Authorization: `Bearer ${token}`,
            Accept: "text/event-stream",
          },
          signal: controller.signal,
        });
        if (!res.ok || !res.body) {
          throw new Error(`runs-list SSE open failed: ${res.status}`);
        }
        backoffIdx = 0;
        handlers.onOpen?.();

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        while (!closed) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          let idx: number;
          while ((idx = buffer.indexOf("\n\n")) >= 0) {
            const chunk = buffer.slice(0, idx);
            buffer = buffer.slice(idx + 2);
            handleChunk(chunk, handlers);
          }
        }
      } catch (err) {
        if (closed) return;
        if ((err as { name?: string }).name === "AbortError") return;
        handlers.onError?.(err);
      }
      if (closed) return;
      const delay = BACKOFF_MS[Math.min(backoffIdx, BACKOFF_MS.length - 1)];
      backoffIdx += 1;
      await sleep(delay, controller.signal);
    }
  }

  void loop().finally(() => handlers.onClose?.());

  return {
    close: () => {
      closed = true;
      controller.abort();
    },
  };
}

function handleChunk(chunk: string, handlers: RunsListSseHandlers): void {
  let event: string | null = null;
  const dataLines: string[] = [];
  for (const rawLine of chunk.split("\n")) {
    const line = rawLine.trimEnd();
    if (!line || line.startsWith(":")) continue;
    const colonIdx = line.indexOf(":");
    if (colonIdx < 0) continue;
    const field = line.slice(0, colonIdx);
    const value = line.slice(colonIdx + 1).replace(/^ /, "");
    if (field === "event") event = value;
    else if (field === "data") dataLines.push(value);
  }
  if (dataLines.length === 0) return;
  const data = dataLines.join("\n");
  if (event === "marsys.stream.lagged") {
    try {
      const parsed = JSON.parse(data) as { dropped_count?: number };
      handlers.onLagged?.(parsed.dropped_count ?? 0);
    } catch {
      /* swallow */
    }
    return;
  }
  let parsed: RunsListEvent;
  try {
    parsed = JSON.parse(data) as RunsListEvent;
  } catch {
    return;
  }
  switch (parsed.type) {
    case "RunCreated":
      handlers.onCreated?.(parsed.run);
      break;
    case "RunUpdated":
      handlers.onUpdated?.(parsed.run);
      break;
    case "RunFinished":
      handlers.onFinished?.(parsed.run);
      break;
    case "RunCancelled":
      handlers.onCancelled?.(parsed.run);
      break;
  }
}

function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(new DOMException("aborted", "AbortError"));
      return;
    }
    const timer = setTimeout(resolve, ms);
    signal.addEventListener("abort", () => {
      clearTimeout(timer);
      reject(new DOMException("aborted", "AbortError"));
    }, { once: true });
  });
}
