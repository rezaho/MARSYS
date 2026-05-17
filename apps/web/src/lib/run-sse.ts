/**
 * Per-run SSE consumer.
 *
 * Uses ``fetch`` + ``ReadableStream`` instead of native ``EventSource``
 * because:
 * 1. Spren's auth dep requires ``Authorization: Bearer <token>``.
 *    Browsers do not let raw ``EventSource`` set headers.
 * 2. ``fetch`` lets us pass headers + observe ``Last-Event-ID`` resume
 *    semantics ourselves on reconnect.
 *
 * Reconnect: on transport error, we wait (exponential backoff
 * 1s → 2s → 4s → max 8s, infinite retries until close()) then re-open
 * with the most recent event id in the ``Last-Event-ID`` header.
 *
 * Parses the SSE wire format (``id: ...`` / ``event: ...`` / ``data: ...``
 * blocks separated by ``\n\n``) directly. The framework's
 * ``aggui_event_to_sse`` produces this exact format via
 * ``ag_ui.encoder.EventEncoder``.
 */

export interface AGUIEventPayload {
  type: string;
  [key: string]: unknown;
}

export interface RunSseHandlers {
  onEvent?: (event: AGUIEventPayload, eventName: string) => void;
  onOpen?: () => void;
  onError?: (err: unknown) => void;
  onClose?: () => void;
  onGeneration?: (metadata: GenerationMetadata) => void;
  onStreamGap?: () => void;
  onStreamLagged?: (droppedCount: number) => void;
  onReconnecting?: (nextDelayMs: number) => void;
}

export interface GenerationMetadata {
  model: string;
  provider: string;
  prompt_tokens: number;
  completion_tokens: number;
  reasoning_tokens?: number;
  finish_reason?: string;
}

export interface RunSseHandle {
  close: () => void;
}

const BACKOFF_MS = [1000, 2000, 4000, 8000];

export function openRunSse(
  baseUrl: string,
  runId: string,
  token: string,
  handlers: RunSseHandlers = {},
): RunSseHandle {
  const controller = new AbortController();
  let closed = false;
  let lastEventId: string | null = null;
  let backoffIdx = 0;

  async function loop(): Promise<void> {
    while (!closed) {
      try {
        const headers: HeadersInit = {
          Authorization: `Bearer ${token}`,
          Accept: "text/event-stream",
        };
        if (lastEventId) headers["Last-Event-ID"] = lastEventId;

        const res = await fetch(
          `${baseUrl || ""}/v1/runs/${runId}/events`,
          { headers, signal: controller.signal },
        );
        if (res.status === 204) {
          // Terminal run; no live stream.
          return;
        }
        if (!res.ok || !res.body) {
          throw new Error(`SSE open failed: ${res.status}`);
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
            const parsed = parseSseChunk(chunk);
            if (!parsed) continue;
            if (parsed.id) lastEventId = parsed.id;
            const eventName = parsed.event ?? "message";
            const payload = parseEventData(parsed.data);
            if (!payload) continue;
            const withType: AGUIEventPayload = payload.type
              ? payload
              : { ...payload, type: eventName };
            handlers.onEvent?.(withType, eventName);
            dispatchCustomEvent(withType, eventName, handlers);
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
      handlers.onReconnecting?.(delay);
      await sleep(delay, controller.signal);
    }
  }

  void loop().finally(() => {
    handlers.onClose?.();
  });

  return {
    close: () => {
      closed = true;
      controller.abort();
    },
  };
}

interface SseChunk {
  id: string | null;
  event: string | null;
  data: string;
}

function parseSseChunk(chunk: string): SseChunk | null {
  let id: string | null = null;
  let event: string | null = null;
  const dataLines: string[] = [];
  for (const rawLine of chunk.split("\n")) {
    const line = rawLine.trimEnd();
    if (!line || line.startsWith(":")) continue;
    const colonIdx = line.indexOf(":");
    if (colonIdx < 0) continue;
    const field = line.slice(0, colonIdx);
    const value = line.slice(colonIdx + 1).replace(/^ /, "");
    if (field === "id") id = value;
    else if (field === "event") event = value;
    else if (field === "data") dataLines.push(value);
  }
  if (dataLines.length === 0) return null;
  return { id, event, data: dataLines.join("\n") };
}

function parseEventData(data: string): AGUIEventPayload | null {
  try {
    return JSON.parse(data) as AGUIEventPayload;
  } catch {
    return null;
  }
}

function dispatchCustomEvent(
  event: AGUIEventPayload,
  eventName: string,
  handlers: RunSseHandlers,
): void {
  // Spren-side server emits Custom("marsys.stream.gap") as a named SSE event.
  if (eventName === "marsys.stream.gap") {
    handlers.onStreamGap?.();
    return;
  }

  const isCustom = event.type === "Custom";
  if (!isCustom) return;

  const name = (event["name"] as string | undefined) ?? "";
  const value = (event["value"] as Record<string, unknown> | undefined) ?? {};
  if (name === "marsys.generation.metadata") {
    handlers.onGeneration?.(value as unknown as GenerationMetadata);
  } else if (name === "marsys.stream.gap") {
    handlers.onStreamGap?.();
  } else if (name === "marsys.stream.lagged") {
    const dropped = Number(value["dropped_count"]) || 0;
    handlers.onStreamLagged?.(dropped);
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
