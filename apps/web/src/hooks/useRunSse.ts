/**
 * SSE → jotai bridge for one in-flight run.
 *
 * Opens the per-run SSE on activation, derives orb state + token count
 * + cost from incoming AG-UI events, writes the run-state atoms.
 */
import { useSetAtom } from "jotai";
import { useEffect } from "react";

import { resolveBaseUrl } from "../lib/api";
import {
  type AGUIEventPayload,
  type GenerationMetadata,
  openRunSse,
} from "../lib/run-sse";
import {
  elapsedMsAtom,
  orbStateAtom,
  reconnectingAtom,
  runStatusAtom,
  tokenCountAtom,
  totalCostAtom,
} from "../stores/run";
import { useCapabilities } from "../providers/capabilities";

import { calculateCostFromMetadata } from "../lib/cost-client";

export function useRunSse(runId: string | null): void {
  const setOrbState = useSetAtom(orbStateAtom);
  const setStatus = useSetAtom(runStatusAtom);
  const setTokens = useSetAtom(tokenCountAtom);
  const setElapsed = useSetAtom(elapsedMsAtom);
  const setCost = useSetAtom(totalCostAtom);
  const setReconnecting = useSetAtom(reconnectingAtom);
  const { token } = useCapabilities();

  useEffect(() => {
    if (!runId || !token) return;

    const startedAt = Date.now();
    const tickInterval = window.setInterval(() => {
      setElapsed(Date.now() - startedAt);
    }, 200);

    const handle = openRunSse(resolveBaseUrl(), runId, token, {
      onOpen: () => setReconnecting(false),
      onReconnecting: () => setReconnecting(true),
      onEvent: (event: AGUIEventPayload, eventName: string) => {
        if (eventName === "RunStarted" || event.type === "RunStarted") {
          setOrbState("thinking");
          setStatus("running");
        } else if (
          eventName === "TextMessageStart" ||
          event.type === "TextMessageStart"
        ) {
          setOrbState("speaking");
        } else if (
          eventName === "TextMessageEnd" ||
          event.type === "TextMessageEnd"
        ) {
          setOrbState("thinking");
        } else if (
          eventName === "RunFinished" ||
          event.type === "RunFinished" ||
          eventName === "RunError" ||
          event.type === "RunError"
        ) {
          setOrbState("idle");
        }
      },
      onGeneration: (metadata: GenerationMetadata) => {
        const total =
          (metadata.prompt_tokens ?? 0) +
          (metadata.completion_tokens ?? 0) +
          (metadata.reasoning_tokens ?? 0);
        setTokens((prev) => prev + total);
        setCost((prev) => prev + calculateCostFromMetadata(metadata));
      },
    });

    return () => {
      window.clearInterval(tickInterval);
      handle.close();
    };
  }, [
    runId,
    token,
    setOrbState,
    setStatus,
    setTokens,
    setElapsed,
    setCost,
    setReconnecting,
  ]);
}
