/**
 * Jotai atoms for the in-flight run state.
 *
 * Drives the canvas Run button + presence orb + completion toast from a
 * single source. The SSE consumer hook (`useRunSse`) writes; the
 * components read.
 *
 * One run is "active" at a time on the canvas (Session 04 demo gate);
 * the activeRunIdAtom keys all the others. When activeRunIdAtom is null,
 * the canvas is idle.
 */
import { atom } from "jotai";

import type { RunStatus } from "../lib/api";

/** Spren orb visual state. */
export type OrbState = "idle" | "thinking" | "speaking";

/** Terminal toast variant — for the completion toast component. */
export type ToastVariant = "succeeded" | "failed" | "cancelled";

export interface ToastPayload {
  runId: string;
  workflowName: string;
  variant: ToastVariant;
  durationMs: number | null;
  costUsd: number;
  errorMessage?: string | null;
}

export const activeRunIdAtom = atom<string | null>(null);
export const runStatusAtom = atom<RunStatus | null>(null);
export const orbStateAtom = atom<OrbState>("idle");
export const tokenCountAtom = atom<number>(0);
export const elapsedMsAtom = atom<number>(0);
export const totalCostAtom = atom<number>(0);
export const reconnectingAtom = atom<boolean>(false);
export const completionToastAtom = atom<ToastPayload | null>(null);

/** Reset every run-tracking atom to its idle default. Used on completion + on cancel. */
export const resetRunAtom = atom(null, (_get, set) => {
  set(activeRunIdAtom, null);
  set(runStatusAtom, null);
  set(orbStateAtom, "idle");
  set(tokenCountAtom, 0);
  set(elapsedMsAtom, 0);
  set(totalCostAtom, 0);
  set(reconnectingAtom, false);
});
