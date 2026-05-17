/**
 * Unit tests for the run-state jotai atoms.
 *
 * Covers the run store defaults + the resetRunAtom write atom.
 */
import { createStore } from "jotai";
import { describe, expect, it } from "vitest";

import {
  activeRunIdAtom,
  completionToastAtom,
  elapsedMsAtom,
  orbStateAtom,
  reconnectingAtom,
  resetRunAtom,
  runStatusAtom,
  tokenCountAtom,
  totalCostAtom,
} from "../src/stores/run";

describe("run atoms", () => {
  it("default to an idle run state", () => {
    const store = createStore();
    expect(store.get(activeRunIdAtom)).toBeNull();
    expect(store.get(runStatusAtom)).toBeNull();
    expect(store.get(orbStateAtom)).toBe("idle");
    expect(store.get(tokenCountAtom)).toBe(0);
    expect(store.get(elapsedMsAtom)).toBe(0);
    expect(store.get(totalCostAtom)).toBe(0);
    expect(store.get(reconnectingAtom)).toBe(false);
    expect(store.get(completionToastAtom)).toBeNull();
  });

  it("resetRunAtom returns every atom to its default", () => {
    const store = createStore();
    store.set(activeRunIdAtom, "run-1");
    store.set(runStatusAtom, "running");
    store.set(orbStateAtom, "speaking");
    store.set(tokenCountAtom, 250);
    store.set(elapsedMsAtom, 4200);
    store.set(totalCostAtom, 0.025);
    store.set(reconnectingAtom, true);

    store.set(resetRunAtom);

    expect(store.get(activeRunIdAtom)).toBeNull();
    expect(store.get(runStatusAtom)).toBeNull();
    expect(store.get(orbStateAtom)).toBe("idle");
    expect(store.get(tokenCountAtom)).toBe(0);
    expect(store.get(elapsedMsAtom)).toBe(0);
    expect(store.get(totalCostAtom)).toBe(0);
    expect(store.get(reconnectingAtom)).toBe(false);
  });
});
