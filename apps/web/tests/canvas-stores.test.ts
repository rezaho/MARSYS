/**
 * Unit tests for the canvas-side state primitives.
 *
 * Covers AC-155 (the Vitest unit tests the brief calls for):
 *   - Zustand command-palette slice registration / deregistration
 *   - Jotai canvas atoms default values
 */
import { createStore } from "jotai";
import { describe, expect, it } from "vitest";

import {
  dirtyAtom,
  lintFindingsAtom,
  lintPanelOpenAtom,
  lintStatusAtom,
  selectedEdgeIdAtom,
  selectedNodeIdAtom,
} from "../src/stores/canvas";
import { useCommandStore } from "../src/stores/commands";

describe("canvas atoms", () => {
  it("default to a clean canvas state", () => {
    const store = createStore();
    expect(store.get(selectedNodeIdAtom)).toBeNull();
    expect(store.get(selectedEdgeIdAtom)).toBeNull();
    expect(store.get(dirtyAtom)).toBe(false);
    expect(store.get(lintFindingsAtom)).toEqual([]);
    expect(store.get(lintStatusAtom)).toBe("idle");
    expect(store.get(lintPanelOpenAtom)).toBe(false);
  });

  it("track edits via the dirty atom", () => {
    const store = createStore();
    store.set(dirtyAtom, true);
    expect(store.get(dirtyAtom)).toBe(true);
    store.set(dirtyAtom, false);
    expect(store.get(dirtyAtom)).toBe(false);
  });
});

describe("command-palette slice", () => {
  it("registers and lists commands by id", () => {
    const store = useCommandStore.getState();
    store.register("home", [
      {
        id: "go-workflows",
        label: "Go to Workflows",
        section: "navigate",
        run: () => undefined,
      },
    ]);
    const list = useCommandStore.getState().list();
    expect(list.map((c) => c.id)).toContain("go-workflows");
    useCommandStore.getState().unregister("home");
  });

  it("deregisters commands when the source unmounts", () => {
    useCommandStore.getState().register("canvas", [
      {
        id: "save-workflow",
        label: "Save workflow",
        section: "canvas",
        run: () => undefined,
      },
    ]);
    expect(useCommandStore.getState().list().some((c) => c.id === "save-workflow")).toBe(true);
    useCommandStore.getState().unregister("canvas");
    expect(useCommandStore.getState().list().some((c) => c.id === "save-workflow")).toBe(false);
  });

  it("toggles the overlay open state", () => {
    const store = useCommandStore.getState();
    expect(store.open).toBe(false);
    store.toggle();
    expect(useCommandStore.getState().open).toBe(true);
    store.toggle();
    expect(useCommandStore.getState().open).toBe(false);
  });

  it("ignores duplicate registrations for the same id by replacing them", () => {
    useCommandStore.getState().register("home", [
      { id: "first", label: "first", section: "navigate", run: () => undefined },
    ]);
    useCommandStore.getState().register("home", [
      { id: "second", label: "second", section: "navigate", run: () => undefined },
    ]);
    const ids = useCommandStore.getState().list().map((c) => c.id);
    expect(ids).toContain("second");
    expect(ids).not.toContain("first");
    useCommandStore.getState().unregister("home");
  });
});
