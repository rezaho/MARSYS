/**
 * Unit tests for the canvas-attachments store.
 *
 * Covers default state, derived atoms, and resetCanvasAttachmentsAtom.
 */
import { createStore } from "jotai";
import { describe, expect, it } from "vitest";

import {
  canvasAttachmentsAtom,
  dragOverlayActiveAtom,
  hasInflightUploadAtom,
  resetCanvasAttachmentsAtom,
  uploadedFileIdsAtom,
  type CanvasAttachment,
} from "../src/stores/canvasAttachments";

function fixture(overrides: Partial<CanvasAttachment> = {}): CanvasAttachment {
  return {
    tempId: "tmp-1",
    fileId: "tmp-1",
    name: "a.txt",
    mimeType: "text/plain",
    sizeBytes: 1,
    state: "uploading",
    progress: 0,
    ...overrides,
  };
}

describe("canvas attachments store", () => {
  it("defaults to an empty list with no overlay", () => {
    const store = createStore();
    expect(store.get(canvasAttachmentsAtom)).toEqual([]);
    expect(store.get(dragOverlayActiveAtom)).toBe(false);
    expect(store.get(uploadedFileIdsAtom)).toEqual([]);
    expect(store.get(hasInflightUploadAtom)).toBe(false);
  });

  it("uploadedFileIdsAtom only includes uploaded entries", () => {
    const store = createStore();
    store.set(canvasAttachmentsAtom, [
      fixture({ tempId: "a", fileId: "a", state: "uploaded" }),
      fixture({ tempId: "b", fileId: "b", state: "uploading" }),
      fixture({ tempId: "c", fileId: "c-server", state: "uploaded" }),
      fixture({ tempId: "d", fileId: "d", state: "failed" }),
    ]);
    expect(store.get(uploadedFileIdsAtom)).toEqual(["a", "c-server"]);
  });

  it("hasInflightUploadAtom flips when any entry is uploading", () => {
    const store = createStore();
    store.set(canvasAttachmentsAtom, [
      fixture({ tempId: "a", state: "uploaded" }),
    ]);
    expect(store.get(hasInflightUploadAtom)).toBe(false);
    store.set(canvasAttachmentsAtom, [
      ...store.get(canvasAttachmentsAtom),
      fixture({ tempId: "b", state: "uploading" }),
    ]);
    expect(store.get(hasInflightUploadAtom)).toBe(true);
  });

  it("resetCanvasAttachmentsAtom clears everything", () => {
    const store = createStore();
    store.set(canvasAttachmentsAtom, [
      fixture({ tempId: "a", state: "uploaded" }),
    ]);
    store.set(dragOverlayActiveAtom, true);
    store.set(resetCanvasAttachmentsAtom);
    expect(store.get(canvasAttachmentsAtom)).toEqual([]);
    expect(store.get(dragOverlayActiveAtom)).toBe(false);
  });
});
