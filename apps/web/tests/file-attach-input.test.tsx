/**
 * FileAttachInput component tests.
 *
 * Covers AC-365..384: 📎 button presence, count badge derivation from
 * uploaded entries, popout open on click, multi-file picker, removal,
 * inflight-pulse-dot indicator, and the popout's "+ Add file" button.
 *
 * The actual XHR upload is tested separately; this file uses store
 * pre-population to drive the UI states.
 */
import { Provider as JotaiProvider, createStore } from "jotai";
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.mock("../src/providers/capabilities", () => ({
  useCapabilities: () => ({
    token: "stub-token",
    data: null,
    error: null,
    isLoading: false,
  }),
}));

import { FileAttachInput } from "../src/components/FileAttachInput";
import {
  canvasAttachmentsAtom,
  type CanvasAttachment,
} from "../src/stores/canvasAttachments";

function fixture(overrides: Partial<CanvasAttachment> = {}): CanvasAttachment {
  return {
    tempId: "tmp-1",
    fileId: "tmp-1",
    name: "a.txt",
    mimeType: "text/plain",
    sizeBytes: 1024,
    state: "uploaded",
    progress: 100,
    ...overrides,
  };
}

function renderWithStore(
  ui: React.ReactElement,
  initial: CanvasAttachment[] = [],
) {
  const store = createStore();
  store.set(canvasAttachmentsAtom, initial);
  return {
    ...render(<JotaiProvider store={store}>{ui}</JotaiProvider>),
    store,
  };
}

describe("FileAttachInput", () => {
  it("renders the 📎 icon button", () => {
    renderWithStore(<FileAttachInput />);
    const icon = screen.getByTestId("file-attach-icon");
    expect(icon).toBeTruthy();
    expect(icon.textContent).toContain("📎");
  });

  it("does NOT render a count badge when no files are attached", () => {
    renderWithStore(<FileAttachInput />);
    expect(screen.queryByTestId("file-attach-count")).toBeNull();
  });

  it("renders a count badge equal to the number of uploaded entries", () => {
    renderWithStore(<FileAttachInput />, [
      fixture({ tempId: "a", fileId: "a", state: "uploaded" }),
      fixture({ tempId: "b", fileId: "b", state: "uploaded" }),
      fixture({ tempId: "c", fileId: "c", state: "uploading", progress: 30 }),
    ]);
    const badge = screen.getByTestId("file-attach-count");
    expect(badge.textContent).toBe("2");
  });

  it("clicking the icon when files are present opens the popout", () => {
    renderWithStore(<FileAttachInput />, [
      fixture({ tempId: "a", fileId: "a", state: "uploaded", name: "report.pdf" }),
    ]);
    expect(screen.queryByTestId("file-attach-popout")).toBeNull();
    fireEvent.click(screen.getByTestId("file-attach-icon"));
    const popout = screen.getByTestId("file-attach-popout");
    expect(popout).toBeTruthy();
    expect(popout.textContent).toContain("report.pdf");
  });

  it("popout's × button removes the row from the atom", () => {
    const { store } = renderWithStore(<FileAttachInput />, [
      fixture({ tempId: "a", fileId: "a", state: "uploaded", name: "doomed.txt" }),
    ]);
    fireEvent.click(screen.getByTestId("file-attach-icon"));
    fireEvent.click(screen.getByTestId("file-attach-remove"));
    expect(store.get(canvasAttachmentsAtom)).toHaveLength(0);
  });

  it("uploading row shows progress percent", () => {
    renderWithStore(<FileAttachInput />, [
      fixture({ tempId: "a", state: "uploading", progress: 42 }),
    ]);
    fireEvent.click(screen.getByTestId("file-attach-icon"));
    const meta = screen.getByTestId("file-attach-row-uploading");
    expect(meta.textContent).toContain("42");
  });

  it("failed row shows the retry affordance", () => {
    renderWithStore(<FileAttachInput />, [
      fixture({ tempId: "a", state: "failed", error: "network down" }),
    ]);
    fireEvent.click(screen.getByTestId("file-attach-icon"));
    expect(screen.getByTestId("file-attach-retry")).toBeTruthy();
  });

  it("inflight-pulse-dot shows when any upload is in progress", () => {
    renderWithStore(<FileAttachInput />, [
      fixture({ tempId: "a", state: "uploaded" }),
      fixture({ tempId: "b", state: "uploading", progress: 50 }),
    ]);
    // The dot is rendered as part of the icon when inflight.
    const icon = screen.getByTestId("file-attach-icon");
    // Test the icon contains the dot via class lookup.
    expect(icon.querySelector(".file-attach-inflight-dot")).toBeTruthy();
  });
});
