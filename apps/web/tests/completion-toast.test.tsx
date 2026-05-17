/**
 * CompletionToast variant + a11y tests.
 *
 * Covers AC-124..132: succeeded / failed / cancelled variants, click-to-navigate,
 * manual dismiss, aria-live, dismiss button.
 */
import { Provider as JotaiProvider, createStore } from "jotai";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";

const navigateMock = vi.fn();

vi.mock("@tanstack/react-router", async () => {
  const actual = await vi.importActual<typeof import("@tanstack/react-router")>(
    "@tanstack/react-router",
  );
  return {
    ...actual,
    useNavigate: () => navigateMock,
  };
});

import { CompletionToast } from "../src/components/CompletionToast";
import { completionToastAtom, type ToastPayload } from "../src/stores/run";

function renderWithStore(toastValue: ToastPayload | null) {
  const store = createStore();
  store.set(completionToastAtom, toastValue);
  return {
    store,
    ...render(
      <JotaiProvider store={store}>
        <CompletionToast />
      </JotaiProvider>,
    ),
  };
}

beforeEach(() => {
  navigateMock.mockClear();
});

describe("CompletionToast", () => {
  it("renders nothing when toast atom is null", () => {
    renderWithStore(null);
    expect(screen.queryByTestId("completion-toast")).toBeNull();
  });

  it("renders 'Completed in <duration> · <cost>' for succeeded variant (AC-125)", () => {
    renderWithStore({
      runId: "run-A",
      workflowName: "research-pipeline",
      variant: "succeeded",
      durationMs: 12300,
      costUsd: 0.026,
    });
    const toast = screen.getByTestId("completion-toast");
    expect(toast.textContent).toContain("Completed in 12.3s");
    expect(toast.textContent).toContain("$0.026");
    expect(toast.textContent).toContain("research-pipeline");
  });

  it("renders 'Failed: <reason>' for failed variant (AC-126)", () => {
    renderWithStore({
      runId: "run-A",
      workflowName: "research-pipeline",
      variant: "failed",
      durationMs: 4200,
      costUsd: 0.003,
      errorMessage: "Researcher returned bad output",
    });
    const toast = screen.getByTestId("completion-toast");
    expect(toast.textContent).toContain("Failed: Researcher returned bad output");
  });

  it("renders 'Cancelled after <duration> · <cost>' for cancelled variant (AC-127)", () => {
    renderWithStore({
      runId: "run-A",
      workflowName: "long-task",
      variant: "cancelled",
      durationMs: 22000,
      costUsd: 0.008,
    });
    const toast = screen.getByTestId("completion-toast");
    expect(toast.textContent).toContain("Cancelled after 22.0s");
    expect(toast.textContent).toContain("$0.008");
  });

  it("uses aria-live='polite' on the root (AC-130)", () => {
    renderWithStore({
      runId: "run-A",
      workflowName: "test",
      variant: "succeeded",
      durationMs: 1000,
      costUsd: 0.001,
    });
    const toast = screen.getByTestId("completion-toast");
    expect(toast.getAttribute("aria-live")).toBe("polite");
    expect(toast.getAttribute("role")).toBe("status");
  });

  it("renders a manual dismiss button (AC-131)", () => {
    renderWithStore({
      runId: "run-A",
      workflowName: "test",
      variant: "succeeded",
      durationMs: 1000,
      costUsd: 0.001,
    });
    expect(screen.getByTestId("completion-toast-dismiss")).toBeTruthy();
  });

  it("dismisses on dismiss-button click", async () => {
    const { store } = renderWithStore({
      runId: "run-A",
      workflowName: "test",
      variant: "succeeded",
      durationMs: 1000,
      costUsd: 0.001,
    });
    expect(screen.getByTestId("completion-toast")).toBeTruthy();
    fireEvent.click(screen.getByTestId("completion-toast-dismiss"));
    await waitFor(() => {
      expect(screen.queryByTestId("completion-toast")).toBeNull();
    });
    expect(store.get(completionToastAtom)).toBeNull();
  });

  it("auto-dismisses after 6 seconds (AC-128)", () => {
    vi.useFakeTimers();
    const { store } = renderWithStore({
      runId: "run-A",
      workflowName: "test",
      variant: "succeeded",
      durationMs: 1000,
      costUsd: 0.001,
    });
    expect(store.get(completionToastAtom)).not.toBeNull();
    act(() => {
      vi.advanceTimersByTime(6500);
    });
    expect(store.get(completionToastAtom)).toBeNull();
    vi.useRealTimers();
  });

  it("clicking the toast invokes navigate to /runs/{run_id} (AC-129)", () => {
    renderWithStore({
      runId: "run-A",
      workflowName: "test",
      variant: "succeeded",
      durationMs: 1000,
      costUsd: 0.001,
    });
    fireEvent.click(screen.getByTestId("completion-toast"));
    expect(navigateMock).toHaveBeenCalledWith({
      to: "/runs/$runId",
      params: { runId: "run-A" },
    });
  });

  it("status dot variant class matches the toast variant", () => {
    const { unmount } = renderWithStore({
      runId: "run-A",
      workflowName: "test",
      variant: "succeeded",
      durationMs: 1000,
      costUsd: 0.001,
    });
    let dot = screen.getByTestId("completion-toast").querySelector(".completion-toast-dot");
    expect(dot?.className).toContain("completion-toast-dot--succeeded");
    unmount();

    renderWithStore({
      runId: "run-A",
      workflowName: "test",
      variant: "failed",
      durationMs: 1000,
      costUsd: 0.001,
    });
    dot = screen.getByTestId("completion-toast").querySelector(".completion-toast-dot");
    expect(dot?.className).toContain("completion-toast-dot--failed");
  });
});
