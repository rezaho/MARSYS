/**
 * RunButton state-machine tests.
 *
 * Covers AC-113..123: idle / submitting / running / cancelling phases,
 * inline cancel-confirm, return-to-idle.
 *
 * Uses lightweight stubs for createRun + useCapabilities to avoid
 * spinning up the sidecar; the SSE hook is tested separately.
 */
import { Provider as JotaiProvider, createStore } from "jotai";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";

vi.mock("../src/lib/api", async () => {
  const actual = await vi.importActual<typeof import("../src/lib/api")>("../src/lib/api");
  return {
    ...actual,
    createRun: vi.fn(),
    cancelRun: vi.fn(),
  };
});

vi.mock("../src/providers/capabilities", () => ({
  useCapabilities: () => ({ token: "stub-token", data: null, error: null, isLoading: false }),
}));

// useRunSse opens an EventSource against the backend; tests do not need
// the actual stream. Stub it as a no-op.
vi.mock("../src/hooks/useRunSse", () => ({
  useRunSse: () => undefined,
}));

import {
  cancelRun as mockCancelRun,
  createRun as mockCreateRun,
} from "../src/lib/api";
import { RunButton } from "../src/components/RunButton";
import {
  activeRunIdAtom,
  runStatusAtom,
} from "../src/stores/run";

beforeEach(() => {
  vi.clearAllMocks();
});

function renderWithStore(ui: React.ReactElement) {
  const store = createStore();
  return {
    store,
    ...render(<JotaiProvider store={store}>{ui}</JotaiProvider>),
  };
}

describe("RunButton", () => {
  it("renders idle state with 'Run' label by default", () => {
    renderWithStore(<RunButton workflowId="wf-1" workflowName="test" testId="rb" />);
    const idleBtn = screen.getByTestId("run-button-idle");
    expect(idleBtn.textContent).toContain("Run");
  });

  it("transitions to submitting state on click", async () => {
    (mockCreateRun as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      run_id: "run-A",
      status: "queued",
      schema_version: 1,
    });

    renderWithStore(<RunButton workflowId="wf-1" workflowName="test" />);
    fireEvent.click(screen.getByTestId("run-button-idle"));

    await waitFor(() => {
      expect(mockCreateRun).toHaveBeenCalledWith("stub-token", {
        workflow_id: "wf-1",
        task_input: { text: "", attachments: [] },
        trigger: "manual",
      });
    });
  });

  it("transitions to running state when activeRunId + status=running are set", async () => {
    const { store } = renderWithStore(
      <RunButton workflowId="wf-1" workflowName="test" />,
    );
    act(() => {
      store.set(activeRunIdAtom, "run-A");
      store.set(runStatusAtom, "running");
    });
    await waitFor(() => {
      expect(screen.getByTestId("run-button-cancel")).toBeTruthy();
    });
  });

  it("opens inline cancel confirmation on Cancel click", async () => {
    const { store } = renderWithStore(
      <RunButton workflowId="wf-1" workflowName="test" />,
    );
    act(() => {
      store.set(activeRunIdAtom, "run-A");
      store.set(runStatusAtom, "running");
    });
    fireEvent.click(screen.getByTestId("run-button-cancel"));
    await waitFor(() => {
      expect(screen.getByTestId("run-button-cancel-confirm")).toBeTruthy();
    });
    const confirm = screen.getByTestId("run-button-cancel-confirm");
    expect(confirm.textContent).toContain("Cancel run?");
    expect(confirm.textContent).toContain("Tool calls in flight will finish");
  });

  it("dismisses cancel confirmation on 'Keep running' click", async () => {
    const { store } = renderWithStore(
      <RunButton workflowId="wf-1" workflowName="test" />,
    );
    act(() => {
      store.set(activeRunIdAtom, "run-A");
      store.set(runStatusAtom, "running");
    });
    fireEvent.click(screen.getByTestId("run-button-cancel"));
    await waitFor(() => screen.getByTestId("run-button-cancel-confirm"));
    fireEvent.click(screen.getByTestId("run-button-cancel-keep"));
    await waitFor(() => {
      expect(screen.queryByTestId("run-button-cancel-confirm")).toBeNull();
    });
  });

  it("transitions to cancelling on confirmed cancel + invokes cancelRun", async () => {
    (mockCancelRun as ReturnType<typeof vi.fn>).mockResolvedValueOnce({});
    const { store } = renderWithStore(
      <RunButton workflowId="wf-1" workflowName="test" />,
    );
    act(() => {
      store.set(activeRunIdAtom, "run-A");
      store.set(runStatusAtom, "running");
    });
    fireEvent.click(screen.getByTestId("run-button-cancel"));
    await waitFor(() => screen.getByTestId("run-button-cancel-confirm"));
    fireEvent.click(screen.getByTestId("run-button-cancel-stop"));
    await waitFor(() => {
      expect(mockCancelRun).toHaveBeenCalledWith("stub-token", "run-A");
    });
    await waitFor(() => {
      expect(screen.getByTestId("run-button-cancelling")).toBeTruthy();
    });
  });

  it("transitions to cancelling phase when status atom is set to 'cancelling'", () => {
    const { store } = renderWithStore(
      <RunButton workflowId="wf-1" workflowName="test" />,
    );
    act(() => {
      store.set(activeRunIdAtom, "run-A");
      store.set(runStatusAtom, "cancelling");
    });
    const cancelling = screen.getByTestId("run-button-cancelling");
    expect(cancelling.textContent).toContain("Cancelling");
  });

});
