/**
 * /runs/$runId placeholder component tests.
 *
 * Covers AC-110, 111, 112: status badge, every metadata field, the
 * trace-viewer-coming-soon empty state, and the failed-run error row.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";

vi.mock("../src/lib/api", async () => {
  const actual = await vi.importActual<typeof import("../src/lib/api")>("../src/lib/api");
  return {
    ...actual,
    getRun: vi.fn(),
  };
});

vi.mock("../src/providers/capabilities", () => ({
  useCapabilities: () => ({ token: "stub-token", data: null, error: null, isLoading: false }),
}));

// TopBar + PresenceOrb pull in TanStack Router internals through Link/useNavigate;
// stub them to keep the test self-contained.
vi.mock("../src/components/TopBar", () => ({
  TopBar: () => <div data-testid="topbar-stub" />,
  PresenceOrb: () => <div data-testid="presence-orb-stub" />,
}));
vi.mock("../src/components/TopBar/PresenceOrb", () => ({
  PresenceOrb: () => <div data-testid="presence-orb-stub" />,
}));
vi.mock("@tanstack/react-router", async () => {
  const actual = await vi.importActual<typeof import("@tanstack/react-router")>(
    "@tanstack/react-router",
  );
  return {
    ...actual,
    Link: ({ children, ...rest }: { children: React.ReactNode } & Record<string, unknown>) => (
      <a data-testid="link-stub" {...(rest as Record<string, unknown>)}>{children}</a>
    ),
    createFileRoute: () => () => ({ component: () => null }),
  };
});

import { getRun as mockGetRun } from "../src/lib/api";
import { RunDetailView } from "../src/routes/runs/$runId";

beforeEach(() => {
  vi.clearAllMocks();
});

function renderWithQueryClient(ui: React.ReactElement) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });
  return render(
    <QueryClientProvider client={queryClient}>{ui}</QueryClientProvider>,
  );
}

describe("RunDetailView", () => {
  it("renders the loading state initially (AC-110)", () => {
    (mockGetRun as ReturnType<typeof vi.fn>).mockImplementation(() => new Promise(() => {}));
    renderWithQueryClient(<RunDetailView runId="01J9X4ABCDEFGHJKMP" />);
    expect(screen.getByTestId("run-detail-shell")).toBeTruthy();
    expect(screen.getByTestId("run-detail-loading")).toBeTruthy();
  });

  it("renders the status badge + workflow link + every metadata field for a succeeded run (AC-111)", async () => {
    (mockGetRun as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      schema_version: 1,
      id: "01J9X4ABCDEFGHJKMP",
      workflow_id: "wf-research-pipeline",
      status: "succeeded",
      task_input: { text: "", attachments: [] },
      trigger: "manual",
      started_at: "2026-05-13T14:32:08+00:00",
      finished_at: "2026-05-13T14:32:20+00:00",
      total_steps: 6,
      total_duration_ms: 12300,
      total_tokens_input: 8432,
      total_tokens_output: 1221,
      total_cost_usd: 0.026,
      final_response: null,
      error: null,
      created_at: "2026-05-13T14:32:08+00:00",
      updated_at: "2026-05-13T14:32:20+00:00",
    });

    renderWithQueryClient(<RunDetailView runId="01J9X4ABCDEFGHJKMP" />);

    await waitFor(() => screen.getByTestId("status-badge"));
    const badge = screen.getByTestId("status-badge");
    expect(badge.getAttribute("data-status")).toBe("succeeded");

    // Workflow link
    expect(screen.getAllByText("wf-research-pipeline").length).toBeGreaterThan(0);

    // Metadata
    const main = screen.getByTestId("run-detail-shell");
    expect(main.textContent).toContain("Duration");
    expect(main.textContent).toContain("12.3s");
    expect(main.textContent).toContain("Cost");
    expect(main.textContent).toContain("$0.026");
    expect(main.textContent).toContain("Tokens");
    expect(main.textContent).toContain("8,432 in");
    expect(main.textContent).toContain("1,221 out");
    expect(main.textContent).toContain("Started");
    expect(main.textContent).toContain("Finished");
  });

  it("renders the Trace section header for a succeeded run (Session 05)", async () => {
    (mockGetRun as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      schema_version: 1,
      id: "01J9X4ABCDEFGHJKMP",
      workflow_id: "wf-1",
      status: "succeeded",
      task_input: { text: "", attachments: [] },
      trigger: "manual",
      started_at: null,
      finished_at: null,
      total_steps: null,
      total_duration_ms: null,
      total_tokens_input: 0,
      total_tokens_output: 0,
      total_cost_usd: 0.0,
      final_response: null,
      error: null,
      created_at: "2026-05-13T14:32:08+00:00",
      updated_at: "2026-05-13T14:32:08+00:00",
    });

    renderWithQueryClient(<RunDetailView runId="01J9X4ABCDEFGHJKMP" />);

    await waitFor(() => screen.getByTestId("status-badge"));
    const main = screen.getByTestId("run-detail-shell");
    // Session 05 inspector renders a Trace section heading. The trace
    // body itself either renders the tree (when fetched) or a loading /
    // waiting placeholder; in this stub the trace fetch is unmocked so
    // we just assert the section header is present.
    expect(main.textContent).toContain("Trace");
  });

  it("renders the error row for a failed run (AC-187)", async () => {
    (mockGetRun as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      schema_version: 1,
      id: "01J9X4ABCDEFGHJKMP",
      workflow_id: "wf-1",
      status: "failed",
      task_input: { text: "", attachments: [] },
      trigger: "manual",
      started_at: "2026-05-13T14:32:08+00:00",
      finished_at: "2026-05-13T14:32:12+00:00",
      total_steps: 2,
      total_duration_ms: 4200,
      total_tokens_input: 100,
      total_tokens_output: 50,
      total_cost_usd: 0.003,
      final_response: null,
      error: "RuntimeError: synthetic failure",
      created_at: "2026-05-13T14:32:08+00:00",
      updated_at: "2026-05-13T14:32:12+00:00",
    });

    renderWithQueryClient(<RunDetailView runId="01J9X4ABCDEFGHJKMP" />);

    await waitFor(() => screen.getByTestId("status-badge"));
    const main = screen.getByTestId("run-detail-shell");
    expect(main.textContent).toContain("Error");
    expect(main.textContent).toContain("RuntimeError: synthetic failure");
  });

  it("falls back to '—' for missing duration/started_at/finished_at", async () => {
    (mockGetRun as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      schema_version: 1,
      id: "01J9X4ABCDEFGHJKMP",
      workflow_id: "wf-1",
      status: "queued",
      task_input: { text: "", attachments: [] },
      trigger: "manual",
      started_at: null,
      finished_at: null,
      total_steps: null,
      total_duration_ms: null,
      total_tokens_input: 0,
      total_tokens_output: 0,
      total_cost_usd: 0.0,
      final_response: null,
      error: null,
      created_at: "2026-05-13T14:32:08+00:00",
      updated_at: "2026-05-13T14:32:08+00:00",
    });

    renderWithQueryClient(<RunDetailView runId="01J9X4ABCDEFGHJKMP" />);

    await waitFor(() => screen.getByTestId("status-badge"));
    const main = screen.getByTestId("run-detail-shell");
    expect(main.textContent).toContain("—");
  });

  it("renders an error message when the API call fails", async () => {
    (mockGetRun as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
      new Error("network down"),
    );
    renderWithQueryClient(<RunDetailView runId="01J9X4ABCDEFGHJKMP" />);
    await waitFor(() => screen.getByTestId("run-detail-error"));
    const err = screen.getByTestId("run-detail-error");
    expect(err.textContent).toContain("Couldn't load run");
    expect(err.textContent).toContain("network down");
  });
});
