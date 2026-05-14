/**
 * AttachmentList tests.
 *
 * Covers AC-391..396: empty state, per-row name + size + mime icon +
 * Download button, missing-file chip on 404 (proactive surfacing per
 * plan §10.11), download click wiring.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../src/lib/files", async () => {
  const actual = await vi.importActual<typeof import("../src/lib/files")>(
    "../src/lib/files",
  );
  return {
    ...actual,
    getFileMetadata: vi.fn(),
    downloadFile: vi.fn(),
  };
});

vi.mock("../src/providers/capabilities", () => ({
  useCapabilities: () => ({ token: "stub-token" }),
}));

import { AttachmentList } from "../src/components/AttachmentList";
import {
  downloadFile as mockDownloadFile,
  getFileMetadata as mockGetFileMetadata,
} from "../src/lib/files";

beforeEach(() => {
  vi.clearAllMocks();
});

function renderWithQuery(ui: React.ReactElement) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });
  return render(
    <QueryClientProvider client={client}>{ui}</QueryClientProvider>,
  );
}

describe("AttachmentList", () => {
  it("renders the empty state when fileIds is empty", () => {
    renderWithQuery(<AttachmentList fileIds={[]} />);
    expect(screen.getByTestId("attachment-list-empty")).toBeTruthy();
  });

  it("renders a row per file_id", async () => {
    (mockGetFileMetadata as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      schema_version: 1,
      id: "file-1",
      original_name: "report.pdf",
      mime_type: "application/pdf",
      size_bytes: 1_200_000,
      sha256: "abc",
      created_at: "2026-05-13T00:00:00Z",
    });
    renderWithQuery(<AttachmentList fileIds={["file-1"]} />);
    // Wait for the loading-state to flip to present-state.
    await waitFor(() => {
      const rows = screen.getAllByTestId("attachment-row");
      expect(rows[0].getAttribute("data-state")).toBe("present");
    });
    const row = screen.getAllByTestId("attachment-row")[0];
    expect(row.textContent).toContain("report.pdf");
    expect(row.textContent).toContain("application/pdf");
  });

  it("renders a Missing chip when the file no longer exists", async () => {
    (mockGetFileMetadata as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
      new Error("get file metadata failed: 404"),
    );
    renderWithQuery(<AttachmentList fileIds={["file-gone"]} />);
    await waitFor(() => screen.getByTestId("attachment-missing-chip"));
    const chip = screen.getByTestId("attachment-missing-chip");
    expect(chip.textContent).toContain("Missing");
  });

  it("Download button calls downloadFile with the original_name", async () => {
    (mockGetFileMetadata as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      schema_version: 1,
      id: "file-1",
      original_name: "doc.txt",
      mime_type: "text/plain",
      size_bytes: 100,
      sha256: "x",
      created_at: "2026-05-13T00:00:00Z",
    });
    (mockDownloadFile as ReturnType<typeof vi.fn>).mockResolvedValueOnce(undefined);
    renderWithQuery(<AttachmentList fileIds={["file-1"]} />);
    // Wait for the row to land in present-state (Download button is enabled).
    await waitFor(() => {
      const rows = screen.getAllByTestId("attachment-row");
      expect(rows[0].getAttribute("data-state")).toBe("present");
    });
    fireEvent.click(screen.getAllByTestId("attachment-download")[0]);
    await waitFor(() =>
      expect(mockDownloadFile).toHaveBeenCalledWith("stub-token", "file-1", "doc.txt"),
    );
  });
});
