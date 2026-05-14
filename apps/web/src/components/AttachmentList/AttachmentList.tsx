/**
 * Read-only attachment list rendered on ``/runs/{id}``.
 *
 * Fetches each ``file_id``'s metadata; renders Download links via the
 * authenticated ``downloadFile`` helper. When a referenced file has
 * been deleted, the row renders a "Missing" chip in ``--magenta-deep``
 * (plan §10.11 + improver F: proactive surfacing, not just on Re-run).
 */
import { useQuery } from "@tanstack/react-query";
import { useState, type ReactElement } from "react";

import { downloadFile, getFileMetadata } from "../../lib/files";
import { useCapabilities } from "../../providers/capabilities";
import { TagMarkup } from "../ui";

import "./AttachmentList.css";

export interface AttachmentListProps {
  fileIds: string[];
  testId?: string;
}

export function AttachmentList({
  fileIds,
  testId = "attachment-list",
}: AttachmentListProps): ReactElement {
  if (fileIds.length === 0) {
    return (
      <div className="attachment-list-empty" data-testid={`${testId}-empty`}>
        <TagMarkup tag="attachments" size="sm" attrs={[["status", "none"]]} />
      </div>
    );
  }
  return (
    <ul className="attachment-list" data-testid={testId}>
      {fileIds.map((fileId) => (
        <AttachmentRow key={fileId} fileId={fileId} />
      ))}
    </ul>
  );
}

function AttachmentRow({ fileId }: { fileId: string }): ReactElement {
  const { token } = useCapabilities();
  const [downloading, setDownloading] = useState(false);
  const [downloadError, setDownloadError] = useState<string | null>(null);

  const query = useQuery({
    queryKey: ["file-metadata", fileId],
    queryFn: () => getFileMetadata(token ?? "", fileId),
    enabled: Boolean(token),
    retry: false,
  });

  const handleDownload = async (): Promise<void> => {
    if (!token || !query.data) return;
    setDownloading(true);
    setDownloadError(null);
    try {
      await downloadFile(token, fileId, query.data.original_name);
    } catch (err) {
      setDownloadError(err instanceof Error ? err.message : "download failed");
    } finally {
      setDownloading(false);
    }
  };

  if (query.isError) {
    return (
      <li
        className="attachment-row attachment-row--missing"
        data-testid="attachment-row"
        data-state="missing"
        data-file-id={fileId}
      >
        <span className="attachment-row-icon" aria-hidden="true">📄</span>
        <div className="attachment-row-content">
          <div className="attachment-row-name">{fileId.slice(0, 12)}…</div>
          <div className="attachment-row-meta">
            <span className="attachment-missing-chip" data-testid="attachment-missing-chip">
              Missing
            </span>
            <span> · file no longer available</span>
          </div>
        </div>
      </li>
    );
  }

  if (query.isLoading || !query.data) {
    return (
      <li className="attachment-row attachment-row--loading" data-testid="attachment-row">
        <span className="attachment-row-icon" aria-hidden="true">📄</span>
        <div className="attachment-row-content">
          <div className="attachment-row-name">Loading…</div>
        </div>
      </li>
    );
  }

  const meta = query.data;
  return (
    <li
      className="attachment-row"
      data-testid="attachment-row"
      data-state="present"
      data-file-id={fileId}
    >
      <span className="attachment-row-icon" aria-hidden="true">{iconForMime(meta.mime_type)}</span>
      <div className="attachment-row-content">
        <div className="attachment-row-name">{meta.original_name}</div>
        <div className="attachment-row-meta">
          {formatBytes(meta.size_bytes)}
          <span className="sep"> · </span>
          {meta.mime_type}
        </div>
        {downloadError && (
          <div className="attachment-row-error" data-testid="attachment-row-error">
            {downloadError}
          </div>
        )}
      </div>
      <button
        type="button"
        className="attachment-download"
        onClick={handleDownload}
        disabled={downloading}
        data-testid="attachment-download"
      >
        {downloading ? "Downloading…" : "Download"}
      </button>
    </li>
  );
}

function iconForMime(mime: string): string {
  if (mime.startsWith("image/")) return "🖼";
  if (mime.includes("pdf")) return "📄";
  if (mime.includes("csv") || mime.includes("excel") || mime.includes("spreadsheet")) return "📊";
  if (mime.startsWith("text/")) return "📝";
  return "📁";
}

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  return `${(n / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}
