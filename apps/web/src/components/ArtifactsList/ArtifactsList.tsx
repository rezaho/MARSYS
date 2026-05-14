/**
 * Read-only artifacts list rendered on ``/runs/{id}``.
 *
 * v0.3 ships no in-tree framework tool that produces artifacts, so the
 * common case is empty. Per plan §10.10, the entire accordion is hidden
 * by the parent route when ``items.length === 0``; this component
 * still renders an empty state when invoked with no items, for
 * defensive rendering.
 */
import { useState, type ReactElement } from "react";

import type { ArtifactInfo } from "../../lib/api";
import { downloadArtifact } from "../../lib/run-trace";
import { useCapabilities } from "../../providers/capabilities";
import { TagMarkup } from "../ui";

import "./ArtifactsList.css";

export interface ArtifactsListProps {
  runId: string;
  items: ArtifactInfo[];
  testId?: string;
}

export function ArtifactsList({
  runId,
  items,
  testId = "artifacts-list",
}: ArtifactsListProps): ReactElement {
  if (items.length === 0) {
    return (
      <div className="artifacts-list-empty" data-testid={`${testId}-empty`}>
        <TagMarkup tag="artifacts" size="sm" attrs={[["status", "none"]]} />
      </div>
    );
  }
  return (
    <ul className="artifacts-list" data-testid={testId}>
      {items.map((artifact) => (
        <ArtifactRow key={artifact.name} runId={runId} artifact={artifact} />
      ))}
    </ul>
  );
}

function ArtifactRow({ runId, artifact }: { runId: string; artifact: ArtifactInfo }): ReactElement {
  const { token } = useCapabilities();
  const [downloading, setDownloading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleDownload = async (): Promise<void> => {
    if (!token) return;
    setDownloading(true);
    setError(null);
    try {
      await downloadArtifact(token, runId, artifact.name);
    } catch (err) {
      setError(err instanceof Error ? err.message : "download failed");
    } finally {
      setDownloading(false);
    }
  };

  return (
    <li className="artifact-row" data-testid="artifact-row">
      <span className="artifact-row-icon" aria-hidden="true">{iconForMime(artifact.mime_type)}</span>
      <div className="artifact-row-content">
        <div className="artifact-row-name">{artifact.name}</div>
        <div className="artifact-row-meta">
          {formatBytes(artifact.size_bytes)}
          <span className="sep"> · </span>
          {artifact.mime_type}
        </div>
        {error && (
          <div className="artifact-row-error" data-testid="artifact-row-error">
            {error}
          </div>
        )}
      </div>
      <button
        type="button"
        className="artifact-download"
        onClick={handleDownload}
        disabled={downloading}
        data-testid="artifact-download"
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
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}
