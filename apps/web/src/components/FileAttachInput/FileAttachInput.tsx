/**
 * Canvas-side file attach affordance: 📎 icon + count badge + popout
 * with attachment list + drag-and-drop overlay.
 *
 * Uploads write to ``POST /v1/files`` (XHR with progress per plan §10.3);
 * the resulting ``file_id`` joins ``canvasAttachmentsAtom``. The Run
 * button reads ``uploadedFileIdsAtom`` for ``task_input.attachments``.
 *
 * Drag-and-drop UX (plan §10.5): the canvas-wide overlay is owned by
 * the parent (the canvas page), since it spans the full viewport;
 * this component only renders the icon + popout. The parent toggles
 * ``dragOverlayActiveAtom`` and pushes any dropped files into
 * ``handleFiles``.
 */
import { useAtom } from "jotai";
import {
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type ReactElement,
} from "react";

import { uploadFile, UploadError } from "../../lib/files";
import { useCapabilities } from "../../providers/capabilities";
import {
  canvasAttachmentsAtom,
  type CanvasAttachment,
} from "../../stores/canvasAttachments";

import "./FileAttachInput.css";

export interface FileAttachInputProps {
  /** Test id forwarded to the icon button. */
  testId?: string;
}

export function FileAttachInput({ testId = "file-attach-input" }: FileAttachInputProps): ReactElement {
  const { token } = useCapabilities();
  const [attachments, setAttachments] = useAtom(canvasAttachmentsAtom);
  const [popoutOpen, setPopoutOpen] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const popoutRef = useRef<HTMLDivElement>(null);

  // Dismiss popout on click-outside.
  useEffect(() => {
    if (!popoutOpen) return;
    const onPointerDown = (event: MouseEvent): void => {
      if (popoutRef.current && !popoutRef.current.contains(event.target as Node)) {
        setPopoutOpen(false);
      }
    };
    document.addEventListener("mousedown", onPointerDown);
    return () => document.removeEventListener("mousedown", onPointerDown);
  }, [popoutOpen]);

  const onPickFiles = (event: ChangeEvent<HTMLInputElement>): void => {
    const files = Array.from(event.target.files ?? []);
    if (files.length > 0) {
      uploadFiles(files, token, setAttachments);
      // Open the popout so the user sees progress immediately.
      setPopoutOpen(true);
    }
    // Reset the input so picking the same file again still fires onChange.
    event.target.value = "";
  };

  const removeAttachment = (id: string): void => {
    setAttachments((prev) => prev.filter((a) => a.fileId !== id && a.tempId !== id));
  };

  const retryAttachment = (id: string): void => {
    const att = attachments.find((a) => a.tempId === id || a.fileId === id);
    if (!att?.file) return;
    setAttachments((prev) =>
      prev.map((a) =>
        a === att ? { ...a, state: "uploading", progress: 0, error: undefined } : a,
      ),
    );
    uploadOne(att.file, token, setAttachments, att.tempId).catch(() => {
      // Error handled in uploadOne's failure path.
    });
  };

  const visibleCount = attachments.filter((a) => a.state === "uploaded").length;
  const showCount = attachments.length > 0 && visibleCount > 0;
  const inflight = attachments.filter((a) => a.state === "uploading").length > 0;

  return (
    <div className="file-attach-input" data-testid={testId}>
      <button
        type="button"
        className="file-attach-icon"
        onClick={() => {
          if (attachments.length === 0) {
            fileInputRef.current?.click();
          } else {
            setPopoutOpen((v) => !v);
          }
        }}
        aria-label={
          showCount ? `Attached files (${visibleCount})` : "Attach files"
        }
        data-testid="file-attach-icon"
        data-count={visibleCount}
      >
        <span aria-hidden="true">📎</span>
        {showCount && (
          <span className="file-attach-count" data-testid="file-attach-count">
            {visibleCount}
          </span>
        )}
        {inflight && (
          <span className="file-attach-inflight-dot" aria-hidden="true" />
        )}
      </button>
      <input
        ref={fileInputRef}
        type="file"
        multiple
        style={{ display: "none" }}
        onChange={onPickFiles}
        data-testid="file-attach-native-input"
      />
      {popoutOpen && (
        <div
          ref={popoutRef}
          className="file-attach-popout"
          role="region"
          aria-label="Attached files"
          data-testid="file-attach-popout"
        >
          {attachments.length === 0 ? (
            <p className="file-attach-empty">No files attached.</p>
          ) : (
            <ul className="file-attach-list">
              {attachments.map((att) => (
                <li
                  key={att.tempId}
                  className={`file-attach-row file-attach-row--${att.state}`}
                  data-testid="file-attach-row"
                  data-state={att.state}
                >
                  <div className="file-attach-row-name">{att.name}</div>
                  <div className="file-attach-row-meta">
                    {att.state === "uploading" ? (
                      <span data-testid="file-attach-row-uploading">
                        uploading… {att.progress ?? 0}%
                      </span>
                    ) : att.state === "failed" ? (
                      <button
                        type="button"
                        className="file-attach-retry"
                        onClick={() => retryAttachment(att.tempId)}
                        data-testid="file-attach-retry"
                      >
                        × upload failed; retry
                      </button>
                    ) : (
                      <span>{formatBytes(att.sizeBytes)}</span>
                    )}
                  </div>
                  <button
                    type="button"
                    className="file-attach-remove"
                    onClick={() => removeAttachment(att.tempId)}
                    aria-label={`Remove ${att.name}`}
                    data-testid="file-attach-remove"
                  >
                    ×
                  </button>
                </li>
              ))}
            </ul>
          )}
          <button
            type="button"
            className="file-attach-add-more"
            onClick={() => fileInputRef.current?.click()}
            data-testid="file-attach-add-more"
          >
            + Add file
          </button>
        </div>
      )}
    </div>
  );
}

function uploadFiles(
  files: File[],
  token: string | null,
  setAttachments: (
    update: (prev: CanvasAttachment[]) => CanvasAttachment[],
  ) => void,
): void {
  for (const file of files) {
    const tempId = `tmp-${crypto.randomUUID()}`;
    setAttachments((prev) => [
      ...prev,
      {
        tempId,
        fileId: tempId,
        name: file.name,
        mimeType: file.type || "application/octet-stream",
        sizeBytes: file.size,
        state: "uploading",
        progress: 0,
        file,
      } as CanvasAttachment,
    ]);
    void uploadOne(file, token, setAttachments, tempId);
  }
}

async function uploadOne(
  file: File,
  token: string | null,
  setAttachments: (
    update: (prev: CanvasAttachment[]) => CanvasAttachment[],
  ) => void,
  tempId: string,
): Promise<void> {
  if (!token) {
    setAttachments((prev) =>
      prev.map((a) =>
        a.tempId === tempId
          ? { ...a, state: "failed", error: "Not authenticated" }
          : a,
      ),
    );
    return;
  }
  try {
    const result = await uploadFile(token, file, {
      onProgress: ({ percent }) => {
        setAttachments((prev) =>
          prev.map((a) => (a.tempId === tempId ? { ...a, progress: percent } : a)),
        );
      },
    });
    setAttachments((prev) =>
      prev.map((a) =>
        a.tempId === tempId
          ? {
              ...a,
              state: "uploaded",
              fileId: result.file_id,
              mimeType: result.mime_type,
              sizeBytes: result.size_bytes,
              progress: 100,
            }
          : a,
      ),
    );
  } catch (err) {
    const message =
      err instanceof UploadError
        ? err.message
        : err instanceof Error
          ? err.message
          : "upload failed";
    setAttachments((prev) =>
      prev.map((a) =>
        a.tempId === tempId
          ? { ...a, state: "failed", error: message }
          : a,
      ),
    );
  }
}

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  return `${(n / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

/** Helper used by the canvas's drop handler — feeds files into the
 * upload pipeline without going through the picker. */
export function uploadFilesViaDrop(
  files: File[],
  token: string | null,
  setAttachments: (
    update: (prev: CanvasAttachment[]) => CanvasAttachment[],
  ) => void,
): void {
  uploadFiles(files, token, setAttachments);
}
