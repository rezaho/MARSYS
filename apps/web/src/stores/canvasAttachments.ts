/**
 * Jotai atoms for the canvas-side file-attachment state.
 *
 * The canvas's `📎` icon button + drag-and-drop overlay write to this
 * atom; the Run button reads it for ``task_input.attachments`` when
 * firing ``POST /v1/runs``. On run completion (success or cancel), the
 * canvas resets the atom via ``resetCanvasAttachmentsAtom``.
 *
 * One canvas-attachment list at a time (mirrors Bundle 02's "single
 * active run on canvas" model); cross-canvas attachments are not in
 * v0.3 scope.
 */
import { atom } from "jotai";

export interface CanvasAttachment {
  /** Stable client-side id used as React key + retry handle. Survives
   *  the upload state transition; matches ``fileId`` once uploaded. */
  tempId: string;
  /** ULID returned by ``POST /v1/files``. Equal to ``tempId`` while
   *  uploading; populated with the server-side id on success. */
  fileId: string;
  /** User-uploaded filename (verbatim, not sanitized). */
  name: string;
  /** Detected mime type from the upload response. */
  mimeType: string;
  /** Size in bytes. */
  sizeBytes: number;
  /** Per-file upload state. ``"uploaded"`` is the only state the run
   *  button cares about; ``"uploading"`` rows still show in the popout
   *  but disable Run; ``"failed"`` rows show retry. */
  state: "uploading" | "uploaded" | "failed";
  /** Upload progress percentage (0-100); set during ``"uploading"``. */
  progress?: number;
  /** Failure message; set when ``state === "failed"``. */
  error?: string;
  /** Reference to the original ``File`` for retry. Cleared after
   *  successful upload to release memory. */
  file?: File;
}

export const canvasAttachmentsAtom = atom<CanvasAttachment[]>([]);

/** Read-only derived atom: file_ids of fully-uploaded attachments. */
export const uploadedFileIdsAtom = atom((get) =>
  get(canvasAttachmentsAtom)
    .filter((a) => a.state === "uploaded")
    .map((a) => a.fileId),
);

/** Read-only: true if any attachment is mid-upload. Used to disable Run. */
export const hasInflightUploadAtom = atom((get) =>
  get(canvasAttachmentsAtom).some((a) => a.state === "uploading"),
);

/** Drag-overlay visibility (translucent overlay across the canvas while a
 *  file is being dragged over). */
export const dragOverlayActiveAtom = atom<boolean>(false);

/** Reset the canvas attachment list. Called on run completion. */
export const resetCanvasAttachmentsAtom = atom(null, (_get, set) => {
  set(canvasAttachmentsAtom, []);
  set(dragOverlayActiveAtom, false);
});
