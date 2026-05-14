/**
 * Re-run helper: copies a run's ``task_input`` (text + attachments) verbatim
 * and fires ``POST /v1/runs``.
 *
 * If a referenced file_id has been deleted between the original run and the
 * re-run click, the server returns 400 ``ATTACHMENT_NOT_FOUND``. The caller
 * (the inspector's Re-run button handler) catches this and shows the
 * "remaining attachments?" inline confirm per plan §10.11.
 */
import type { RunCreateResponse, RunRead } from "./api";
import { createRun } from "./api";

export class StaleAttachmentError extends Error {
  fileId: string;
  remaining: string[];

  constructor(message: string, fileId: string, remaining: string[]) {
    super(message);
    this.name = "StaleAttachmentError";
    this.fileId = fileId;
    this.remaining = remaining;
  }
}

export async function rerunRun(
  token: string,
  source: RunRead,
): Promise<RunCreateResponse> {
  return createRun(token, {
    workflow_id: source.workflow_id,
    task_input: {
      text: source.task_input.text ?? "",
      attachments: [...(source.task_input.attachments ?? [])],
    },
    trigger: "manual",
  });
}

export async function rerunRunWithAttachments(
  token: string,
  source: RunRead,
  attachmentFileIds: string[],
): Promise<RunCreateResponse> {
  return createRun(token, {
    workflow_id: source.workflow_id,
    task_input: {
      text: source.task_input.text ?? "",
      attachments: attachmentFileIds,
    },
    trigger: "manual",
  });
}
