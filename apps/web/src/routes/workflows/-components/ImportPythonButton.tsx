/**
 * "+ Import from Python" button on the workflow list.
 *
 * Opens a native file picker filtered to `.py`. On selection, posts the
 * file to `POST /v1/workflows/import-python`, then refetches the list
 * and navigates to the new workflow's canvas (J-3 step 4).
 */
import { useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import { useRef, useState, type ChangeEvent, type ReactElement } from "react";

import { importPythonWorkflow } from "../../../lib/api";
import { useCapabilities } from "../../../providers/capabilities";

export function ImportPythonButton(): ReactElement {
  const { token } = useCapabilities();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const inputRef = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file || !token) return;
    setBusy(true);
    setError(null);
    try {
      const envelope = await importPythonWorkflow(token, file);
      await queryClient.invalidateQueries({ queryKey: ["workflows"] });
      if (envelope.warnings && envelope.warnings.length > 0) {
        const summary = envelope.warnings
          .map((w) => w.message)
          .join(" · ");
        // The toast component lands with shadcn install; until then,
        // surface the warning summary inline next to the button. The
        // canvas paints converted-edge markers from `metadata`.
        setError(`import succeeded with ${envelope.warnings.length} warning(s): ${summary}`);
      }
      navigate({
        to: "/workflows/$workflowId",
        params: { workflowId: envelope.workflow.id },
      });
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  }

  return (
    <>
      <input
        ref={inputRef}
        type="file"
        accept=".py"
        style={{ display: "none" }}
        onChange={handleChange}
        data-testid="import-python-file"
      />
      <button
        type="button"
        className="workflows-button"
        onClick={() => inputRef.current?.click()}
        disabled={busy}
        data-testid="import-python-button"
      >
        {busy ? "Importing…" : "+ Import from Python"}
      </button>
      {error ? (
        <span className="workflows-error" data-testid="import-python-error">
          {error}
        </span>
      ) : null}
    </>
  );
}
