/**
 * Root-level cmdk commands — registered once at the layout level so
 * every route surfaces the global Navigate + Create commands.
 *
 * Per-route surfaces (canvas, workflows list, home) register *their own*
 * commands on mount in addition to these. The cmdk palette renders the
 * union grouped by section (Create / Navigate / Workflows / Canvas).
 */
import { useNavigate } from "@tanstack/react-router";
import { useQueryClient } from "@tanstack/react-query";
import { useRef, type ChangeEvent, type ReactElement } from "react";

import { importPythonWorkflow } from "../lib/api";
import { useCapabilities } from "../providers/capabilities";
import { useCommands } from "../stores/useCommands";

export function GlobalCommands(): ReactElement {
  const navigate = useNavigate();
  const { token } = useCapabilities();
  const queryClient = useQueryClient();
  const fileInputRef = useRef<HTMLInputElement>(null);

  async function handleImportFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file || !token) return;
    try {
      const envelope = await importPythonWorkflow(token, file);
      await queryClient.invalidateQueries({ queryKey: ["workflows"] });
      navigate({
        to: "/workflows/$workflowId",
        params: { workflowId: envelope.workflow.id },
      });
    } finally {
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  useCommands(
    "global-nav",
    () => [
      {
        id: "go-home",
        label: "Go home",
        section: "navigate",
        keywords: ["spren"],
        run: () => navigate({ to: "/" }),
      },
      {
        id: "go-workflows",
        label: "Go to Workflows",
        section: "navigate",
        keywords: ["graph", "canvas"],
        run: () => navigate({ to: "/workflows" }),
      },
      {
        id: "go-runs",
        label: "Go to Runs",
        section: "navigate",
        keywords: ["history", "traces", "executions"],
        run: () => navigate({ to: "/runs" }),
      },
      {
        id: "go-memory",
        label: "Go to Memory",
        section: "navigate",
        keywords: ["kb", "facts"],
        run: () => navigate({ to: "/", hash: "memory-coming-soon" }),
      },
      {
        id: "go-settings",
        label: "Go to Settings",
        section: "navigate",
        keywords: ["secrets", "keys", "budget"],
        run: () => navigate({ to: "/", hash: "settings-coming-soon" }),
      },
      {
        id: "create-workflow",
        label: "Create new workflow",
        section: "create",
        keywords: ["new", "blank", "canvas"],
        run: () => navigate({ to: "/workflows/new" }),
      },
      {
        id: "import-python",
        label: "Import from Python",
        section: "create",
        keywords: ["py", "marsys", "file"],
        run: () => fileInputRef.current?.click(),
      },
    ],
    [navigate],
  );

  return (
    <input
      ref={fileInputRef}
      type="file"
      accept=".py"
      style={{ display: "none" }}
      onChange={handleImportFile}
      data-testid="global-import-python-input"
      aria-hidden="true"
      tabIndex={-1}
    />
  );
}
