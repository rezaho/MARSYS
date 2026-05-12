/**
 * Top-toolbar lint chip + bottom-panel issues drawer.
 *
 * The chip aggregates lint findings (debounced 300ms POST to
 * `/v1/workflows/{id}/lint`); clicking the chip toggles the bottom
 * panel; each finding has a "Go to node" button that scrolls the
 * canvas to the affected node.
 */
import { useAtom, useAtomValue } from "jotai";
import type { ReactElement } from "react";

import type { LintFinding } from "../../../lib/api";
import {
  lintFindingsAtom,
  lintPanelOpenAtom,
  lintStatusAtom,
} from "../../../stores/canvas";

import "./LintChip.css";

export function LintChip({
  onGoToNode,
}: {
  onGoToNode: (nodeName: string) => void;
}): ReactElement {
  const status = useAtomValue(lintStatusAtom);
  const findings = useAtomValue(lintFindingsAtom);
  const [open, setOpen] = useAtom(lintPanelOpenAtom);

  const errors = findings.filter((f) => f.severity === "error").length;
  const warnings = findings.filter((f) => f.severity === "warning").length;

  const variant = errors > 0 ? "error" : warnings > 0 ? "warning" : "ok";
  const label =
    status === "loading"
      ? "Lint …"
      : errors > 0
        ? `Lint  ${errors} ✕`
        : warnings > 0
          ? `Lint  ${warnings} ⚠`
          : "Lint ✓";

  return (
    <>
      <button
        type="button"
        className={`lint-chip lint-chip--${variant}`}
        onClick={() => setOpen(!open)}
        data-testid="lint-chip"
        data-variant={variant}
        aria-expanded={open}
        aria-controls="lint-panel"
      >
        {label}
      </button>
      {open ? <LintPanel findings={findings} onGoToNode={onGoToNode} onClose={() => setOpen(false)} /> : null}
    </>
  );
}

function LintPanel({
  findings,
  onGoToNode,
  onClose,
}: {
  findings: LintFinding[];
  onGoToNode: (nodeName: string) => void;
  onClose: () => void;
}): ReactElement {
  return (
    <aside className="lint-panel" id="lint-panel" data-testid="lint-panel">
      <header className="lint-panel-header">
        <h2>Lint issues</h2>
        <button
          type="button"
          onClick={onClose}
          className="lint-panel-close"
          aria-label="Close"
        >
          ×
        </button>
      </header>
      {findings.length === 0 ? (
        <p className="lint-panel-empty">No issues — your workflow checks clean.</p>
      ) : (
        <ul className="lint-panel-list">
          {findings.map((f, i) => (
            <li
              key={`${f.code}-${f.node_name ?? "global"}-${i}`}
              className={`lint-panel-item lint-panel-item--${f.severity}`}
              data-testid="lint-finding"
              data-severity={f.severity}
              data-code={f.code}
            >
              <span className="lint-panel-marker">{f.severity === "error" ? "✕" : "⚠"}</span>
              <div className="lint-panel-body">
                <p>
                  {f.node_name ? <code>{f.node_name}</code> : null}
                  <span> {f.message}</span>
                </p>
                {f.suggestion ? (
                  <p className="lint-panel-suggestion">{f.suggestion}</p>
                ) : null}
                {f.node_name ? (
                  <button
                    type="button"
                    className="lint-panel-goto"
                    onClick={() => onGoToNode(f.node_name as string)}
                    data-testid="lint-goto-node"
                  >
                    Go to node →
                  </button>
                ) : null}
              </div>
            </li>
          ))}
        </ul>
      )}
    </aside>
  );
}
