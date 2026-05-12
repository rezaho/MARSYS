/**
 * Provenance badge for the workflow list cards.
 *
 * Colors are pinned in tokens.css; the badge component just chooses
 * which token to apply based on the workflow's provenance value.
 */
import type { ReactElement } from "react";

import type { WorkflowProvenance } from "../../lib/api";

import "./ProvenanceBadge.css";

const LABEL: Record<WorkflowProvenance, string> = {
  visual_builder: "visual_builder",
  meta_agent: "meta_agent",
  code_import: "code_import",
  template: "template",
  api: "api",
};

export function ProvenanceBadge({
  provenance,
  testId,
}: {
  provenance: WorkflowProvenance;
  testId?: string;
}): ReactElement {
  return (
    <span
      className={`provenance-badge provenance-badge--${provenance}`}
      data-testid={testId ?? "provenance-badge"}
      data-provenance={provenance}
    >
      &lt;provenance:{LABEL[provenance]}/&gt;
    </span>
  );
}
