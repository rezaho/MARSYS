/**
 * Run status badge.
 *
 * Six values: queued / running / cancelling / succeeded / failed / cancelled.
 * Tag-markup typographic device matches the ProvenanceBadge family.
 * `running` and `cancelling` show a leading PulseDot via shared primitive.
 */
import type { ReactElement } from "react";

import type { RunStatus } from "../../lib/api";
import { PulseDot } from "../PulseDot";

import "./StatusBadge.css";

const LABEL: Record<RunStatus, string> = {
  queued: "queued",
  running: "running",
  cancelling: "cancelling",
  succeeded: "succeeded",
  failed: "failed",
  cancelled: "cancelled",
};

export function StatusBadge({
  status,
  testId,
}: {
  status: RunStatus;
  testId?: string;
}): ReactElement {
  const showPulse = status === "running" || status === "cancelling";
  return (
    <span
      className={`status-badge status-badge--${status}`}
      data-testid={testId ?? "status-badge"}
      data-status={status}
    >
      {showPulse ? <PulseDot /> : null}
      &lt;status:{LABEL[status]}/&gt;
    </span>
  );
}
