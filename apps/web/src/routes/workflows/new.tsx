/**
 * `/workflows/new` — create-on-mount → redirect to `/workflows/$id`.
 *
 * The POST fires from `useEffect`, NOT from a TanStack Router loader.
 * Loaders run on hover-prefetch; firing POST there would create ghost
 * draft rows whenever a user merely hovered the link. The empty draft
 * sweeper would clean them up after 24h, but the workflow list would
 * briefly show them (the predicate hides them, but the round-trip is
 * still wasted).
 *
 * Visible states the user can land in:
 *  - capabilities still bootstrapping → "Connecting to Spren…"
 *  - capabilities failed (no auth token, sidecar dead) → error + link home
 *  - mutation pending → "Setting up a new canvas…" + elapsed-time counter
 *  - mutation errored → error + Retry + Cancel
 *  - mutation succeeded → flash success then redirect
 *
 * Earlier versions silently sat on the pending state forever when auth
 * was missing or the POST never resolved — the user got no signal. This
 * version surfaces every state.
 */
import { useMutation } from "@tanstack/react-query";
import { Link, createFileRoute, useNavigate } from "@tanstack/react-router";
import { useEffect, useRef, useState, type ReactElement } from "react";

import { Spren } from "../../components/Spren";
import { createWorkflow, type Workflow, type WorkflowCreateRequest } from "../../lib/api";
import { useCapabilities } from "../../providers/capabilities";

import "./new.css";

export const Route = createFileRoute("/workflows/new")({
  component: NewWorkflowRoute,
});

// A fresh canvas is seeded with exactly one Start node — the single,
// non-deletable, canvas-wide entry point. It is a real persisted
// `kind="start"` node (not a shim), so a new workflow round-trips
// through the framework without the permissive missing-Start path.
const EMPTY_DEFINITION: WorkflowCreateRequest["definition"] = {
  topology: {
    nodes: [{ name: "Start", kind: "start" }],
    edges: [],
    rules: [],
  },
  agents: {},
  execution_config: {},
};

function NewWorkflowRoute(): ReactElement {
  const navigate = useNavigate();
  const { token, isLoading: capsLoading, error: capsError } = useCapabilities();
  const firedRef = useRef(false);
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const [elapsed, setElapsed] = useState(0);

  const mutation = useMutation({
    mutationFn: async (): Promise<Workflow> => {
      if (!token) throw new Error("no auth token (bootstrap returned without one)");
      return createWorkflow(token, {
        name: "Untitled workflow",
        description: null,
        definition: EMPTY_DEFINITION,
        provenance: "visual_builder",
        provenance_metadata: null,
      });
    },
    onSuccess: (workflow) => {
      navigate({
        to: "/workflows/$workflowId",
        params: { workflowId: workflow.id },
        replace: true,
      });
    },
    onError: (err) => {
      // eslint-disable-next-line no-console
      console.error("workflow create failed", err);
    },
  });

  // Fire the mutation once on the first mount where we have a token.
  useEffect(() => {
    if (firedRef.current) return;
    if (capsLoading) return;
    if (!token) return;
    firedRef.current = true;
    setStartedAt(Date.now());
    mutation.mutate();
    // mutation.mutate is stable; intentionally one-shot on token-arrival.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token, capsLoading]);

  // Elapsed-time counter so a hung pending state surfaces a number rather
  // than looking frozen. Cheap setInterval; cleared on unmount.
  useEffect(() => {
    if (!startedAt) return;
    if (!mutation.isPending) return;
    const id = window.setInterval(() => setElapsed(Date.now() - startedAt), 250);
    return () => window.clearInterval(id);
  }, [startedAt, mutation.isPending]);

  function retry() {
    firedRef.current = false;
    mutation.reset();
    setStartedAt(null);
    setElapsed(0);
    if (token) {
      firedRef.current = true;
      setStartedAt(Date.now());
      mutation.mutate();
    }
  }

  return (
    <div className="new-workflow-shell" data-testid="new-workflow-shell">
      <div className="new-workflow-orb">
        <Spren state={mutation.isError ? "idle" : "thinking"} size="presence" />
      </div>
      <NewWorkflowStatus
        capsLoading={capsLoading}
        capsError={capsError}
        token={token}
        mutation={mutation}
        elapsed={elapsed}
        onRetry={retry}
      />
    </div>
  );
}

interface StatusProps {
  capsLoading: boolean;
  capsError: Error | null;
  token: string | null;
  mutation: ReturnType<typeof useMutation<Workflow, Error, void>>;
  elapsed: number;
  onRetry: () => void;
}

function NewWorkflowStatus({
  capsLoading,
  capsError,
  token,
  mutation,
  elapsed,
  onRetry,
}: StatusProps): ReactElement {
  if (capsLoading) {
    return (
      <p className="new-workflow-msg" data-testid="new-workflow-status">
        Connecting to Spren…
      </p>
    );
  }

  if (capsError || !token) {
    return (
      <div className="new-workflow-error" data-testid="new-workflow-status">
        <p>Can't reach the Spren sidecar.</p>
        <p className="new-workflow-detail">{capsError?.message ?? "no auth token in URL or window"}</p>
        <Link to="/" className="new-workflow-link">← back home</Link>
      </div>
    );
  }

  if (mutation.isError) {
    return (
      <div className="new-workflow-error" data-testid="new-workflow-status">
        <p>Couldn't create workflow.</p>
        <p className="new-workflow-detail">{mutation.error.message}</p>
        <div className="new-workflow-actions">
          <button type="button" className="new-workflow-retry" onClick={onRetry}>
            Retry
          </button>
          <Link to="/workflows" className="new-workflow-link">Cancel</Link>
        </div>
      </div>
    );
  }

  if (mutation.isSuccess) {
    return (
      <p className="new-workflow-msg" data-testid="new-workflow-status">
        Opening canvas…
      </p>
    );
  }

  const seconds = Math.floor(elapsed / 1000);
  const slow = seconds >= 5;
  return (
    <div className="new-workflow-pending" data-testid="new-workflow-status">
      <p className="new-workflow-msg">Setting up a new canvas…</p>
      {slow ? (
        <p className="new-workflow-detail">
          Still waiting on the sidecar ({seconds}s elapsed). Network or server may be stuck.
        </p>
      ) : null}
      {slow ? (
        <div className="new-workflow-actions">
          <button type="button" className="new-workflow-retry" onClick={onRetry}>
            Retry
          </button>
          <Link to="/workflows" className="new-workflow-link">Cancel</Link>
        </div>
      ) : null}
    </div>
  );
}
