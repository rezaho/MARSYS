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
 * On mount: POST an empty workflow with `provenance=visual_builder`,
 * grab the returned id, and `replace`-navigate to `/workflows/{id}`. If
 * the user opens the canvas and never saves a single node, the draft
 * sweeper deletes the row after 24h.
 */
import { useMutation } from "@tanstack/react-query";
import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useEffect, useRef, type ReactElement } from "react";

import { Spren } from "../../components/Spren";
import { createWorkflow, type Workflow, type WorkflowCreateRequest } from "../../lib/api";
import { useCapabilities } from "../../providers/capabilities";

import "./new.css";

export const Route = createFileRoute("/workflows/new")({
  component: NewWorkflowRoute,
});

const EMPTY_DEFINITION: WorkflowCreateRequest["definition"] = {
  topology: { nodes: [], edges: [], rules: [] },
  agents: {},
  execution_config: {},
};

function NewWorkflowRoute(): ReactElement {
  const navigate = useNavigate();
  const { token } = useCapabilities();
  const firedRef = useRef(false);

  const mutation = useMutation({
    mutationFn: async (): Promise<Workflow> => {
      if (!token) throw new Error("no auth token");
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
  });

  useEffect(() => {
    if (firedRef.current) return;
    if (!token) return;
    firedRef.current = true;
    mutation.mutate();
    // mutation is stable; we only want this on first ready mount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  return (
    <div className="new-workflow-shell" data-testid="new-workflow-shell">
      <Spren state="thinking" size="presence" />
      <p data-testid="new-workflow-status">
        {mutation.isError
          ? `Couldn't create workflow: ${(mutation.error as Error).message}`
          : "Setting up a new canvas…"}
      </p>
    </div>
  );
}
