import { useQuery } from "@tanstack/react-query";
import { Link, createFileRoute, useNavigate } from "@tanstack/react-router";
import { useState, type ReactElement } from "react";

import { listWorkflows, type Workflow, type WorkflowProvenance } from "../../lib/api";
import { PresenceOrb } from "../../components/TopBar/PresenceOrb";
import { ProvenanceBadge } from "../../components/ProvenanceBadge";
import { TopBar } from "../../components/TopBar";
import { Button, Card, TagMarkup } from "../../components/ui";
import { useCapabilities } from "../../providers/capabilities";
import { ImportPythonButton } from "./-components/ImportPythonButton";
import { useCommands } from "../../stores/useCommands";

import "./workflows.css";

export const Route = createFileRoute("/workflows/")({
  component: WorkflowsRoute,
});

type Filter = "all" | WorkflowProvenance;

const FILTERS: { id: Filter; label: string }[] = [
  { id: "all", label: "All" },
  { id: "visual_builder", label: "Visual" },
  { id: "code_import", label: "Imported" },
  { id: "meta_agent", label: "Meta-agent" },
];

function WorkflowsRoute(): ReactElement {
  const navigate = useNavigate();
  const { token } = useCapabilities();
  const [filter, setFilter] = useState<Filter>("all");

  const query = useQuery({
    queryKey: ["workflows", filter],
    queryFn: () =>
      listWorkflows(token ?? "", {
        provenance: filter === "all" ? undefined : filter,
      }),
    enabled: Boolean(token),
  });

  useCommands(
    "workflows-page",
    () => [
      {
        id: "create-workflow",
        label: "Create new workflow",
        section: "create",
        keywords: ["new", "blank"],
        run: () => navigate({ to: "/workflows/new" }),
      },
      {
        id: "go-home",
        label: "Go home",
        section: "navigate",
        run: () => navigate({ to: "/" }),
      },
    ],
    [navigate],
  );

  return (
    <div className="workflows-shell" data-testid="workflows-shell">
      <TopBar
        breadcrumb={
          <span data-testid="workflows-breadcrumb">
            <Link to="/">spren</Link>
            <span className="sep">›</span>
            <span>Workflows</span>
          </span>
        }
      />
      <PresenceOrb />
      <main className="workflows-main">
        <header className="workflows-header">
          <h1>Workflows</h1>
          <div className="workflows-actions">
            <Button
              variant="primary"
              size="md"
              data-testid="new-workflow-button"
              onClick={() => navigate({ to: "/workflows/new" })}
            >
              + New
            </Button>
            <ImportPythonButton />
          </div>
        </header>
        <nav className="workflows-filters" aria-label="Filter by provenance">
          {FILTERS.map((f) => (
            <button
              key={f.id}
              type="button"
              className={`workflows-filter${filter === f.id ? " is-active" : ""}`}
              onClick={() => setFilter(f.id)}
              data-testid={`workflows-filter-${f.id}`}
              aria-pressed={filter === f.id}
            >
              {f.label}
            </button>
          ))}
        </nav>
        {query.isError ? (
          <p className="workflows-error" data-testid="workflows-error">
            Couldn't load workflows: {(query.error as Error).message}
          </p>
        ) : query.isLoading ? (
          <p className="workflows-empty" data-testid="workflows-loading">
            Loading workflows…
          </p>
        ) : !query.data || query.data.items.length === 0 ? (
          <EmptyState />
        ) : (
          <ul className="workflows-list" data-testid="workflows-list">
            {query.data.items.map((wf) => (
              <li key={wf.id}>
                <WorkflowCard workflow={wf} />
              </li>
            ))}
          </ul>
        )}
      </main>
    </div>
  );
}

function EmptyState(): ReactElement {
  return (
    <div className="workflows-empty-state" data-testid="workflows-empty">
      <TagMarkup
        tag="workflow"
        size="sm"
        block
        attrs={[
          ["name", ""],
          ["agents", ["..."]],
          ["edges", ["..."]],
        ]}
      />
      <p>No workflows yet. Create one or import a Python file.</p>
    </div>
  );
}

function WorkflowCard({ workflow }: { workflow: Workflow }): ReactElement {
  const agentCount = Object.keys(workflow.definition.agents ?? {}).length;
  const flow = (workflow.definition.topology.nodes ?? [])
    .filter((n) => (n.node_type ?? "agent") === "agent" || n.node_type === "user")
    .slice(0, 5)
    .map((n) => n.name)
    .join(" → ");
  return (
    <Card
      as={Link}
      interactive
      to="/workflows/$workflowId"
      params={{ workflowId: workflow.id }}
      className="workflow-card"
      data-testid="workflow-card"
      data-workflow-id={workflow.id}
    >
      <div className="workflow-card-row">
        <span className="workflow-card-name" data-testid="workflow-card-name">
          {workflow.name}
        </span>
        <ProvenanceBadge provenance={workflow.provenance} />
      </div>
      <p className="workflow-card-meta">
        {agentCount} agent{agentCount === 1 ? "" : "s"}
        <span className="sep"> · </span>
        last edited {new Date(workflow.updated_at).toLocaleString()}
      </p>
      {flow ? <p className="workflow-card-flow">{flow}</p> : null}
    </Card>
  );
}
