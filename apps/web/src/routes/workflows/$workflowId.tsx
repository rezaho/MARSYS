import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, createFileRoute, useNavigate } from "@tanstack/react-router";
import {
  Background,
  ConnectionMode,
  Controls,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
  type Connection,
  type Edge,
  type EdgeChange,
  type Node,
  type NodeChange,
} from "@xyflow/react";
import { useAtom, useSetAtom } from "jotai";
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type DragEvent,
  type ReactElement,
} from "react";

import { CompletionToast } from "../../components/CompletionToast";
import { FileAttachInput, uploadFilesViaDrop } from "../../components/FileAttachInput";
import { RunButton } from "../../components/RunButton";
import { PresenceOrb, TopBar } from "../../components/TopBar";
import { Button } from "../../components/ui";
import {
  canvasAttachmentsAtom,
  dragOverlayActiveAtom,
} from "../../stores/canvasAttachments";
import { orbStateAtom } from "../../stores/run";
import {
  getWorkflow,
  lintWorkflow,
  replaceWorkflow,
  type AgentSpec,
  type NodeKind,
  type WorkflowDefinition,
} from "../../lib/api";
import type { PatternResult } from "../../lib/pattern-presets";
import type { PatternInsertMode } from "./-canvas/PatternModal";
import { autoLayout } from "../../lib/auto-layout";
import {
  isStartKind,
  reactFlowToWorkflow,
  workflowToReactFlow,
} from "../../lib/canvas-serialize";
import { useCapabilities } from "../../providers/capabilities";
import {
  dirtyAtom,
  lintFindingsAtom,
  lintPanelOpenAtom,
  lintStatusAtom,
  selectedEdgeIdAtom,
  selectedNodeIdAtom,
} from "../../stores/canvas";
import { useCommands } from "../../stores/useCommands";
import { AgentConfigForm } from "./-canvas/AgentConfigForm";
import { CanvasEdge, CanvasEdgeArrow, type CanvasEdgeData } from "./-canvas/CanvasEdge";
import { CanvasNode, type CanvasNodeData } from "./-canvas/CanvasNode";
import { LintChip } from "./-canvas/LintChip";
import { Palette } from "./-canvas/Palette";
import { PatternModal } from "./-canvas/PatternModal";

import "@xyflow/react/dist/style.css";
import "./canvas.css";

export const Route = createFileRoute("/workflows/$workflowId")({
  component: CanvasRoute,
});

const NODE_TYPES = { spren: CanvasNode };
const EDGE_TYPES = { spren: CanvasEdge };

function CanvasRoute(): ReactElement {
  return (
    <ReactFlowProvider>
      <CanvasInner />
    </ReactFlowProvider>
  );
}

function CanvasInner(): ReactElement {
  const { workflowId } = Route.useParams();
  const navigate = useNavigate();
  const { token } = useCapabilities();
  const queryClient = useQueryClient();

  const flow = useReactFlow();

  const workflowQuery = useQuery({
    queryKey: ["workflow", workflowId],
    queryFn: () => getWorkflow(token ?? "", workflowId),
    enabled: Boolean(token),
    staleTime: 0,
  });

  // ---- Working state: nodes, edges, agents, name ----
  const [name, setName] = useState<string>("Untitled workflow");
  const [nodes, setNodes] = useState<Node<CanvasNodeData>[]>([]);
  const [edges, setEdges] = useState<Edge<CanvasEdgeData>[]>([]);
  const [agents, setAgents] = useState<Record<string, AgentSpec>>({});

  // Selection
  const [selectedNodeId, setSelectedNodeId] = useAtom(selectedNodeIdAtom);
  const [_selectedEdgeId, setSelectedEdgeId] = useAtom(selectedEdgeIdAtom);
  const setDirty = useSetAtom(dirtyAtom);

  // Lint
  const setLintStatus = useSetAtom(lintStatusAtom);
  const setLintFindings = useSetAtom(lintFindingsAtom);
  const setLintPanel = useSetAtom(lintPanelOpenAtom);

  // Modal
  const [patternModalOpen, setPatternModalOpen] = useState(false);

  // Workflow → canvas state, once when query resolves.
  useEffect(() => {
    if (!workflowQuery.data) return;
    const wf = workflowQuery.data;
    setName(wf.name);
    setAgents(wf.definition.agents ?? {});
    const { nodes: rfNodes, edges: rfEdges } = workflowToReactFlow(wf.definition);
    const needsLayout = rfNodes.every(
      (n) => n.position.x === 0 && n.position.y === 0,
    );
    const positioned = needsLayout ? autoLayout(rfNodes, rfEdges) : rfNodes;
    setNodes(positioned);
    setEdges(rfEdges);
    setDirty(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [workflowQuery.data?.id]);

  // ---- Save ----
  const saveMutation = useMutation({
    mutationFn: async () => {
      if (!token) throw new Error("no auth token");
      const definition: WorkflowDefinition = reactFlowToWorkflow(nodes, edges, agents);
      const payload = {
        name,
        description: workflowQuery.data?.description ?? null,
        definition,
        provenance: workflowQuery.data?.provenance ?? "visual_builder",
        provenance_metadata: workflowQuery.data?.provenance_metadata ?? null,
      };
      return replaceWorkflow(token, workflowId, payload);
    },
    onSuccess: async () => {
      setDirty(false);
      await queryClient.invalidateQueries({ queryKey: ["workflows"] });
      await queryClient.invalidateQueries({ queryKey: ["workflow", workflowId] });
    },
  });

  // ---- Lint (debounced 300ms) ----
  const lintTimerRef = useRef<number | null>(null);
  useEffect(() => {
    if (!token) return;
    if (lintTimerRef.current) window.clearTimeout(lintTimerRef.current);
    lintTimerRef.current = window.setTimeout(async () => {
      setLintStatus("loading");
      try {
        // Lint the LIVE canvas (serialized like Save does), not the
        // stored definition — fixes WF-BUG-LINT-REACTIVITY: a fix made
        // on the canvas now clears its finding immediately, no save or
        // page reload needed. Effect deps already include nodes/edges/
        // agents, so every edit re-lints what's on screen.
        const definition = reactFlowToWorkflow(nodes, edges, agents);
        const result = await lintWorkflow(token, workflowId, definition);
        setLintFindings(result.findings);
        const hasErrors = result.findings.some((f) => f.severity === "error");
        const hasWarnings = result.findings.some((f) => f.severity === "warning");
        setLintStatus(hasErrors ? "error" : hasWarnings ? "warning" : "ok");
      } catch (err) {
        // eslint-disable-next-line no-console
        console.error("lint request failed", err);
        setLintStatus("error");
      }
    }, 300);
    return () => {
      if (lintTimerRef.current) window.clearTimeout(lintTimerRef.current);
    };
    // Re-run lint when topology, agents, or the workflow id changes.
  }, [nodes, edges, agents, token, workflowId, setLintFindings, setLintStatus]);

  // ---- Cmdk commands ----
  useCommands(
    "canvas-page",
    () => [
      {
        id: "add-agent-node",
        label: "Add Agent node",
        section: "canvas",
        run: () => addNode("agent"),
      },
      {
        id: "add-user-node",
        label: "Add User node",
        section: "canvas",
        run: () => addNode("user"),
      },
      {
        id: "add-end-node",
        label: "Add End node",
        section: "canvas",
        run: () => addNode("end"),
      },
      {
        id: "insert-pattern",
        label: "Insert pattern…",
        section: "canvas",
        keywords: ["hub", "pipeline", "mesh", "hierarchical"],
        run: () => setPatternModalOpen(true),
      },
      {
        id: "run-lint",
        label: "Run lint",
        section: "canvas",
        run: () => setLintPanel(true),
      },
      {
        id: "save-workflow",
        label: "Save workflow",
        section: "canvas",
        run: () => saveMutation.mutate(),
      },
      {
        id: "go-workflows",
        label: "Go to Workflows",
        section: "navigate",
        run: () => navigate({ to: "/workflows" }),
      },
      {
        id: "go-home",
        label: "Go home",
        section: "navigate",
        run: () => navigate({ to: "/" }),
      },
    ],
    [navigate, nodes, edges, agents, name],
  );

  // ---- Node + edge change handlers ----
  const onNodesChange = useCallback((changes: NodeChange[]) => {
    setNodes((prev) => applyNodeChanges(prev, changes));
    // Selection alone shouldn't mark the workflow dirty — only structural
    // changes (position, removal, dimensions) do.
    if (changes.some((c) => c.type !== "select")) {
      setDirty(true);
    }
  }, [setDirty]);

  const onEdgesChange = useCallback((changes: EdgeChange[]) => {
    setEdges((prev) => applyEdgeChanges(prev, changes));
    if (changes.some((c) => c.type !== "select")) {
      setDirty(true);
    }
  }, [setDirty]);

  const onConnect = useCallback((connection: Connection) => {
    if (!connection.source || !connection.target) return;
    setEdges((prev) => [
      ...prev,
      {
        id: `e-${connection.source}-${connection.target}-${prev.length}`,
        source: connection.source,
        target: connection.target,
        type: "spren",
        data: { bidirectional: false } satisfies CanvasEdgeData,
      },
    ]);
    setDirty(true);
  }, [setDirty]);

  // ---- Drop from palette OR file drag ----
  const setCanvasAttachments = useSetAtom(canvasAttachmentsAtom);
  const [dragOverlay, setDragOverlay] = useAtom(dragOverlayActiveAtom);

  const onDragOver = useCallback((event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    const isFile = (event.dataTransfer.types || []).includes("Files");
    if (isFile) {
      event.dataTransfer.dropEffect = "copy";
      if (!dragOverlay) setDragOverlay(true);
    } else {
      event.dataTransfer.dropEffect = "move";
    }
  }, [dragOverlay, setDragOverlay]);

  const onDragLeave = useCallback((event: DragEvent<HTMLDivElement>) => {
    // Only clear when leaving the canvas-stage entirely, not when crossing
    // into a child element. Cast through unknown because xyflow's Node
    // type shadows the DOM Node global.
    const target = event.currentTarget as unknown as globalThis.Node;
    const related = event.relatedTarget as unknown as globalThis.Node | null;
    if (related && target.contains(related)) return;
    setDragOverlay(false);
  }, [setDragOverlay]);

  const onDrop = useCallback((event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragOverlay(false);
    const files = Array.from(event.dataTransfer.files ?? []);
    if (files.length > 0) {
      uploadFilesViaDrop(files, token, setCanvasAttachments);
      return;
    }
    const kind = event.dataTransfer.getData("application/spren-node-type") as NodeKind;
    if (!kind) return;
    const position = flow.screenToFlowPosition({ x: event.clientX, y: event.clientY });
    addNode(kind, position);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [flow, token, setCanvasAttachments, setDragOverlay]);

  function addNode(
    kind: NodeKind,
    position?: { x: number; y: number },
  ) {
    const existingNames = new Set(nodes.map((n) => n.data.name));
    let counter = 1;
    let baseName = kind === "agent" ? "Agent" : kind[0].toUpperCase() + kind.slice(1);
    let name = `${baseName} ${counter}`;
    while (existingNames.has(name)) {
      counter++;
      name = `${baseName} ${counter}`;
    }
    const id = `node-${Math.random().toString(36).slice(2, 9)}`;
    const data: CanvasNodeData = { name, kind };
    if (kind === "agent") {
      // Key the agent by its (unique) node name so it round-trips into the
      // framework's name-based bind contract (agents key === AgentSpec.name
      // === node.agent_ref). No random id scheme.
      setAgents({
        ...agents,
        [name]: {
          agent_model: { type: "api", name: "", provider: "anthropic" },
          name,
          goal: "",
          instruction: "",
          tools: [],
          memory_retention: "session",
          allowed_peers: [],
        },
      });
      data.agentRefAgentId = name;
    }
    const newNode: Node<CanvasNodeData> = {
      id,
      type: "spren",
      position: position ?? { x: 200, y: 200 },
      data,
    };
    setNodes((prev) => [...prev, newNode]);
    setSelectedNodeId(id);
    setDirty(true);
  }

  // ---- Selected agent for the right rail ----
  const selectedNode = nodes.find((n) => n.id === selectedNodeId);
  const selectedAgentEntry = useMemo<{ id: string; agent: AgentSpec } | null>(() => {
    if (!selectedNode || selectedNode.data.kind !== "agent") return null;
    const agentRef = (selectedNode.data as CanvasNodeData & { agentRefAgentId?: string }).agentRefAgentId;
    if (!agentRef) {
      // Existing imported workflow: look up by matching node.name → topology
      const fallbackId = Object.keys(agents).find((k) => agents[k].name === selectedNode.data.name);
      if (fallbackId) return { id: fallbackId, agent: agents[fallbackId] };
      return null;
    }
    const agent = agents[agentRef];
    if (!agent) return null;
    return { id: agentRef, agent };
  }, [agents, nodes, selectedNode]);

  function onAgentApply(next: AgentSpec) {
    if (!selectedAgentEntry) return;
    setAgents({ ...agents, [selectedAgentEntry.id]: next });
    setNodes((prev) =>
      prev.map((n) =>
        n.id === selectedNode?.id
          ? {
              ...n,
              data: {
                ...n.data,
                name: next.name,
                agentName: next.name,
                agentModel: next.agent_model.name,
              },
            }
          : n,
      ),
    );
    setDirty(true);
  }

  function onAgentDelete() {
    if (!selectedNode) return;
    setNodes((prev) => prev.filter((n) => n.id !== selectedNode.id));
    setEdges((prev) =>
      prev.filter(
        (e) => e.source !== selectedNode.id && e.target !== selectedNode.id,
      ),
    );
    if (selectedAgentEntry) {
      const next = { ...agents };
      delete next[selectedAgentEntry.id];
      setAgents(next);
    }
    setSelectedNodeId(null);
    setDirty(true);
  }

  function focusNodeByName(targetName: string) {
    const target = nodes.find((n) => n.data.name === targetName);
    if (!target) return;
    setSelectedNodeId(target.id);
    flow.setCenter(target.position.x + 110, target.position.y + 46, { duration: 600, zoom: 1.4 });
  }

  function insertPattern(preset: PatternResult, mode: PatternInsertMode) {
    const { nodes: newNodes, edges: newEdges, agents: newAgents } = preset;
    // "empty_canvas" and "replace" both clear the canvas before inserting.
    // The difference is intent: empty_canvas is gated on the canvas being
    // empty (sensible default); replace is destructive. The pattern modal
    // disables "empty_canvas" when the canvas has nodes.
    const replaces = mode === "replace" || mode === "empty_canvas";
    // Every preset is a complete runnable shape: it owns its single Start
    // node + the edge into its entry agent (see pattern-presets.ts). The
    // canvas only has to keep "exactly one Start" — a graph-combination
    // concern that belongs here, in the inserter.
    if (replaces) {
      const rfNodes: Node<CanvasNodeData>[] = newNodes.map((n) => {
        const kind = (n.kind ?? "agent") as NodeKind;
        return {
          id: n.name,
          type: "spren",
          position: { x: 0, y: 0 },
          deletable: !isStartKind(kind),
          data: {
            name: n.name,
            kind,
            agentRefAgentId: n.agent_ref ?? undefined,
          },
        };
      });
      const rfEdges: Edge<CanvasEdgeData>[] = newEdges.map((e, i) => ({
        id: `e-${e.source}-${e.target}-${i}`,
        source: e.source,
        target: e.target,
        type: "spren",
        data: { bidirectional: Boolean(e.bidirectional) },
      }));
      setNodes(autoLayout(rfNodes, rfEdges));
      setEdges(rfEdges);
      setAgents(newAgents);
    } else {
      // Merge into the existing canvas. It already has its single Start —
      // drop the preset's Start and retarget the preset's Start-out edges
      // onto the canvas Start so we never end up with two.
      const existingStart = nodes.find((nd) => isStartKind(nd.data.kind));
      const presetStartName = newNodes.find(
        (n) => (n.kind ?? "agent") === "start",
      )?.name;
      const mergeNodes = existingStart
        ? newNodes.filter((n) => (n.kind ?? "agent") !== "start")
        : newNodes;
      const startIndex = nodes.length;
      const rfNodes: Node<CanvasNodeData>[] = mergeNodes.map((n, i) => {
        const kind = (n.kind ?? "agent") as NodeKind;
        return {
          id: `node-${startIndex + i}-${n.name}`,
          type: "spren",
          position: { x: 200 + (i % 4) * 240, y: 200 + Math.floor(i / 4) * 140 },
          deletable: !isStartKind(kind),
          data: {
            name: n.name,
            kind,
            agentRefAgentId: n.agent_ref ?? undefined,
          },
        };
      });
      const idByName = new Map<string, string>();
      rfNodes.forEach((n) => idByName.set(n.data.name, n.id));
      if (existingStart && presetStartName) {
        idByName.set(presetStartName, existingStart.id);
      }
      const rfEdges: Edge<CanvasEdgeData>[] = newEdges.map((e, i) => ({
        id: `e-merge-${i}-${e.source}-${e.target}`,
        source: idByName.get(e.source) ?? e.source,
        target: idByName.get(e.target) ?? e.target,
        type: "spren",
        data: { bidirectional: Boolean(e.bidirectional) },
      }));
      setNodes((prev) => [...prev, ...rfNodes]);
      setEdges((prev) => [...prev, ...rfEdges]);
      setAgents({ ...agents, ...newAgents });
    }
    setPatternModalOpen(false);
    setDirty(true);
  }

  return (
    <div className="canvas-shell" data-testid="canvas-shell">
      <TopBar
        breadcrumb={
          <span>
            <Link to="/workflows">Workflows</Link>
            <span className="sep">›</span>
            <input
              className="canvas-name-input"
              value={name}
              onChange={(e) => {
                setName(e.target.value);
                setDirty(true);
              }}
              aria-label="Workflow name"
              data-testid="canvas-name-input"
            />
          </span>
        }
      />
      <CanvasPresenceOrb />
      <CanvasEdgeArrow />
      <div className="canvas-toolbar" data-testid="canvas-toolbar">
        <LintChip onGoToNode={focusNodeByName} />
        <Button
          variant="secondary"
          size="sm"
          onClick={() => setPatternModalOpen(true)}
          data-testid="canvas-toolbar-pattern"
        >
          + Pattern
        </Button>
        <Button
          variant="primary"
          size="sm"
          onClick={() => saveMutation.mutate()}
          disabled={saveMutation.isPending}
          data-testid="canvas-toolbar-save"
        >
          {saveMutation.isPending ? "Saving…" : "Save"}
        </Button>
        <FileAttachInput testId="canvas-toolbar-attach" />
        <RunButton
          workflowId={workflowId}
          workflowName={name || workflowId}
          testId="canvas-toolbar-run"
        />
        {saveMutation.isSuccess ? (
          <span className="canvas-toolbar-toast" data-testid="canvas-save-toast">
            saved
          </span>
        ) : null}
        {saveMutation.isError ? (
          <span
            className="canvas-toolbar-toast canvas-toolbar-toast--error"
            data-testid="canvas-save-error"
          >
            save failed: {(saveMutation.error as Error).message}
          </span>
        ) : null}
      </div>
      <div
        className="canvas-stage"
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        data-testid="canvas-stage"
      >
        {dragOverlay && (
          <div
            className="canvas-drag-overlay"
            data-testid="canvas-drag-overlay"
            aria-hidden="true"
          >
            Drop to attach
          </div>
        )}
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onNodeClick={(_, n) => setSelectedNodeId(n.id)}
          onEdgeClick={(_, e) => setSelectedEdgeId(e.id)}
          onPaneClick={() => {
            setSelectedNodeId(null);
            setSelectedEdgeId(null);
          }}
          nodeTypes={NODE_TYPES}
          edgeTypes={EDGE_TYPES}
          fitView={nodes.length > 0}
          connectionMode={ConnectionMode.Loose}
          defaultEdgeOptions={{ type: "spren" }}
          panOnDrag
          attributionPosition="bottom-right"
        >
          <Background gap={24} size={1} color="var(--rule)" />
          <Controls position="bottom-right" showInteractive={false} />
        </ReactFlow>
        <Palette onAdd={addNode} />
        {nodes.length === 0 ? <CanvasEmpty /> : null}
      </div>
      {selectedAgentEntry ? (
        <aside className="canvas-rail" data-testid="canvas-rail">
          <AgentConfigForm
            agentId={selectedAgentEntry.id}
            agent={selectedAgentEntry.agent}
            onApply={onAgentApply}
            onDelete={onAgentDelete}
          />
        </aside>
      ) : null}
      <PatternModal
        open={patternModalOpen}
        canvasEmpty={nodes.length === 0}
        onInsert={insertPattern}
        onClose={() => setPatternModalOpen(false)}
      />
      <CompletionToast />
    </div>
  );
}

function CanvasPresenceOrb(): ReactElement {
  const [orbState] = useAtom(orbStateAtom);
  return <PresenceOrb state={orbState} />;
}

function CanvasEmpty(): ReactElement {
  return (
    <div className="canvas-empty" data-testid="canvas-empty">
      <pre>
        {`<agent name=""
   model=""
   tools={...} />`}
      </pre>
      <p>Drag from the palette, pick a pattern, or open ⌘K.</p>
    </div>
  );
}

// ---------------- helpers ----------------

function applyNodeChanges(
  nodes: Node<CanvasNodeData>[],
  changes: NodeChange[],
): Node<CanvasNodeData>[] {
  let next = nodes;
  for (const change of changes) {
    switch (change.type) {
      case "position":
        next = next.map((n) =>
          n.id === change.id && change.position
            ? { ...n, position: change.position }
            : n,
        );
        break;
      case "remove":
        next = next.filter(
          (n) => n.id !== change.id || isStartKind(n.data.kind),
        );
        break;
      case "select":
        next = next.map((n) =>
          n.id === change.id ? { ...n, selected: change.selected } : n,
        );
        break;
      default:
        break;
    }
  }
  return next;
}

function applyEdgeChanges(
  edges: Edge<CanvasEdgeData>[],
  changes: EdgeChange[],
): Edge<CanvasEdgeData>[] {
  let next = edges;
  for (const change of changes) {
    switch (change.type) {
      case "remove":
        next = next.filter((e) => e.id !== change.id);
        break;
      case "select":
        next = next.map((e) =>
          e.id === change.id ? { ...e, selected: change.selected } : e,
        );
        break;
      default:
        break;
    }
  }
  return next;
}

// `workflowToReactFlow` / `reactFlowToWorkflow` now live in
// `../../lib/canvas-serialize` (pure, unit-tested).

