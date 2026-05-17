/**
 * Left-edge node palette: drag-from / click to add a node to the canvas.
 *
 * Category model (locked — see docs/architecture/spren/11-node-model.md):
 *   - Agents: the standard Agent (active) + a specialized catalog
 *     (Browser / Code / … — frontend authoring presets; their tool- and
 *     instruction-templating is task #21, so they are listed but inactive
 *     here).
 *   - Core: Start (the single, default, non-deletable canvas entry — it
 *     always exists, so it is shown but not droppable), End and User
 *     (both droppable, 0..N).
 *   - Logic / Tools / Data: modeled but not yet wired — shown inactive
 *     ("soon"), non-droppable.
 *
 * Session 08 ships only this minimal category-aware palette; the full
 * specialized-agent card/detail UX is task #21.
 *
 * Active items use HTML5 drag-and-drop with a `application/spren-node-type`
 * payload (the `NodeKind`) that the canvas's `onDrop` reads.
 */
import type { DragEvent, ReactElement } from "react";

import type { NodeKind } from "../../../lib/api";

import "./Palette.css";

/** A droppable Core / Agent item — produces a real `kind` node. */
const ACTIVE_AGENT: { kind: NodeKind; label: string } = { kind: "agent", label: "Agent" };
const ACTIVE_CORE: { kind: NodeKind; label: string }[] = [
  { kind: "end", label: "End" },
  { kind: "user", label: "User" },
];

/**
 * Specialized agents are authoring presets (AC-PALETTE-2): they compile to
 * a generic `kind="agent"` node + a templated `AgentSpec`. The templating
 * UX is task #21, so they are catalogued here but inactive.
 */
const SPECIALIZED_AGENTS = [
  "Browser",
  "Code",
  "DataAnalysis",
  "FileOperation",
  "WebSearch",
  "InteractiveElements",
  "Learnable",
  "Guardrail",
];

const LOGIC_ITEMS = ["if / else", "while"];

export function Palette({
  onAdd,
}: {
  onAdd: (kind: NodeKind) => void;
}): ReactElement {
  function handleDragStart(event: DragEvent<HTMLButtonElement>, kind: NodeKind) {
    event.dataTransfer.setData("application/spren-node-type", kind);
    event.dataTransfer.effectAllowed = "move";
  }

  function activeChip(kind: NodeKind, label: string): ReactElement {
    return (
      <button
        key={kind}
        type="button"
        className="canvas-palette-chip"
        draggable
        onDragStart={(e) => handleDragStart(e, kind)}
        onClick={() => onAdd(kind)}
        data-testid={`canvas-palette-${kind}`}
      >
        {label}
      </button>
    );
  }

  function inactiveChip(label: string, title: string, key: string): ReactElement {
    return (
      <button
        key={key}
        type="button"
        className="canvas-palette-chip is-inactive"
        disabled
        title={title}
        data-testid={`canvas-palette-soon-${key}`}
      >
        {label}
      </button>
    );
  }

  return (
    <div className="canvas-palette" data-testid="canvas-palette">
      <span className="canvas-palette-label">Palette</span>

      <details className="canvas-palette-group" open data-testid="canvas-palette-agents">
        <summary>Agents</summary>
        <div className="canvas-palette-group-items">
          {activeChip(ACTIVE_AGENT.kind, ACTIVE_AGENT.label)}
          {SPECIALIZED_AGENTS.map((s) =>
            inactiveChip(s, "Specialized agent presets — coming in a later session", s),
          )}
        </div>
      </details>

      <span className="canvas-palette-cat">Core</span>
      {inactiveChip("Start", "Every canvas already has its single Start node", "start")}
      {ACTIVE_CORE.map((c) => activeChip(c.kind, c.label))}

      <span className="canvas-palette-cat">Logic</span>
      {LOGIC_ITEMS.map((l) => inactiveChip(l, "Logic nodes — coming soon", l))}

      <span className="canvas-palette-cat">Tools</span>
      {inactiveChip("Tool", "Tool nodes — coming soon", "tools")}

      <span className="canvas-palette-cat">Data</span>
      {inactiveChip("Data", "Data nodes — coming soon", "data")}
    </div>
  );
}
