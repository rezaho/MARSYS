/**
 * Left-edge node palette: drag-from to add a new node to the canvas.
 *
 * Uses HTML5 drag-and-drop with a `data-spren-node-type` payload that
 * the canvas's `onDrop` reads to pick the new node's type.
 */
import type { DragEvent, ReactElement } from "react";

import type { NodeType } from "../../../lib/api";

import "./Palette.css";

const PALETTE_NODES: { type: NodeType; label: string }[] = [
  { type: "agent", label: "Agent" },
  { type: "user", label: "User" },
  { type: "system", label: "System" },
  { type: "tool", label: "Tool" },
];

export function Palette({
  onAdd,
}: {
  onAdd: (type: NodeType) => void;
}): ReactElement {
  function handleDragStart(event: DragEvent<HTMLButtonElement>, type: NodeType) {
    event.dataTransfer.setData("application/spren-node-type", type);
    event.dataTransfer.effectAllowed = "move";
  }

  return (
    <div className="canvas-palette" data-testid="canvas-palette">
      <span className="canvas-palette-label">Palette</span>
      {PALETTE_NODES.map((item) => (
        <button
          key={item.type}
          type="button"
          className="canvas-palette-chip"
          draggable
          onDragStart={(e) => handleDragStart(e, item.type)}
          onClick={() => onAdd(item.type)}
          data-testid={`canvas-palette-${item.type}`}
        >
          {item.label}
        </button>
      ))}
    </div>
  );
}
