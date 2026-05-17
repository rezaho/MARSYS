/**
 * `+ Pattern` modal — pick one of HUB_AND_SPOKE / PIPELINE /
 * HIERARCHICAL / MESH and insert into the canvas.
 *
 * The four presets live in `lib/pattern-presets.ts`; this component is
 * just the picker. "Insert at" controls whether the preset replaces
 * the current canvas, merges with it, or only fires on an empty canvas.
 */
import { useState, type ReactElement } from "react";

import { Dialog } from "../../../components/ui";
import {
  generatePattern,
  PATTERN_META,
  type PatternKey,
  type PatternResult,
} from "../../../lib/pattern-presets";

import "./PatternModal.css";

export type PatternInsertMode = "empty_canvas" | "replace" | "merge";

interface PatternModalProps {
  open: boolean;
  canvasEmpty: boolean;
  onInsert: (preset: PatternResult, mode: PatternInsertMode) => void;
  onClose: () => void;
}

export function PatternModal({
  open,
  canvasEmpty,
  onInsert,
  onClose,
}: PatternModalProps): ReactElement | null {
  const [selected, setSelected] = useState<PatternKey>("HUB_AND_SPOKE");
  const [count, setCount] = useState(3);
  const [mode, setMode] = useState<PatternInsertMode>(
    canvasEmpty ? "empty_canvas" : "merge",
  );

  const meta = PATTERN_META.find((p) => p.key === selected)!;
  const clamped = Math.min(meta.maxAgents, Math.max(meta.minAgents, count));

  return (
    <Dialog
      open={open}
      onClose={onClose}
      ariaLabel="Insert pattern"
      position="center"
      className="pattern-modal"
      testId="pattern-modal"
    >
      <header className="pattern-modal-header">
        <h2>Insert pattern</h2>
        <button
          type="button"
          className="pattern-modal-close"
          onClick={onClose}
          aria-label="Close"
        >
          ×
        </button>
      </header>
      <ul className="pattern-modal-options">
        {PATTERN_META.map((p) => (
          <li key={p.key}>
            <label className={`pattern-option${selected === p.key ? " is-selected" : ""}`}>
              <input
                type="radio"
                name="pattern"
                value={p.key}
                checked={selected === p.key}
                onChange={() => {
                  setSelected(p.key);
                  setCount(Math.min(Math.max(count, p.minAgents), p.maxAgents));
                }}
                data-testid={`pattern-radio-${p.key}`}
              />
              <span className="pattern-option-label">{p.label}</span>
              <span className="pattern-option-desc">{p.description}</span>
              <span className="pattern-option-use">For: {p.use}</span>
            </label>
          </li>
        ))}
      </ul>

      <div className="pattern-modal-controls">
        <label>
          <span>Number of agents</span>
          <input
            type="number"
            min={meta.minAgents}
            max={meta.maxAgents}
            value={clamped}
            onChange={(e) => setCount(Number(e.target.value))}
            data-testid="pattern-count"
          />
        </label>
        <label>
          <span>Insert at</span>
          <select
            value={mode}
            onChange={(e) => setMode(e.target.value as PatternInsertMode)}
            data-testid="pattern-mode"
          >
            <option value="empty_canvas" disabled={!canvasEmpty}>
              empty canvas
            </option>
            <option value="merge">merge with existing nodes</option>
            <option value="replace">replace canvas</option>
          </select>
        </label>
      </div>

      <footer className="pattern-modal-actions">
        <button type="button" onClick={onClose} className="pattern-modal-cancel">
          Cancel
        </button>
        <button
          type="button"
          className="pattern-modal-insert"
          onClick={() => onInsert(generatePattern(selected, clamped), mode)}
          data-testid="pattern-insert"
        >
          Insert
        </button>
      </footer>
    </Dialog>
  );
}
