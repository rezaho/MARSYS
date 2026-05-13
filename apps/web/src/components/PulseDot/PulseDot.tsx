/**
 * Shared pulsing-dot primitive.
 *
 * Used by RunButton (running), StatusBadge (running / cancelling), and
 * any future live indicator. Pulses on the same cadence as the Spren
 * orb's `speaking` state (~2s period, peaks at 1.0, dips to 0.4).
 *
 * Respects `prefers-reduced-motion`: degrades to a static dot via the
 * tokens.css global rule that disables animations for that media query.
 */
import type { ReactElement } from "react";

import "./PulseDot.css";

export interface PulseDotProps {
  /** CSS color string; defaults to `--peach`. */
  color?: string;
  /** Pixel size; defaults to 8. */
  size?: number;
  testId?: string;
}

export function PulseDot({ color, size, testId }: PulseDotProps = {}): ReactElement {
  const style: React.CSSProperties = {};
  if (color) style.backgroundColor = color;
  if (size) {
    style.width = size;
    style.height = size;
  }
  return (
    <span
      className="pulse-dot"
      style={style}
      data-testid={testId ?? "pulse-dot"}
      aria-hidden="true"
    />
  );
}
