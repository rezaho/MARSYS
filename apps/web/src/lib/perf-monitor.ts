/**
 * Frame-time monitor for the Spren orb perf kill switch.
 *
 * Samples `requestAnimationFrame` deltas over a rolling 60-frame
 * window. If the median exceeds 20 ms for two consecutive windows,
 * set `data-spren-degraded="true"` on `<html>`. CSS reads that
 * attribute (in Spren.css) to progressively drop behaviors.
 *
 * Pauses on `document.hidden` so a backgrounded tab doesn't trip the
 * kill switch with throttled rAF deltas.
 *
 * Strategy: docs/implementation/spren/v0.3.0/01-visual-builder/sessions/03-visual-builder/orb-micro-interactions.md § 4
 */

const WINDOW_SIZE = 60;
const MEDIAN_BUDGET_MS = 20;
const REQUIRED_BAD_WINDOWS = 2;

let stopped = false;
let badWindows = 0;
let frames: number[] = [];
let lastFrame = 0;

function median(samples: number[]): number {
  if (samples.length === 0) return 0;
  const sorted = [...samples].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

function tick(now: number) {
  if (stopped) return;

  if (document.hidden) {
    frames = [];
    lastFrame = 0;
    requestAnimationFrame(tick);
    return;
  }

  if (lastFrame > 0) {
    frames.push(now - lastFrame);
    if (frames.length >= WINDOW_SIZE) {
      const med = median(frames);
      frames = [];
      if (med > MEDIAN_BUDGET_MS) {
        badWindows += 1;
        if (badWindows >= REQUIRED_BAD_WINDOWS) {
          document.documentElement.dataset.sprenDegraded = "true";
        }
      } else {
        badWindows = 0;
        if (document.documentElement.dataset.sprenDegraded === "true") {
          delete document.documentElement.dataset.sprenDegraded;
        }
      }
    }
  }
  lastFrame = now;
  requestAnimationFrame(tick);
}

export function startPerfMonitor(): () => void {
  if (stopped) return () => undefined;
  const onVisibility = () => {
    frames = [];
    lastFrame = 0;
  };
  document.addEventListener("visibilitychange", onVisibility);
  requestAnimationFrame(tick);
  return () => {
    stopped = true;
    document.removeEventListener("visibilitychange", onVisibility);
  };
}
