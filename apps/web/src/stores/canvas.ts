/**
 * Jotai atoms for the high-frequency canvas state.
 *
 * Why Jotai (vs Zustand for commands): canvas selection / hover / drag
 * happen at 60 fps. Jotai's atom granularity means each subscriber only
 * re-renders when its specific atom changes; a Zustand selector would
 * either over-render (whole store reads) or require manual selector
 * memoization.
 *
 * The atoms hold ONLY transient client-only state: which node is
 * selected, the in-flight name edit, the lint findings cache, and the
 * dirty flag (whether the canvas has unsaved changes). The workflow
 * definition itself lives in the canvas component's React state because
 * (a) we don't want to re-render every observer when one field flips
 * and (b) the form library (React Hook Form) owns its own state on top.
 */
import { atom } from "jotai";

import type { LintFinding } from "../lib/api";

/** ID of the currently-selected canvas node, or null when none. */
export const selectedNodeIdAtom = atom<string | null>(null);

/** ID of the currently-selected canvas edge, or null when none. */
export const selectedEdgeIdAtom = atom<string | null>(null);

/** True when the canvas has unsaved edits relative to the last fetch. */
export const dirtyAtom = atom<boolean>(false);

/** Latest lint findings; empty array if lint hasn't run or returned no findings. */
export const lintFindingsAtom = atom<LintFinding[]>([]);

/** "loading" | "ok" | "warning" | "error" — drives the top-toolbar chip. */
export const lintStatusAtom = atom<"idle" | "loading" | "ok" | "warning" | "error">(
  "idle",
);

/** True when the lint panel is open. */
export const lintPanelOpenAtom = atom<boolean>(false);
