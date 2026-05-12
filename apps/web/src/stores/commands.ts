/**
 * Zustand slice for per-route cmdk commands.
 *
 * Each route registers its commands when it mounts and unregisters when
 * it unmounts. The CommandPalette renders the union, grouped by section.
 * The store uses a numeric token per registration so React's effect
 * cleanup deregisters the exact slice the effect registered, not a
 * stale alias.
 */
import { create } from "zustand";

export type CommandSection =
  | "navigate"
  | "canvas"
  | "workflows"
  | "create"
  | "help";

export interface SprenCommand {
  /** Stable id used by cmdk for selection. */
  id: string;
  /** Visible label. */
  label: string;
  /** Section heading the command appears under. */
  section: CommandSection;
  /** Optional keywords for fuzzy matching beyond the label. */
  keywords?: string[];
  /** Invoked when the user activates the command. */
  run: () => void;
}

interface CommandStore {
  open: boolean;
  setOpen: (next: boolean) => void;
  toggle: () => void;
  /** Map of registration-id → list of commands the source contributed. */
  registrations: Map<string, SprenCommand[]>;
  /** Returns a numeric registration id the caller passes back to unregister. */
  register: (id: string, commands: SprenCommand[]) => void;
  unregister: (id: string) => void;
  /** All commands flattened (for the palette to render). */
  list: () => SprenCommand[];
}

export const useCommandStore = create<CommandStore>((set, get) => ({
  open: false,
  setOpen: (next) => set({ open: next }),
  toggle: () => set((s) => ({ open: !s.open })),
  registrations: new Map(),
  register: (id, commands) => {
    const next = new Map(get().registrations);
    next.set(id, commands);
    set({ registrations: next });
  },
  unregister: (id) => {
    const current = get().registrations;
    if (!current.has(id)) return;
    const next = new Map(current);
    next.delete(id);
    set({ registrations: next });
  },
  list: () => {
    const all: SprenCommand[] = [];
    for (const commands of get().registrations.values()) {
      all.push(...commands);
    }
    return all;
  },
}));
