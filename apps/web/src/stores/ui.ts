/**
 * Zustand slice for ambient UI state that any component can read or
 * trigger. Right now: the sidebar open/closed state. Likely growth:
 * theme preference, chat sheet pin, toast queue.
 *
 * Kept separate from `commands.ts` because cmdk is its own dedicated
 * surface; sidebar is a sibling chrome element.
 */
import { create } from "zustand";

interface UIStore {
  sidebarOpen: boolean;
  setSidebarOpen: (next: boolean) => void;
  toggleSidebar: () => void;
}

export const useUIStore = create<UIStore>((set) => ({
  sidebarOpen: false,
  setSidebarOpen: (next) => set({ sidebarOpen: next }),
  toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),
}));
