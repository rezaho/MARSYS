/**
 * Cmdk command palette mounted at the root.
 *
 * Opens on ⌘K (or Ctrl+K). Each route registers its commands via the
 * `useCommands` hook + the Zustand slice in `stores/commands.ts`.
 * Fuzzy filter is cmdk's built-in (`Command.Input` driven).
 */
import { Command } from "cmdk";
import { useEffect, type ReactElement } from "react";

import { useCommandStore, type CommandSection } from "../../stores/commands";

import "./CommandPalette.css";

const SECTION_ORDER: readonly CommandSection[] = [
  "create",
  "navigate",
  "workflows",
  "canvas",
  "help",
];

const SECTION_LABEL: Record<CommandSection, string> = {
  create: "Create",
  navigate: "Navigate",
  workflows: "Workflows",
  canvas: "Canvas",
  help: "Help",
};

export function CommandPalette(): ReactElement | null {
  const open = useCommandStore((s) => s.open);
  const setOpen = useCommandStore((s) => s.setOpen);
  const toggle = useCommandStore((s) => s.toggle);
  const registrations = useCommandStore((s) => s.registrations);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        toggle();
      }
      if (e.key === "Escape" && open) {
        setOpen(false);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, setOpen, toggle]);

  if (!open) return null;

  const commands = Array.from(registrations.values()).flat();
  const bySection = new Map<CommandSection, typeof commands>();
  for (const cmd of commands) {
    const bucket = bySection.get(cmd.section) ?? [];
    bucket.push(cmd);
    bySection.set(cmd.section, bucket);
  }

  return (
    <div
      className="cmdk-backdrop"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) setOpen(false);
      }}
      data-testid="cmdk-overlay"
    >
      <Command
        className="cmdk-root"
        label="Spren commands"
        loop
        data-testid="cmdk-root"
      >
        <Command.Input
          placeholder="Type a command or search…"
          className="cmdk-input"
          autoFocus
          data-testid="cmdk-input"
        />
        <Command.List className="cmdk-list">
          <Command.Empty className="cmdk-empty">
            No commands matched.
          </Command.Empty>
          {SECTION_ORDER.flatMap((section) => {
            const bucket = bySection.get(section);
            if (!bucket || bucket.length === 0) return [];
            return [
              <Command.Group
                key={section}
                heading={SECTION_LABEL[section]}
                className="cmdk-group"
              >
                {bucket.map((cmd) => (
                  <Command.Item
                    key={cmd.id}
                    value={`${cmd.label} ${(cmd.keywords ?? []).join(" ")}`}
                    onSelect={() => {
                      cmd.run();
                      setOpen(false);
                    }}
                    className="cmdk-item"
                    data-testid={`cmdk-item-${cmd.id}`}
                  >
                    {cmd.label}
                  </Command.Item>
                ))}
              </Command.Group>,
            ];
          })}
        </Command.List>
      </Command>
    </div>
  );
}
