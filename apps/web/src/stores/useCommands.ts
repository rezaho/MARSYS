/**
 * Hook that registers a list of commands for the lifetime of a component.
 *
 * Pass a stable `id` (route name typically) and a list factory. The hook
 * registers on mount and unregisters on unmount, so the cmdk palette
 * shows only the commands relevant to the currently-mounted surfaces.
 *
 * The dependency array is opaque — callers pass values whose change
 * should cause re-registration (e.g., loaded workflows for the "Open: X"
 * commands).
 */
import { useEffect } from "react";

import { useCommandStore, type SprenCommand } from "./commands";

export function useCommands(
  id: string,
  commandsFactory: () => SprenCommand[],
  deps: ReadonlyArray<unknown>,
): void {
  const register = useCommandStore((s) => s.register);
  const unregister = useCommandStore((s) => s.unregister);

  useEffect(() => {
    register(id, commandsFactory());
    return () => unregister(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id, register, unregister, ...deps]);
}
