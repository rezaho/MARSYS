import { Outlet, createRootRoute } from "@tanstack/react-router";

import { CommandPalette } from "../components/CommandPalette";
import { GlobalCommands } from "../components/GlobalCommands";
import { CapabilitiesProvider } from "../providers/capabilities";
import { QueryProvider } from "../providers/query";

export const Route = createRootRoute({
  component: RootComponent,
});

function RootComponent() {
  return (
    <QueryProvider>
      <CapabilitiesProvider>
        <Outlet />
        <CommandPalette />
        <GlobalCommands />
      </CapabilitiesProvider>
    </QueryProvider>
  );
}
