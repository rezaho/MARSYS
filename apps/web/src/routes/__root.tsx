import { Outlet, createRootRoute } from "@tanstack/react-router";

import { CommandPalette } from "../components/CommandPalette";
import { GlobalCommands } from "../components/GlobalCommands";
import { Sidebar } from "../components/Sidebar";
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
        <Sidebar />
        <CommandPalette />
        <GlobalCommands />
      </CapabilitiesProvider>
    </QueryProvider>
  );
}
