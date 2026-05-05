import { Outlet, createRootRoute } from "@tanstack/react-router";
import { CapabilitiesProvider } from "../providers/capabilities";

export const Route = createRootRoute({
  component: RootComponent,
});

function RootComponent() {
  return (
    <CapabilitiesProvider>
      <Outlet />
    </CapabilitiesProvider>
  );
}
