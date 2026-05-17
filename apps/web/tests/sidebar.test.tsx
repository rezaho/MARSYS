/**
 * Sidebar tests — verify the trigger toggles store state, the panel
 * mounts only when open, and the dismiss handlers work.
 *
 * SidebarTrigger has no router dependency so we render it standalone.
 * The Sidebar itself uses TanStack Router's Link, so we mock it to a
 * plain anchor — the test cares about store + DOM behavior, not about
 * route resolution.
 */
import { fireEvent, render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.mock("@tanstack/react-router", async () => {
  return {
    Link: ({ children, ...rest }: { children: React.ReactNode } & Record<string, unknown>) => (
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      <a {...(rest as any)}>{children}</a>
    ),
  };
});

// Import AFTER the mock so the Sidebar picks up the mocked Link.
const { Sidebar, SidebarTrigger } = await import("../src/components/Sidebar");
const { useUIStore } = await import("../src/stores/ui");


describe("Sidebar", () => {
  it("trigger toggles the store state", () => {
    useUIStore.setState({ sidebarOpen: false });
    const { getByTestId } = render(<SidebarTrigger />);
    expect(useUIStore.getState().sidebarOpen).toBe(false);
    fireEvent.click(getByTestId("sidebar-trigger"));
    expect(useUIStore.getState().sidebarOpen).toBe(true);
    fireEvent.click(getByTestId("sidebar-trigger"));
    expect(useUIStore.getState().sidebarOpen).toBe(false);
  });

  it("panel does not render when sidebarOpen is false", () => {
    useUIStore.setState({ sidebarOpen: false });
    const { queryByTestId } = render(<Sidebar />);
    expect(queryByTestId("sidebar")).toBeNull();
  });

  it("panel renders when sidebarOpen is true", () => {
    useUIStore.setState({ sidebarOpen: true });
    const { queryByTestId } = render(<Sidebar />);
    expect(queryByTestId("sidebar")).not.toBeNull();
    useUIStore.setState({ sidebarOpen: false });
  });

  it("clicking the backdrop closes the sidebar", () => {
    useUIStore.setState({ sidebarOpen: true });
    const { getByTestId } = render(<Sidebar />);
    fireEvent.mouseDown(getByTestId("sidebar-backdrop"));
    expect(useUIStore.getState().sidebarOpen).toBe(false);
  });

  it("aria-expanded on the trigger reflects open state", () => {
    useUIStore.setState({ sidebarOpen: false });
    const { getByTestId, rerender } = render(<SidebarTrigger />);
    expect(getByTestId("sidebar-trigger").getAttribute("aria-expanded")).toBe("false");
    useUIStore.setState({ sidebarOpen: true });
    rerender(<SidebarTrigger />);
    expect(getByTestId("sidebar-trigger").getAttribute("aria-expanded")).toBe("true");
    useUIStore.setState({ sidebarOpen: false });
  });
});
