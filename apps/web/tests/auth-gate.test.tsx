/**
 * AuthGate tests.
 *
 * Verifies the gate shows the splash while resolving, the form when no
 * token is available, and the children when authenticated. Also verifies
 * pasting a token submits to the provider and shows an error on failure.
 */
import { StrictMode } from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../src/lib/api", async () => {
  const actual = await vi.importActual<typeof import("../src/lib/api")>("../src/lib/api");
  return { ...actual, fetchBootstrap: vi.fn() };
});

import { AuthGate } from "../src/components/AuthGate";
import { fetchBootstrap as mockFetchBootstrap } from "../src/lib/api";
import { CapabilitiesProvider } from "../src/providers/capabilities";

beforeEach(() => {
  vi.clearAllMocks();
  // Reset window state so tests start fresh.
  if (typeof window !== "undefined") {
    window.__SPREN_AUTH__ = undefined;
    window.location.hash = "";
    try {
      window.localStorage.clear();
    } catch {
      /* ignore */
    }
  }
});

const BOOTSTRAP = {
  framework: { version: "0.3.0" },
  spren: { active: false, version: "0.3.0" },
  surfaces: ["gui"],
  capabilities: {},
  endpoints: {},
  started_at: "2026-05-13T00:00:00Z",
  data_dir: "/tmp",
};

function Children(): React.ReactElement {
  return <div data-testid="protected-child">authenticated content</div>;
}

function withProvider(node: React.ReactElement): React.ReactElement {
  return <CapabilitiesProvider>{node}</CapabilitiesProvider>;
}

describe("AuthGate", () => {
  it("shows the token-entry form when no token is provided", async () => {
    render(withProvider(<AuthGate><Children /></AuthGate>));
    await waitFor(() => screen.getByTestId("auth-gate"));
    expect(screen.getByTestId("auth-gate-token-input")).toBeTruthy();
    expect(screen.queryByTestId("protected-child")).toBeNull();
  });

  it("hides the children entirely until authenticated", async () => {
    render(withProvider(<AuthGate><Children /></AuthGate>));
    await waitFor(() => screen.getByTestId("auth-gate"));
    expect(screen.queryByTestId("protected-child")).toBeNull();
  });

  it("renders children once a token from the URL fragment validates", async () => {
    (mockFetchBootstrap as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      framework: { version: "0.3.0" },
      spren: { active: false, version: "0.3.0" },
      surfaces: ["gui"],
      capabilities: {},
      endpoints: {},
      started_at: "2026-05-13T00:00:00Z",
      data_dir: "/tmp",
    });
    window.location.hash = "#token=valid-fragment-token";
    render(withProvider(<AuthGate><Children /></AuthGate>));
    await waitFor(() => screen.getByTestId("protected-child"));
    expect(mockFetchBootstrap).toHaveBeenCalledWith("valid-fragment-token");
  });

  it("submitting a valid token unlocks the app", async () => {
    (mockFetchBootstrap as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      framework: { version: "0.3.0" },
      spren: { active: false, version: "0.3.0" },
      surfaces: ["gui"],
      capabilities: {},
      endpoints: {},
      started_at: "2026-05-13T00:00:00Z",
      data_dir: "/tmp",
    });
    render(withProvider(<AuthGate><Children /></AuthGate>));
    await waitFor(() => screen.getByTestId("auth-gate-token-input"));
    fireEvent.change(screen.getByTestId("auth-gate-token-input"), {
      target: { value: "pasted-token" },
    });
    fireEvent.click(screen.getByTestId("auth-gate-submit"));
    await waitFor(() => screen.getByTestId("protected-child"));
    expect(mockFetchBootstrap).toHaveBeenCalledWith("pasted-token");
  });

  it("shows an error and stays gated when the token is rejected", async () => {
    (mockFetchBootstrap as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
      new Error("bootstrap failed: 401 Unauthorized"),
    );
    render(withProvider(<AuthGate><Children /></AuthGate>));
    await waitFor(() => screen.getByTestId("auth-gate-token-input"));
    fireEvent.change(screen.getByTestId("auth-gate-token-input"), {
      target: { value: "bad-token" },
    });
    fireEvent.click(screen.getByTestId("auth-gate-submit"));
    await waitFor(() => screen.getByTestId("auth-gate-error"));
    expect(screen.getByTestId("auth-gate-error").textContent).toContain("401");
    expect(screen.queryByTestId("protected-child")).toBeNull();
  });

  it("authenticates under React.StrictMode with a #token fragment (UI-BUG-007 regression)", async () => {
    // The real app (main.tsx) renders inside <React.StrictMode>, which
    // double-invokes effects in dev. The old code stripped the URL
    // fragment INSIDE the token read, so invoke #1 destroyed it and
    // invoke #2 found nothing → AuthGate. This reproduces that path.
    (mockFetchBootstrap as ReturnType<typeof vi.fn>).mockResolvedValue(BOOTSTRAP);
    window.location.hash = "#token=strict-frag-token";
    render(
      <StrictMode>{withProvider(<AuthGate><Children /></AuthGate>)}</StrictMode>,
    );
    await waitFor(() => screen.getByTestId("protected-child"));
    expect(mockFetchBootstrap).toHaveBeenCalledWith("strict-frag-token");
  });

  it("persists a verified token so a reload without a fragment stays authenticated (ARCH-Q-001)", async () => {
    (mockFetchBootstrap as ReturnType<typeof vi.fn>).mockResolvedValue(BOOTSTRAP);
    window.location.hash = "#token=persist-me";
    const first = render(withProvider(<AuthGate><Children /></AuthGate>));
    await waitFor(() => screen.getByTestId("protected-child"));
    first.unmount();

    // Simulate a fresh page load: fragment already stripped, in-memory
    // global gone, but localStorage survives a reload.
    window.location.hash = "";
    window.__SPREN_AUTH__ = undefined;
    render(withProvider(<AuthGate><Children /></AuthGate>));
    await waitFor(() => screen.getByTestId("protected-child"));
    expect(mockFetchBootstrap).toHaveBeenLastCalledWith("persist-me");
  });

  it("clears a stale persisted token and shows the form when it no longer validates", async () => {
    window.localStorage.setItem("spren.auth.token", "stale-token");
    (mockFetchBootstrap as ReturnType<typeof vi.fn>).mockRejectedValue(
      new Error("bootstrap failed: 401 Unauthorized"),
    );
    render(withProvider(<AuthGate><Children /></AuthGate>));
    await waitFor(() => screen.getByTestId("auth-gate-token-input"));
    expect(window.localStorage.getItem("spren.auth.token")).toBeNull();
    expect(screen.queryByTestId("protected-child")).toBeNull();
  });

  it("disables submit while empty input", async () => {
    render(withProvider(<AuthGate><Children /></AuthGate>));
    await waitFor(() => screen.getByTestId("auth-gate-submit"));
    const submit = screen.getByTestId("auth-gate-submit") as HTMLButtonElement;
    expect(submit.disabled).toBe(true);
    fireEvent.change(screen.getByTestId("auth-gate-token-input"), {
      target: { value: "anything" },
    });
    expect(submit.disabled).toBe(false);
  });
});
