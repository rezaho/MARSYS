import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { fetchBootstrap, type BootstrapResponse } from "../lib/api";

interface CapabilitiesState {
  data: BootstrapResponse | null;
  error: Error | null;
  isLoading: boolean;
}

const CapabilitiesContext = createContext<CapabilitiesState | null>(null);

function readToken(): string | null {
  if (typeof window === "undefined") return null;
  if (window.__SPREN_AUTH__) return window.__SPREN_AUTH__;
  const hash = window.location.hash;
  if (hash.startsWith("#")) {
    const params = new URLSearchParams(hash.slice(1));
    const token = params.get("token");
    if (token) {
      // Strip the fragment so the token doesn't sit in the URL bar.
      window.history.replaceState(null, "", window.location.pathname + window.location.search);
      return token;
    }
  }
  return null;
}

export function CapabilitiesProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<CapabilitiesState>({
    data: null,
    error: null,
    isLoading: true,
  });

  useEffect(() => {
    let cancelled = false;
    const token = readToken();
    if (!token) {
      setState({
        data: null,
        error: new Error("no auth token (Tauri injection or #token= fragment required)"),
        isLoading: false,
      });
      return;
    }
    fetchBootstrap(token)
      .then((data) => {
        if (!cancelled) setState({ data, error: null, isLoading: false });
      })
      .catch((error: Error) => {
        if (!cancelled) setState({ data: null, error, isLoading: false });
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return <CapabilitiesContext.Provider value={state}>{children}</CapabilitiesContext.Provider>;
}

export function useCapabilities(): CapabilitiesState {
  const ctx = useContext(CapabilitiesContext);
  if (!ctx) throw new Error("useCapabilities must be used inside CapabilitiesProvider");
  return ctx;
}
