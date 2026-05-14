import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { fetchBootstrap, type BootstrapResponse } from "../lib/api";

interface CapabilitiesState {
  data: BootstrapResponse | null;
  token: string | null;
  error: Error | null;
  /** True while attempting to verify a token against `/v1/bootstrap`. */
  isLoading: boolean;
  /**
   * True until the first auto-detect (Tauri injection / URL fragment)
   * has settled. The auth gate renders a neutral splash instead of the
   * token-entry form during this brief window so we don't flash the
   * form on a tab that's about to authenticate via fragment.
   */
  isResolving: boolean;
  /**
   * Submit a manually-pasted token. The provider validates it via
   * ``/v1/bootstrap``; on success it persists for the tab session and
   * returns ``{ ok: true }``. On failure (401, network error, etc.) it
   * returns ``{ ok: false, error }`` and leaves the state unauthenticated.
   */
  submitToken: (token: string) => Promise<{ ok: true } | { ok: false; error: Error }>;
  /** Drop the token. Reverts to the unauthenticated state. */
  clearToken: () => void;
}

const CapabilitiesContext = createContext<CapabilitiesState | null>(null);

function readTokenOnce(): string | null {
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
  const [data, setData] = useState<BootstrapResponse | null>(null);
  const [token, setTokenState] = useState<string | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isResolving, setIsResolving] = useState(true);
  const cancelledRef = useRef(false);

  const verifyAndStore = useCallback(
    async (candidate: string): Promise<{ ok: true } | { ok: false; error: Error }> => {
      setIsLoading(true);
      setError(null);
      try {
        const bootstrap = await fetchBootstrap(candidate);
        if (cancelledRef.current) return { ok: true };
        if (typeof window !== "undefined") {
          window.__SPREN_AUTH__ = candidate;
        }
        setTokenState(candidate);
        setData(bootstrap);
        setError(null);
        setIsLoading(false);
        return { ok: true };
      } catch (err) {
        const e = err instanceof Error ? err : new Error(String(err));
        if (!cancelledRef.current) {
          setError(e);
          setData(null);
          setTokenState(null);
          setIsLoading(false);
        }
        return { ok: false, error: e };
      }
    },
    [],
  );

  useEffect(() => {
    cancelledRef.current = false;
    const candidate = readTokenOnce();
    if (!candidate) {
      setIsResolving(false);
      return () => {
        cancelledRef.current = true;
      };
    }
    void verifyAndStore(candidate).finally(() => {
      if (!cancelledRef.current) setIsResolving(false);
    });
    return () => {
      cancelledRef.current = true;
    };
    // verifyAndStore is stable (useCallback with no deps).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const submitToken = useCallback(
    (candidate: string) => verifyAndStore(candidate),
    [verifyAndStore],
  );

  const clearToken = useCallback(() => {
    if (typeof window !== "undefined") {
      window.__SPREN_AUTH__ = undefined;
    }
    setTokenState(null);
    setData(null);
    setError(null);
  }, []);

  const state: CapabilitiesState = {
    data,
    token,
    error,
    isLoading,
    isResolving,
    submitToken,
    clearToken,
  };

  return <CapabilitiesContext.Provider value={state}>{children}</CapabilitiesContext.Provider>;
}

export function useCapabilities(): CapabilitiesState {
  const ctx = useContext(CapabilitiesContext);
  if (!ctx) throw new Error("useCapabilities must be used inside CapabilitiesProvider");
  return ctx;
}
