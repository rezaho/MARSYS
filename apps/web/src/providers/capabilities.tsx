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
   * True until the first auto-detect (Tauri injection / URL fragment /
   * persisted token) has settled. The auth gate renders a neutral splash
   * instead of the token-entry form during this brief window so we don't
   * flash the form on a tab that's about to authenticate.
   */
  isResolving: boolean;
  /**
   * Submit a manually-pasted token. The provider validates it via
   * ``/v1/bootstrap``; on success it persists for the browser (so a
   * reload / deep-link does not re-prompt) and returns ``{ ok: true }``.
   * On failure it returns ``{ ok: false, error }`` and stays unauth'd.
   */
  submitToken: (token: string) => Promise<{ ok: true } | { ok: false; error: Error }>;
  /** Drop the token (memory + persisted). Reverts to unauthenticated. */
  clearToken: () => void;
}

const CapabilitiesContext = createContext<CapabilitiesState | null>(null);

// Per-launch token persistence (ARCH-Q-001, user-decided 2026-05-15:
// localStorage, fragment-wins, stale-clears). The token is localhost-only,
// never crosses the wire, rotates every sidecar launch, and a stale one is
// rejected by /v1/bootstrap and cleared here — an accepted trade-off for a
// single-user-local product (SP-002 + SP-008).
const STORAGE_KEY = "spren.auth.token";

function readPersistedToken(): string | null {
  try {
    return window.localStorage.getItem(STORAGE_KEY);
  } catch {
    return null; // private mode / storage disabled
  }
}

function persistToken(token: string): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, token);
  } catch {
    /* storage unavailable — in-memory still works for this session */
  }
}

function clearPersistedToken(): void {
  try {
    window.localStorage.removeItem(STORAGE_KEY);
  } catch {
    /* ignore */
  }
}

type TokenSource = "tauri" | "fragment" | "storage";

/**
 * PURE — no side effects, safe under React StrictMode's double-invoked
 * effect (the UI-BUG-007 root cause was stripping the URL fragment
 * *inside* the read: invoke #1 destroyed it, invoke #2 found nothing).
 *
 * Resolution order: Tauri injection → a `#token=` URL fragment (always
 * wins, so a fresh `just dev` token overrides a stale persisted one) →
 * the persisted token.
 */
function readCandidateToken(): { token: string; source: TokenSource } | null {
  if (typeof window === "undefined") return null;
  if (window.__SPREN_AUTH__) return { token: window.__SPREN_AUTH__, source: "tauri" };
  const hash = window.location.hash;
  if (hash.startsWith("#")) {
    const params = new URLSearchParams(hash.slice(1));
    const token = params.get("token");
    if (token) return { token, source: "fragment" };
  }
  const persisted = readPersistedToken();
  if (persisted) return { token: persisted, source: "storage" };
  return null;
}

/**
 * Idempotent — strips the token param from the URL only if present.
 * Called AFTER the token is captured + persisted, so a StrictMode second
 * invoke (or any remount) that finds no fragment falls back to the
 * persisted token instead of de-authing.
 */
function stripTokenFragment(): void {
  if (typeof window === "undefined") return;
  const hash = window.location.hash;
  if (!hash.startsWith("#")) return;
  const params = new URLSearchParams(hash.slice(1));
  if (!params.has("token")) return;
  window.history.replaceState(
    null,
    "",
    window.location.pathname + window.location.search,
  );
}

export function CapabilitiesProvider({ children }: { children: ReactNode }) {
  const [data, setData] = useState<BootstrapResponse | null>(null);
  const [token, setTokenState] = useState<string | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isResolving, setIsResolving] = useState(true);
  const cancelledRef = useRef(false);

  const verifyAndStore = useCallback(
    async (
      candidate: string,
      source: TokenSource,
    ): Promise<{ ok: true } | { ok: false; error: Error }> => {
      setIsLoading(true);
      setError(null);
      try {
        const bootstrap = await fetchBootstrap(candidate);
        // Persist + expose the verified token BEFORE the cancelled check
        // so a StrictMode-cancelled first invoke still leaves the good
        // token recoverable (storage + window global). Only the React
        // state set is guarded against an unmounted tree.
        if (typeof window !== "undefined") {
          window.__SPREN_AUTH__ = candidate;
        }
        persistToken(candidate);
        stripTokenFragment();
        if (cancelledRef.current) return { ok: true };
        setTokenState(candidate);
        setData(bootstrap);
        setError(null);
        setIsLoading(false);
        return { ok: true };
      } catch (err) {
        const e = err instanceof Error ? err : new Error(String(err));
        // A persisted token that no longer validates is stale (the
        // sidecar restarted with a new per-launch token). Clear it so we
        // don't loop on a dead token and fall through to the entry form.
        if (source === "storage") clearPersistedToken();
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
    const candidate = readCandidateToken();
    if (!candidate) {
      setIsResolving(false);
      return () => {
        cancelledRef.current = true;
      };
    }
    void verifyAndStore(candidate.token, candidate.source).finally(() => {
      if (!cancelledRef.current) setIsResolving(false);
    });
    return () => {
      cancelledRef.current = true;
    };
    // verifyAndStore is stable (useCallback with no deps).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const submitToken = useCallback(
    (candidate: string) => verifyAndStore(candidate, "fragment"),
    [verifyAndStore],
  );

  const clearToken = useCallback(() => {
    if (typeof window !== "undefined") {
      window.__SPREN_AUTH__ = undefined;
    }
    clearPersistedToken();
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
