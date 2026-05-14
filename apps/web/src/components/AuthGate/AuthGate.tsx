/**
 * Auth gate — renders children only when an authenticated session is
 * resolved. Until then, blocks the entire app with either a neutral
 * splash (while the auto-detect is in flight) or a token-entry screen
 * (when no token has been provided yet, or when validation has failed).
 *
 * Why a single gate at the app shell level: every Spren API call
 * requires an auth token. Letting routes render without the token
 * produces "silent failure" (the API call 401s, the user sees a
 * generic error). The gate makes the auth requirement obvious, gives
 * the user a clear paste affordance, and keeps every other surface
 * un-rendered until auth is real — no half-rendered workflows page,
 * no command palette without backend access.
 */
import { type FormEvent, type ReactNode, useState } from "react";

import { useCapabilities } from "../../providers/capabilities";

import "./AuthGate.css";

export function AuthGate({ children }: { children: ReactNode }) {
  const { token, data, error, isLoading, isResolving, submitToken } = useCapabilities();
  const [pasted, setPasted] = useState("");
  const [submitError, setSubmitError] = useState<Error | null>(null);

  // Resolved + verified: show the app.
  if (token && data) {
    return <>{children}</>;
  }

  // First-paint window — auto-detect in flight, no decision yet.
  if (isResolving) {
    return <SplashScreen />;
  }

  // No token, OR a previous candidate failed verification. Show form.
  const handleSubmit = async (event: FormEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault();
    setSubmitError(null);
    const trimmed = pasted.trim();
    if (!trimmed) {
      setSubmitError(new Error("Token can't be empty."));
      return;
    }
    const result = await submitToken(trimmed);
    if (!result.ok) {
      setSubmitError(result.error);
    }
  };

  // Surface the most relevant error: either the live submit error
  // (from this paste) or the provider's auto-detect / verify error.
  const visibleError = submitError ?? error;

  return (
    <div className="auth-gate" data-testid="auth-gate">
      <main className="auth-gate-card">
        <header className="auth-gate-header">
          <span className="auth-gate-brand">spren.</span>
          <h1 className="auth-gate-title">Authentication required</h1>
        </header>

        <p className="auth-gate-copy">
          Spren generates a fresh token each time the sidecar starts. Paste it
          below to continue. The token stays on your machine — Spren never
          sends it anywhere except your local sidecar.
        </p>

        <details className="auth-gate-help">
          <summary>Where to find the token</summary>
          <div className="auth-gate-help-body">
            <p>
              In the terminal that's running <code>just dev</code>, look for the
              banner that says <strong>“Spren ready. Open in your browser:”</strong>
              — the token is the part after <code>#token=</code> in the URL.
            </p>
            <p>
              If you'd rather paste the URL directly, click the printed link
              instead — it carries the token in the fragment and authenticates
              you automatically.
            </p>
          </div>
        </details>

        <form className="auth-gate-form" onSubmit={handleSubmit}>
          <label htmlFor="auth-gate-token" className="auth-gate-label">
            Auth token
          </label>
          <input
            id="auth-gate-token"
            className="auth-gate-input"
            type="password"
            autoComplete="off"
            spellCheck={false}
            value={pasted}
            onChange={(e) => setPasted(e.target.value)}
            placeholder="Paste the per-launch token"
            data-testid="auth-gate-token-input"
            disabled={isLoading}
            autoFocus
          />
          {visibleError && (
            <p className="auth-gate-error" role="alert" data-testid="auth-gate-error">
              {visibleError.message || "Token rejected by sidecar."}
            </p>
          )}
          <button
            type="submit"
            className="auth-gate-submit"
            disabled={isLoading || pasted.trim().length === 0}
            data-testid="auth-gate-submit"
          >
            {isLoading ? "Verifying…" : "Unlock"}
          </button>
        </form>
      </main>
    </div>
  );
}

function SplashScreen() {
  return (
    <div className="auth-gate" data-testid="auth-gate-splash">
      <div className="auth-gate-splash">
        <span className="auth-gate-brand">spren.</span>
      </div>
    </div>
  );
}
