/**
 * TanStack Query client wrapped for the Spren app.
 *
 * One QueryClient lives at the root; child components consume it via
 * `useQueryClient`. Defaults err on the conservative side: 30s stale
 * window for list queries, 5min for single-row reads, no retries on
 * mutations (the server returns structured ErrorEnvelope responses; the
 * frontend should surface those rather than retry blind).
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState, type ReactNode } from "react";

export function QueryProvider({ children }: { children: ReactNode }) {
  const [client] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 30_000,
            refetchOnWindowFocus: false,
            retry: 1,
          },
          mutations: {
            retry: false,
          },
        },
      }),
  );
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}
