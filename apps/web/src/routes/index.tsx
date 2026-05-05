import { createFileRoute } from "@tanstack/react-router";
import { useCapabilities } from "../providers/capabilities";

export const Route = createFileRoute("/")({
  component: HomeRoute,
});

function HomeRoute() {
  const { data, error, isLoading } = useCapabilities();

  if (error) {
    return (
      <main style={{ padding: 24, fontFamily: "monospace" }}>
        <h1>MARSYS Spren — Foundation Session</h1>
        <p style={{ color: "#e07856" }}>auth required: {error.message}</p>
      </main>
    );
  }

  return (
    <main style={{ padding: 24, fontFamily: "monospace" }}>
      <h1>MARSYS Spren — Foundation Session</h1>
      {isLoading ? (
        <p>Loading bootstrap…</p>
      ) : (
        <pre style={{ background: "#14120f", color: "#f6f1e8", padding: 16, borderRadius: 6 }}>
          {JSON.stringify(data, null, 2)}
        </pre>
      )}
    </main>
  );
}
