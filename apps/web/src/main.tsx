import React from "react";
import ReactDOM from "react-dom/client";
import { RouterProvider, createRouter } from "@tanstack/react-router";
import { routeTree } from "./routeTree.gen";
import { startPerfMonitor } from "./lib/perf-monitor";
import "./styles/globals.css";

// Spren orb perf kill switch — drops micro-interactions when sustained
// frame-time exceeds 20 ms median over two consecutive 60-frame windows.
startPerfMonitor();

const router = createRouter({ routeTree });

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}

const rootEl = document.getElementById("root");
if (!rootEl) throw new Error("root element missing");

ReactDOM.createRoot(rootEl).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>,
);
