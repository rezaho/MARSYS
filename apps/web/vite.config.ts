import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { tanstackRouter } from "@tanstack/router-plugin/vite";
import tailwindcss from "@tailwindcss/vite";

// TanStack Router plugin MUST come before @vitejs/plugin-react.
// Tailwind v4 plugin sits alongside the others; it reads
// `@theme inline` blocks from imported CSS to derive utility classes.
export default defineConfig({
  plugins: [
    tanstackRouter({ target: "react", autoCodeSplitting: true }),
    react(),
    tailwindcss(),
  ],
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
  },
  clearScreen: false,
  build: {
    outDir: "dist",
    sourcemap: true,
  },
});
