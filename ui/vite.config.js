import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  base: "/",
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      "/agents": "http://127.0.0.1:8000",
      "/sessions": "http://127.0.0.1:8000",
      "/observability/summary": "http://127.0.0.1:8000",
      "/observability/runs": "http://127.0.0.1:8000",
      "/observability/comments": "http://127.0.0.1:8000",
      "/observability/pricing": "http://127.0.0.1:8000",
      "/learning/summary": "http://127.0.0.1:8000",
      "/learning/feedback": "http://127.0.0.1:8000",
      "/learning/lessons": "http://127.0.0.1:8000",
      "/learning/analyze": "http://127.0.0.1:8000",
      "/learning/incidents": "http://127.0.0.1:8000",
      "/learning/code": "http://127.0.0.1:8000",
      "/learning/preview": "http://127.0.0.1:8000",
      "/learning/eval.jsonl": "http://127.0.0.1:8000",
    },
  },
});
