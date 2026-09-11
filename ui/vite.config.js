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
    },
  },
});
