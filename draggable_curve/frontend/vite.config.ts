import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base: "./" is required -- Streamlit serves this component's built files
// from an arbitrary path, not from the domain root, so asset URLs must be
// relative or they'll 404 once deployed.
export default defineConfig({
  plugins: [react()],
  base: "./",
  server: {
    port: 5173,
  },
  build: {
    outDir: "dist",
  },
});
