import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

// COOP/COEP make the page cross-origin isolated, which SharedArrayBuffer (shared frames, WASM threads) needs.
// Production hosting must send the same two headers (DESIGN.md §7.4).
const isolation = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

export default defineConfig({
  plugins: [svelte()],
  // /shaders sits beside web/ and is shared with the native kernel tests
  server: { headers: isolation, port: 5173, fs: { allow: [".."] } },
  preview: { headers: isolation, port: 4173 },
  worker: { format: "es" },
  build: {
    target: "es2022",
    rollupOptions: { input: { main: "index.html", bench: "bench.html" } },
  },
  // the wasm-bindgen output is loaded as-is; don't pre-bundle it
  optimizeDeps: { exclude: ["./src/wasm/pkg-st", "./src/wasm/pkg-threads"] },
});
