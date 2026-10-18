import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

// COOP/COEP make the page cross-origin isolated, which SharedArrayBuffer (shared frames, WASM threads) needs.
// Production hosting must send the same two headers (DESIGN.md §7.4).
const isolation = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

// The production headers (DESIGN.md §7.4, docs/security.md): isolation plus a strict Content-Security-Policy.
// 'wasm-unsafe-eval' lets WebAssembly compile; 'unsafe-inline' styles are needed by KaTeX's inline style
// attributes. Hosting must send the same set. The dev server only gets isolation (HMR needs more).
const production = {
  ...isolation,
  "Content-Security-Policy": [
    "default-src 'self'",
    "script-src 'self' 'wasm-unsafe-eval'",
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data:",
    "font-src 'self' data:",
    "connect-src 'self'",
    "worker-src 'self' blob:",
    "object-src 'none'",
    "base-uri 'none'",
    "form-action 'none'",
    "frame-ancestors 'none'",
  ].join("; "),
  "Cross-Origin-Resource-Policy": "same-origin",
  "X-Content-Type-Options": "nosniff",
  "Referrer-Policy": "no-referrer",
};

export default defineConfig({
  plugins: [svelte()],
  // /shaders sits beside web/ and is shared with the native kernel tests
  server: { headers: isolation, port: 5173, fs: { allow: [".."] } },
  // explicit IPv4: "localhost" can resolve to ::1 only, which tools probing 127.0.0.1 (Playwright) never reach
  preview: { headers: production, port: 4173, host: "127.0.0.1" },
  worker: { format: "es" },
  build: {
    target: "es2022",
    rollupOptions: { input: { main: "index.html", bench: "bench.html" } },
  },
  // the wasm-bindgen output is loaded as-is; don't pre-bundle it
  optimizeDeps: { exclude: ["./src/wasm/pkg-st", "./src/wasm/pkg-threads"] },
});
