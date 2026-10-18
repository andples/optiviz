// Runs the Playwright suite against the production preview (with its security headers). The preview is started
// here rather than through Playwright's `webServer`, which stalled the test worker on this setup; an already
// running preview on :4173 is reused. Extra arguments go to Playwright.
import { spawn } from "node:child_process";

const url = "http://127.0.0.1:4173/";
const up = async () => fetch(url).then((r) => r.ok, () => false);

let server = null;
if (!(await up())) {
  server = spawn("pnpm", ["preview", "--strictPort"], { stdio: "ignore", detached: true });
  for (let i = 0; i < 100 && !(await up()); i++) await new Promise((r) => setTimeout(r, 100));
  if (!(await up())) {
    console.error("preview server did not start on :4173");
    process.exit(1);
  }
}
const pw = spawn("npx", ["playwright", "test", ...process.argv.slice(2)], { stdio: "inherit" });
const code = await new Promise((r) => pw.on("exit", r));
if (server) process.kill(-server.pid);
process.exit(code ?? 1);
