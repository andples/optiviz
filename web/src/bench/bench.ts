/** The bench page: runs the WASM builds one after another in fresh workers and prints JSON. */

const out = document.getElementById("out")!;
const state = document.getElementById("state")!;

function runIn(threads: boolean): Promise<unknown> {
  return new Promise((resolve, reject) => {
    const w = new Worker(new URL("./bench.worker.ts", import.meta.url), { type: "module" });
    w.onmessage = (e) => (resolve(e.data), w.terminate());
    w.onerror = (e) => reject(e.message);
    w.postMessage({ threads });
  });
}

(async () => {
  const results: Record<string, unknown> = { userAgent: navigator.userAgent, crossOriginIsolated, cores: navigator.hardwareConcurrency };
  results.st = await runIn(false);
  out.textContent = JSON.stringify(results, null, 2);
  if (crossOriginIsolated) results.threads = await runIn(true);
  out.textContent = JSON.stringify(results, null, 2);
  state.textContent = "done";
  document.body.dataset.done = "1";
})().catch((e) => {
  state.textContent = `failed: ${e}`;
  document.body.dataset.done = "1";
});
