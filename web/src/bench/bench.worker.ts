/// <reference lib="webworker" />
/** Runs the M0 workloads in one WebAssembly build and reports timings (best of N). */

type Wasm = typeof import("../wasm/pkg-st/optiviz_wasm.js");

self.onmessage = async (e: MessageEvent<{ threads: boolean }>) => {
  const threads = e.data.threads;
  const mod = (threads ? await import("../wasm/pkg-threads/optiviz_wasm.js") : await import("../wasm/pkg-st/optiviz_wasm.js")) as unknown as Wasm;
  await mod.default();
  let threadCount = 1;
  if (threads) {
    threadCount = Math.min(navigator.hardwareConcurrency || 4, 16);
    await (mod as unknown as { initThreadPool(n: number): Promise<void> }).initThreadPool(threadCount);
  }
  const best = (reps: number, f: () => unknown) => {
    let b = Infinity;
    for (let i = 0; i < reps; i++) {
      const t = performance.now();
      f();
      b = Math.min(b, performance.now() - t);
    }
    return b / 1000;
  };
  const gdIters = 10_000_000;
  const gd = best(5, () => mod.bench_gd_rosenbrock(gdIters));
  const [n, dim, iters] = [4096, 10, 500];
  const many = best(3, () => mod.bench_many(n, dim, iters));
  (self as unknown as Worker).postMessage({
    build: threads ? "wasm-threads" : "wasm-st",
    threads: threadCount,
    gd_rosenbrock_2d: { iters: gdIters, seconds: gd, iters_per_s: gdIters / gd },
    adam_many_rastrigin: { starts: n, dim, iters, seconds: many, start_iters_per_s: (n * iters) / many },
  });
};
