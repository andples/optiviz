/**
 * Boot: detect what the browser can do, start the render side (worker with OffscreenCanvas, else the main
 * thread), then the engine worker, and connect them with the best frame transport available.
 *
 * Test the fallbacks with URL parameters: ?render=main (main-thread rendering), ?backend=webgl2 (skip WebGPU),
 * ?shared=0 (no SharedArrayBuffer: post-message frames and the single-threaded engine).
 */

import { createTriple, type Layout, type TransportInit } from "./engine/frames";
import type { EngineCmd, EngineMsg, RenderCmd, RenderMsg, Theme } from "./engine/protocol";

export interface Caps {
  isolated: boolean; // cross-origin isolated with SharedArrayBuffer
  offscreen: boolean;
  webgpu: boolean;
  wasmSimd: boolean;
  cores: number;
}

const SIMD_PROBE = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11]);

export function detectCaps(): Caps {
  return {
    isolated: self.crossOriginIsolated === true && typeof SharedArrayBuffer !== "undefined",
    offscreen: typeof HTMLCanvasElement !== "undefined" && "transferControlToOffscreen" in HTMLCanvasElement.prototype,
    webgpu: "gpu" in navigator,
    wasmSimd: WebAssembly.validate(SIMD_PROBE),
    cores: navigator.hardwareConcurrency || 1,
  };
}

export const LAYOUT: Layout = { dataCap: 131_072, pointsCap: 1 << 20 };

export interface Tier {
  render: "worker" | "main";
  backend: "webgpu" | "webgl2";
  adapter: string;
  transport: "shared" | "post";
  threads: boolean;
  threadCount: number;
}

export interface Session {
  engine: (cmd: EngineCmd) => void;
  render: (cmd: RenderCmd) => void;
}

export interface Hooks {
  engine: (m: EngineMsg) => void;
  render: (m: RenderMsg) => void;
  /** The canvas was handed to a worker that couldn't use it; give me a fresh one for the main thread. */
  freshCanvas: () => HTMLCanvasElement;
}

export async function boot(canvas: HTMLCanvasElement, theme: Theme, hooks: Hooks): Promise<{ session: Session; tier: Tier; caps: Caps }> {
  const q = new URLSearchParams(location.search);
  const caps = detectCaps();
  const shared = caps.isolated && q.get("shared") !== "0";
  const prefer = q.get("backend") === "webgl2" ? ("webgl2" as const) : undefined;
  const size = () => ({ width: canvas.clientWidth || innerWidth, height: canvas.clientHeight || innerHeight, dpr: Math.min(devicePixelRatio || 1, 2) });

  /* ---- render side ---- */
  let render!: (cmd: RenderCmd, transfer?: Transferable[]) => void;
  let ready!: RenderMsg & { type: "ready" };
  let where: "worker" | "main" = "worker";

  const startMain = async (cv: HTMLCanvasElement) => {
    const { RenderHost } = await import("./render/host");
    let resolve!: (m: RenderMsg) => void;
    const first = new Promise<RenderMsg>((r) => (resolve = r));
    const host = new RenderHost((m) => (m.type === "stats" || m.type === "error" ? hooks.render(m) : resolve(m)), "main");
    render = (cmd) => void host.handle(cmd);
    render({ type: "init", canvas: cv, ...size(), theme, prefer });
    return first;
  };

  let first: RenderMsg;
  if (caps.offscreen && q.get("render") !== "main") {
    const worker = new Worker(new URL("./render/render.worker.ts", import.meta.url), { type: "module" });
    const off = canvas.transferControlToOffscreen();
    first = await new Promise<RenderMsg>((resolve) => {
      worker.onmessage = (e: MessageEvent<RenderMsg>) => (e.data.type === "stats" || e.data.type === "error" ? hooks.render(e.data) : resolve(e.data));
      worker.postMessage({ type: "init", canvas: off, ...size(), theme, prefer } satisfies RenderCmd, [off]);
    });
    render = (cmd, transfer = []) => worker.postMessage(cmd, transfer);
    if (first.type === "failed") {
      worker.terminate();
      where = "main";
      first = await startMain(hooks.freshCanvas());
    }
  } else {
    where = "main";
    first = await startMain(canvas);
  }
  if (first.type !== "ready") throw new Error(first.type === "failed" ? first.reason : "renderer did not start");
  ready = first;

  /* ---- frame transport ---- */
  let toEngine: TransportInit, toRender: TransportInit;
  if (shared) toEngine = toRender = createTriple(LAYOUT);
  else {
    const ch = new MessageChannel();
    toEngine = { kind: "post", port: ch.port1, layout: LAYOUT };
    toRender = { kind: "post", port: ch.port2, layout: LAYOUT };
  }
  render({ type: "transport", transport: toRender }, toRender.kind === "post" ? [toRender.port] : []);

  /* ---- engine ---- */
  const engineWorker = new Worker(new URL("./engine/engine.worker.ts", import.meta.url), { type: "module" });
  const engineReady = new Promise<EngineMsg & { type: "ready" }>((resolve) => {
    engineWorker.onmessage = (e: MessageEvent<EngineMsg>) => {
      if (e.data.type === "ready") resolve(e.data);
      hooks.engine(e.data);
    };
  });
  engineWorker.postMessage({ type: "init", transport: toEngine, threads: shared } satisfies EngineCmd, toEngine.kind === "post" ? [toEngine.port] : []);
  const er = await engineReady;

  return {
    caps,
    session: { engine: (cmd) => engineWorker.postMessage(cmd), render: (cmd) => render(cmd) },
    tier: { render: where, backend: ready.backend, adapter: ready.adapter, transport: shared ? "shared" : "post", threads: er.threads, threadCount: er.threadCount },
  };
}
