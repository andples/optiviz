/// <reference lib="webworker" />
/**
 * The engine worker: owns the WebAssembly engine, steps the current run as fast as asked and publishes frames
 * for the render worker. Never touches the DOM or the GPU; the main thread only sends commands and receives a
 * status about ten times a second.
 */

import { H, writerFor, type FrameWriter, type Layout } from "./frames";
import type { EngineCmd, EngineMsg, EngineStatus, ProblemManifest } from "./protocol";

type Wasm = typeof import("../wasm/pkg-st/optiviz_wasm.js");
type Run = InstanceType<Wasm["Run"]>;
type Points = InstanceType<Wasm["Points"]>;

const post = (m: EngineMsg) => (self as unknown as Worker).postMessage(m);

let wasm: Wasm;
let memory: WebAssembly.Memory;
let writer: FrameWriter;
let layout: Layout;
let problems: ProblemManifest[] = [];
let manifest: ProblemManifest;
let run: Run | null = null;
let points: Points | null = null;

let running = false;
let speed = 60; // steps per second, 0 = flat out
let owed = 0; // fractional steps carried between ticks
let dirty = true;
let lastTick = performance.now();

// status counters
let stepsSince = 0;
let framesSince = 0;
let publishMsSum = 0;
let lastStatus = performance.now();
let threads = false;

async function init(cmd: Extract<EngineCmd, { type: "init" }>) {
  layout = cmd.transport.layout;
  writer = writerFor(cmd.transport);
  threads = cmd.threads;
  const mod: Wasm = threads
    ? ((await import("../wasm/pkg-threads/optiviz_wasm.js")) as unknown as Wasm)
    : await import("../wasm/pkg-st/optiviz_wasm.js");
  const exports = await mod.default();
  memory = exports.memory;
  wasm = mod;
  let threadCount = 1;
  if (threads) {
    threadCount = Math.max(1, Math.min(navigator.hardwareConcurrency || 4, 16));
    await (mod as unknown as { initThreadPool(n: number): Promise<void> }).initThreadPool(threadCount);
  }
  problems = JSON.parse(wasm.problems_json());
  post({ type: "ready", threads, threadCount, problems, optimizers: JSON.parse(wasm.optimizers_json()) });
  schedule();
}

function startRun(cmd: Extract<EngineCmd, { type: "run" }>) {
  const m = problems.find((p) => p.id === cmd.problem);
  if (!m) return post({ type: "error", message: `unknown problem ${cmd.problem}` });
  const changed = !manifest || manifest.id !== m.id;
  manifest = m;
  run?.free();
  run = new wasm.Run(m.id, cmd.optimizer, new Float64Array(cmd.x0 ?? m.start), BigInt(cmd.seed));
  for (const [k, v] of Object.entries(cmd.params ?? {})) run.set_param(k, v);
  if (changed) {
    const zr = wasm.surface_range(m.id, 257);
    post({ type: "problem", manifest: m, zrange: [zr[0], zr[1]] });
  }
  dirty = true;
}

/** Fills the writer's frame from the run (trail in surface coordinates) and the stress points, then publishes. */
function publish() {
  const t0 = performance.now();
  const fr = writer.frame();
  const h = fr.header;
  if (run) {
    const [[x0, x1], [y0, y1]] = manifest.bounds;
    const sx = 2 / (x1 - x0), sy = 2 / (y1 - y0);
    const len = Math.min(run.trail_len(), 2 * layout.trailCap);
    const src = new Float32Array(memory.buffer, run.trail_ptr(), len);
    const dst = fr.trail;
    for (let i = 0; i < len; i += 2) {
      dst[i] = (src[i] - x0) * sx - 1;
      dst[i + 1] = (src[i + 1] - y0) * sy - 1;
    }
    const x = run.x(), y = run.y();
    h[H.iter] = run.iter();
    h[H.f] = run.f_now();
    h[H.gradNorm] = run.grad_norm();
    h[H.x] = x;
    h[H.y] = y;
    h[H.markerU] = (x - x0) * sx - 1;
    h[H.markerV] = (y - y0) * sy - 1;
    h[H.trailCount] = len / 2;
    h[H.diverged] = run.diverged() ? 1 : 0;
  }
  if (points) {
    const n = Math.min(points.len(), 2 * layout.pointsCap);
    fr.points.set(new Float32Array(memory.buffer, points.ptr(), n));
    h[H.pointsCount] = n / 2;
  } else h[H.pointsCount] = 0;
  writer.publish();
  framesSince++;
  publishMsSum += performance.now() - t0;
}

const BUDGET_MS = 8; // per tick when running flat out, so commands are still picked up promptly

function tick() {
  const now = performance.now();
  const dt = Math.min(0.1, (now - lastTick) / 1000);
  lastTick = now;
  if (run && running && !run.diverged()) {
    if (speed === 0) {
      let n = 256;
      while (performance.now() - now < BUDGET_MS && !run.diverged()) {
        stepsSince += run.step(n);
        n = Math.min(n * 2, 1 << 20);
      }
    } else {
      owed += dt * speed;
      const n = Math.floor(owed);
      owed -= n;
      if (n > 0) {
        stepsSince += run.step(n);
        dirty = true;
      }
    }
    if (speed === 0) dirty = true;
  }
  if (points) {
    points.advance(dt);
    dirty = true;
  }
  if (dirty && writer) {
    publish();
    dirty = false;
  }
  if (now - lastStatus > 100) sendStatus(now);
  schedule();
}

function sendStatus(now: number) {
  const secs = (now - lastStatus) / 1000;
  const s: EngineStatus = {
    iter: run?.iter() ?? 0,
    f: run?.f_now() ?? NaN,
    gradNorm: run?.grad_norm() ?? NaN,
    x: run?.x() ?? NaN,
    y: run?.y() ?? NaN,
    diverged: run?.diverged() ?? false,
    running,
    stepsPerSec: stepsSince / secs,
    framesPerSec: framesSince / secs,
    stressPoints: points ? points.len() / 2 : 0,
    publishMs: framesSince ? publishMsSum / framesSince : 0,
  };
  stepsSince = framesSince = 0;
  publishMsSum = 0;
  lastStatus = now;
  post({ type: "status", status: s });
}

// Flat out: zero-delay ticks (a MessageChannel round trip, not setTimeout, which browsers clamp to 4 ms).
// At a set speed: wake every 4 ms (fine-grained enough for 600 steps/s, no busy loop). Idle: every 16 ms.
const chan = new MessageChannel();
chan.port1.onmessage = tick;
let timer = 0;
function schedule() {
  clearTimeout(timer);
  const active = running && run && !run.diverged();
  if ((active && speed === 0) || points) chan.port2.postMessage(0);
  else timer = self.setTimeout(tick, active ? 4 : 16);
}

self.onmessage = async (e: MessageEvent<EngineCmd>) => {
  const cmd = e.data;
  try {
    switch (cmd.type) {
      case "init":
        return await init(cmd);
      case "run":
        startRun(cmd);
        break;
      case "param":
        run?.set_param(cmd.key, cmd.value);
        break;
      case "play":
        running = true;
        lastTick = performance.now();
        break;
      case "pause":
        running = false;
        break;
      case "step":
        if (run) stepsSince += run.step(cmd.n);
        dirty = true;
        break;
      case "restart":
        run?.restart(new Float64Array(cmd.x0 ?? manifest.start));
        dirty = true;
        break;
      case "speed":
        speed = cmd.speed;
        owed = 0;
        break;
      case "stress":
        points?.free();
        points = cmd.points > 0 ? new wasm.Points(Math.min(cmd.points, layout.pointsCap)) : null;
        dirty = true;
        break;
    }
    schedule();
  } catch (err) {
    post({ type: "error", message: String(err) });
  }
};
