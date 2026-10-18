/// <reference lib="webworker" />
/**
 * The engine worker: owns the WebAssembly engine (a Session of runs on one problem), steps it as fast as asked,
 * publishes frames for the render worker, and reports status, metrics, stopping points and traces to the main
 * thread. Never touches the DOM or the GPU.
 */

import { H, writerFor, type FrameWriter, type Layout } from "./frames";
import type { EngineCmd, EngineMsg, EngineStatus, RunStatus } from "./protocol";

type Wasm = typeof import("../wasm/pkg-st/optiviz_wasm.js");
type Engine = InstanceType<Wasm["Engine"]>;
type Points = InstanceType<Wasm["Points"]>;

const post = (m: EngineMsg, transfer: Transferable[] = []) => (self as unknown as Worker).postMessage(m, transfer);

let wasm: Wasm;
let memory: WebAssembly.Memory;
let writer: FrameWriter;
let layout: Layout;
let eng: Engine;
let points: Points | null = null;

let running = false;
let speed = 60; // iterations per second, 0 = flat out
let owed = 0; // fractional steps carried between ticks
let dirty = true;
let lastTick = performance.now();
let selected = 0;

// status counters
let stepsSince = 0;
let framesSince = 0;
let publishMsSum = 0;
let lastStatus = performance.now();
let lastMetrics = 0;
let metricsDirty = true;

async function init(cmd: Extract<EngineCmd, { type: "init" }>) {
  layout = cmd.transport.layout;
  writer = writerFor(cmd.transport);
  const mod: Wasm = cmd.threads
    ? ((await import("../wasm/pkg-threads/optiviz_wasm.js")) as unknown as Wasm)
    : await import("../wasm/pkg-st/optiviz_wasm.js");
  const exports = await mod.default();
  memory = exports.memory;
  wasm = mod;
  let threadCount = 1;
  if (cmd.threads) {
    threadCount = Math.max(1, Math.min(navigator.hardwareConcurrency || 4, 16));
    await (mod as unknown as { initThreadPool(n: number): Promise<void> }).initThreadPool(threadCount);
  }
  eng = new wasm.Engine("rosenbrock", 1n);
  post({ type: "ready", threads: cmd.threads, threadCount, problems: JSON.parse(wasm.problems_json()), optimizers: JSON.parse(wasm.optimizers_json()) });
  sendProblem();
  schedule();
}

function sendProblem() {
  post({ type: "problem", info: JSON.parse(eng.problem_json()) });
  dirty = metricsDirty = true;
}

/** Copies the engine's frame (index + data) and the stress points into the writer's slot, then publishes. */
function publish() {
  const t0 = performance.now();
  const fr = writer.frame();
  const h = fr.header;
  eng.build_frame();
  const n = Math.min(eng.frame_len(), 2 * layout.dataCap);
  fr.data.set(new Float32Array(memory.buffer, eng.frame_ptr(), n));
  const il = Math.min(eng.index_len(), h.length - H.index);
  h.set(new Float64Array(memory.buffer, eng.index_ptr(), il), H.index);
  h[H.indexLen] = il;
  h[H.iter] = eng.iter();
  h[H.head] = eng.head();
  if (points) {
    const pn = Math.min(points.len(), 2 * layout.pointsCap);
    fr.points.set(new Float32Array(memory.buffer, points.ptr(), pn));
    h[H.pointsCount] = pn / 2;
  } else h[H.pointsCount] = 0;
  writer.publish();
  framesSince++;
  publishMsSum += performance.now() - t0;
}

const BUDGET_MS = 8; // per tick when running flat out, so commands are still picked up promptly

/** Runs n iterations; handles a stopping point or every run halting. */
function advance(n: number) {
  const done = eng.step(n);
  stepsSince += done;
  if (done > 0) dirty = metricsDirty = true;
  const stop = eng.take_stop();
  if (stop) {
    running = false;
    post({ type: "stop", stop: JSON.parse(stop) });
    sendInspect(selected);
    return false;
  }
  if (eng.all_halted()) {
    running = false;
    return false;
  }
  return done === n;
}

function tick() {
  const now = performance.now();
  const dt = Math.min(0.1, (now - lastTick) / 1000);
  lastTick = now;
  if (running) {
    if (speed === 0) {
      let n = 256;
      while (running && performance.now() - now < BUDGET_MS) {
        if (!advance(n)) break;
        n = Math.min(n * 2, 1 << 18);
      }
    } else {
      owed += dt * speed;
      const n = Math.floor(owed);
      owed -= n;
      if (n > 0) advance(n);
    }
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
  if (metricsDirty && now - lastMetrics > 200) sendMetrics(now);
  schedule();
}

function sendStatus(now: number) {
  const secs = (now - lastStatus) / 1000;
  const s: EngineStatus = {
    iter: eng.iter(),
    head: eng.head(),
    running,
    stepsPerSec: stepsSince / secs,
    framesPerSec: framesSince / secs,
    stressPoints: points ? points.len() / 2 : 0,
    publishMs: framesSince ? publishMsSum / framesSince : 0,
    runs: JSON.parse(eng.runs_json()) as RunStatus[],
  };
  stepsSince = framesSince = 0;
  publishMsSum = 0;
  lastStatus = now;
  post({ type: "status", status: s });
}

function sendMetrics(now: number) {
  const series: Float64Array[] = [];
  for (let i = 0; i < eng.run_count(); i++) series.push(eng.metrics(i));
  post({ type: "metrics", series }, series.map((s) => s.buffer));
  lastMetrics = now;
  metricsDirty = false;
}

function sendInspect(i: number) {
  if (i >= eng.run_count()) return;
  const t = JSON.parse(eng.inspect(i));
  post({ type: "trace", i, iter: eng.iter(), entries: t.entries ?? [] });
}

// Flat out: zero-delay ticks (a MessageChannel round trip, not setTimeout, which browsers clamp to 4 ms).
// At a set speed: wake every 4 ms (fine-grained enough for 600 steps/s, no busy loop). Idle: every 16 ms.
const chan = new MessageChannel();
chan.port1.onmessage = tick;
let timer = 0;
function schedule() {
  clearTimeout(timer);
  if ((running && speed === 0) || points) chan.port2.postMessage(0);
  else timer = self.setTimeout(tick, running ? 4 : 16);
}

/** Changes that restart or rewrite the timeline report the new state at once. */
function changed(problem = false) {
  if (problem) sendProblem();
  dirty = metricsDirty = true;
  sendStatus(performance.now());
}

self.onmessage = async (e: MessageEvent<EngineCmd>) => {
  const cmd = e.data;
  try {
    switch (cmd.type) {
      case "init":
        return await init(cmd);
      case "problem":
        eng.set_problem(cmd.id, JSON.stringify(cmd.coeffs), cmd.keepStart);
        changed(true);
        break;
      case "addRun":
        eng.add_run(cmd.opt, JSON.stringify(cmd.params ?? {}));
        changed();
        break;
      case "removeRun":
        eng.remove_run(cmd.i);
        changed();
        break;
      case "setOptimizer":
        eng.set_optimizer(cmd.i, cmd.opt);
        changed();
        break;
      case "param":
        eng.set_param(cmd.i, cmd.key, cmd.value);
        changed();
        break;
      case "select":
        selected = cmd.i;
        eng.select(cmd.i);
        if (!running) sendInspect(cmd.i);
        break;
      case "play":
        running = true;
        lastTick = performance.now();
        owed = 0;
        break;
      case "pause":
        running = false;
        sendInspect(selected);
        break;
      case "step":
        running = false;
        advance(cmd.n);
        sendInspect(selected);
        changed();
        break;
      case "restart":
        eng.restart();
        changed();
        break;
      case "seek":
        running = false;
        eng.seek(cmd.k);
        sendInspect(selected);
        changed();
        break;
      case "speed":
        speed = cmd.speed;
        owed = 0;
        break;
      case "stops":
        eng.set_stops(JSON.stringify(cmd.stops));
        break;
      case "start":
        eng.set_start(cmd.x, cmd.y);
        changed(true);
        break;
      case "startUv":
        eng.set_start_uv(cmd.u, cmd.v);
        changed(true);
        break;
      case "pickStart":
        if (eng.pick_start(cmd.o[0], cmd.o[1], cmd.o[2], cmd.d[0], cmd.d[1], cmd.d[2], cmd.hs)) changed(true);
        break;
      case "fitView":
        eng.fit_view();
        changed(true);
        break;
      case "resetView":
        eng.reset_view();
        changed(true);
        break;
      case "inspect":
        sendInspect(cmd.i);
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
