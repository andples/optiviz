/**
 * The render host: owns the canvas and a GPU back-end, reads the newest engine frame each display frame and
 * draws it. Runs in the render worker (with an OffscreenCanvas) or, as a fallback, on the main thread; the
 * code is the same, only the message plumbing differs.
 */

import { H, readerFor, type FrameReader } from "../engine/frames";
import type { ProblemManifest, RenderCmd, RenderMsg, Theme } from "../engine/protocol";
import type { Backend, Layer } from "./backend";
import { OrbitCamera } from "./camera";
import { DRAW_FLOATS, SCENE_FLOATS } from "./shaders";
import { createWebGL2 } from "./webgl2";
import { createWebGPU } from "./webgpu";

const MESH_N = 384;

/** Run colours (first is the single-run colour); the trail is drawn in it, the marker filled with it. */
const RUN_COLOR: [number, number, number] = [0.85, 0.24, 0.1];

export class RenderHost {
  private backend?: Backend;
  private reader?: FrameReader;
  private camera = new OrbitCamera();
  private manifest?: ProblemManifest;
  private zrange: [number, number] = [0, 1];
  private theme: Theme = { bg: [0.98, 0.976, 0.965], dark: false };
  private w = 1;
  private h = 1;
  private dpr = 1;
  private scene = new Float32Array(SCENE_FLOATS);
  private draws = {} as Record<Layer, Float32Array>;
  private lastFrameAt = performance.now();
  private dirty = true;
  private trailCount = 0;
  // stats
  private frames = 0;
  private cpuMs = 0;
  private latency = 0;
  private latencyN = 0;
  private statsAt = performance.now();

  constructor(
    private send: (m: RenderMsg) => void,
    private thread: "worker" | "main",
  ) {
    for (const l of ["trail", "points", "marker", "minima"] as Layer[]) this.draws[l] = new Float32Array(DRAW_FLOATS);
  }

  async handle(cmd: RenderCmd) {
    switch (cmd.type) {
      case "init": {
        this.w = cmd.width;
        this.h = cmd.height;
        this.dpr = cmd.dpr;
        this.theme = cmd.theme;
        let be: Backend | null = null;
        const errors: string[] = [];
        if (cmd.prefer !== "webgl2") be = await createWebGPU(cmd.canvas, (message) => this.send({ type: "error", message })).catch((e) => (errors.push(String(e)), null));
        if (!be) be = (() => { try { return createWebGL2(cmd.canvas); } catch (e) { errors.push(String(e)); return null; } })();
        if (!be) return this.send({ type: "failed", reason: errors.join("; ") || "no WebGPU or WebGL2" });
        this.backend = be;
        be.resize(Math.round(this.w * this.dpr), Math.round(this.h * this.dpr));
        if (this.manifest) this.applyProblem();
        this.send({ type: "ready", backend: be.kind, adapter: be.adapter });
        this.loop();
        break;
      }
      case "transport":
        this.reader = readerFor(cmd.transport);
        break;
      case "problem":
        this.manifest = cmd.manifest;
        this.zrange = cmd.zrange;
        if (this.backend) this.applyProblem();
        break;
      case "resize":
        this.w = cmd.width;
        this.h = cmd.height;
        this.dpr = cmd.dpr;
        this.backend?.resize(Math.round(this.w * this.dpr), Math.round(this.h * this.dpr));
        this.dirty = true;
        break;
      case "orbit":
        this.camera.orbit(cmd.dx, cmd.dy);
        break;
      case "zoom":
        this.camera.zoom(cmd.factor);
        break;
      case "theme":
        this.theme = cmd.theme;
        this.dirty = true;
        break;
    }
  }

  private applyProblem() {
    const m = this.manifest!;
    if (!m.shader_expr) return;
    try {
      this.backend!.setProblem(m.shader_expr, MESH_N);
    } catch (e) {
      this.send({ type: "error", message: String(e) });
      return;
    }
    const [[x0, x1], [y0, y1]] = m.bounds;
    const minima = new Float32Array(m.minima.flatMap(([x, y]) => [((x - x0) / (x1 - x0)) * 2 - 1, ((y - y0) / (y1 - y0)) * 2 - 1]));
    this.backend!.upload("minima", minima, m.minima.length);
    this.dirty = true;
  }

  private loop = () => {
    const raf = (self as unknown as { requestAnimationFrame?: (cb: () => void) => number }).requestAnimationFrame;
    const next = () => (raf ? raf.call(self, this.loop) : setTimeout(this.loop, 1000 / 60));
    const t0 = performance.now();
    this.frame(t0);
    this.cpuMs += performance.now() - t0;
    this.frames++;
    if (t0 - this.statsAt > 500) this.stats(t0);
    next();
  };

  private frame(now: number) {
    const be = this.backend;
    if (!be || !this.manifest) return;
    const dt = Math.min(0.1, (now - this.lastFrameAt) / 1000);
    this.lastFrameAt = now;
    let changed = this.camera.update(dt) || this.dirty;

    const got = this.reader?.latest();
    if (got?.fresh) {
      const h = got.frame.header;
      this.trailCount = h[H.trailCount];
      be.upload("trail", got.frame.trail, this.trailCount);
      be.upload("points", got.frame.points, h[H.pointsCount]);
      be.upload("marker", new Float32Array([h[H.markerU], h[H.markerV]]), h[H.iter] >= 0 ? 1 : 0);
      this.latency += now - h[H.publishedAt];
      this.latencyN++;
      changed = true;
    }
    if (!changed) return; // nothing moved: skip the draw entirely
    this.dirty = false;
    this.fillScene(now);
    be.draw(this.scene, this.draws, this.theme.bg);
  }

  private fillScene(now: number) {
    const s = this.scene;
    const m = this.manifest!;
    s.set(this.camera.viewProj(this.w / this.h), 0);
    s.set([...this.camera.eye(), 1], 16);
    s.set([m.bounds[0][0], m.bounds[0][1], m.bounds[1][0], m.bounds[1][1]], 20);
    s.set([this.zrange[0], this.zrange[1], 0.62, m.log_height ? 1 : 0], 24);
    s.set([-0.45, 0.85, 0.35, 0.42], 28); // key light from the upper left, ambient share
    s.set([this.w * this.dpr, this.h * this.dpr, this.dpr, now / 1000], 32);
    s.set([...this.theme.bg, this.theme.dark ? 1 : 0], 36);
    s.set([24, this.theme.dark ? 0.35 : 0.28, 0.22, 0], 40); // contour levels, strength, fog
    const c = RUN_COLOR;
    const ink = this.theme.dark ? [0.92, 0.92, 0.9] : [0.12, 0.12, 0.14];
    this.draws.trail.set([...c, 0.95, 2.5, 0.006, 0, this.trailCount]);
    this.draws.marker.set([...c, 1, 7, 0.012, 1, 1]);
    this.draws.minima.set([ink[0], ink[1], ink[2], 0.75, 6, 0.004, 2, 0]);
    this.draws.points.set([0.1, 0.1, 0.12, 0.55, 1.2, 0.004, 0, 0]);
  }

  private stats(now: number) {
    const secs = (now - this.statsAt) / 1000;
    this.send({
      type: "stats",
      stats: {
        backend: this.backend!.kind,
        adapter: this.backend!.adapter,
        fps: this.frames / secs,
        frameMs: this.frames ? this.cpuMs / this.frames : 0,
        latencyMs: this.latencyN ? this.latency / this.latencyN : 0,
        thread: this.thread,
      },
    });
    this.frames = 0;
    this.cpuMs = 0;
    this.latency = 0;
    this.latencyN = 0;
    this.statsAt = now;
  }
}
