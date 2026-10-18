/**
 * The render host: owns the canvas and a GPU back-end, reads the newest engine frame each display frame and
 * draws it: the surface (3D, or flat from above in the contour view), each run's trail with a halo so it reads
 * on any colour, Nelder–Mead's simplex, markers, the start and the known minima. Runs in the render worker
 * (with an OffscreenCanvas) or, as a fallback, on the main thread; the code is the same.
 *
 * Health (DESIGN.md §4.3) shows on the marker: out of view → a hollow ring pinned to the box edge; diverging or
 * unbounded → an extra pulsing ring; diverged → a hollow ring at the last finite point.
 */

import { H, IH, INDEX_HEAD, R, RUN_STRIDE, readerFor, type FrameReader } from "../engine/frames";
import type { ProblemInfo, RenderCmd, RenderMsg, Theme, ViewMode } from "../engine/protocol";
import type { Backend, DrawItem } from "./backend";
import { ContourCamera, OrbitCamera, rayThrough } from "./camera";
import { DRAW_FLOATS, SCENE_FLOATS } from "./shaders";
import { createWebGL2 } from "./webgl2";
import { createWebGPU } from "./webgpu";
import { RUN_COLORS } from "../ui/colors";

const MESH_N = 384;
const HEIGHT_SCALE = 0.62;

export class RenderHost {
  private backend?: Backend;
  private reader?: FrameReader;
  private orbit = new OrbitCamera();
  private contour = new ContourCamera();
  private mode: ViewMode = "surface";
  private info?: ProblemInfo;
  private theme: Theme = { bg: [0.98, 0.976, 0.965], dark: false };
  private w = 1;
  private h = 1;
  private dpr = 1;
  private scene = new Float32Array(SCENE_FLOATS);
  private items: DrawItem[] = [];
  private uniforms: Float32Array[] = [];
  private marks = new Float32Array(64);
  private lastFrameAt = performance.now();
  private dirty = true;
  private selected = 0;
  private arrows = false;
  private time = 0;
  /** the latest frame's index, kept for redraws without a new frame */
  private index = new Float64Array(128);
  private points = 0;
  // stats
  private frames = 0;
  private cpuMs = 0;
  private latency = 0;
  private latencyN = 0;
  private statsAt = performance.now();

  constructor(
    private send: (m: RenderMsg) => void,
    private thread: "worker" | "main",
  ) {}

  async handle(cmd: RenderCmd) {
    switch (cmd.type) {
      case "init": {
        this.w = cmd.width;
        this.h = cmd.height;
        this.dpr = cmd.dpr;
        this.theme = cmd.theme;
        let be: Backend | null = null;
        const errors: string[] = [];
        if (cmd.prefer !== "webgl2") {
          be = await createWebGPU(cmd.canvas, (message) => this.send({ type: "error", message })).catch((e) => {
            errors.push(String(e));
            return null;
          });
        }
        if (!be) {
          try {
            be = createWebGL2(cmd.canvas);
          } catch (e) {
            errors.push(String(e));
          }
        }
        if (!be) return this.send({ type: "failed", reason: errors.join("; ") || "no WebGPU or WebGL2" });
        this.backend = be;
        be.resize(Math.round(this.w * this.dpr), Math.round(this.h * this.dpr));
        if (this.info) this.applyProblem();
        this.send({ type: "ready", backend: be.kind, adapter: be.adapter });
        this.loop();
        break;
      }
      case "transport":
        this.reader = readerFor(cmd.transport);
        break;
      case "problem": {
        const old = this.info?.manifest;
        const exprChanged = !old || old.shader_expr !== cmd.info.manifest.shader_expr;
        this.info = cmd.info;
        if (this.backend) {
          if (exprChanged) this.applyProblem();
          else this.dirty = true;
        }
        break;
      }
      case "resize":
        this.w = cmd.width;
        this.h = cmd.height;
        this.dpr = cmd.dpr;
        this.backend?.resize(Math.round(this.w * this.dpr), Math.round(this.h * this.dpr));
        this.dirty = true;
        break;
      case "orbit":
        if (this.mode === "contour") this.contour.pan(cmd.dx, cmd.dy, this.h);
        else this.orbit.orbit(cmd.dx, cmd.dy);
        this.dirty = true;
        break;
      case "zoom":
        if (this.mode === "contour") this.contour.zoom(cmd.factor);
        else this.orbit.zoom(cmd.factor);
        this.dirty = true;
        break;
      case "camera":
        this.orbit.setMode(cmd.mode);
        this.dirty = true;
        break;
      case "keys":
        this.orbit.setKeys(cmd.keys);
        break;
      case "view":
        this.mode = cmd.mode;
        this.dirty = true;
        break;
      case "select":
        this.selected = cmd.i;
        this.dirty = true;
        break;
      case "arrows":
        this.arrows = cmd.on;
        this.dirty = true;
        break;
      case "theme":
        this.theme = cmd.theme;
        this.dirty = true;
        break;
      case "pick": {
        const aspect = this.w / this.h;
        const nx = (2 * cmd.x) / this.w - 1;
        const ny = 1 - (2 * cmd.y) / this.h;
        if (this.mode === "contour") this.send({ type: "picked", uv: this.contour.uvAt(nx, ny, aspect) });
        else {
          const ray = rayThrough(this.orbit.viewProj(aspect), nx, ny);
          if (ray) this.send({ type: "picked", ray: { ...ray, hs: HEIGHT_SCALE } });
        }
        break;
      }
    }
  }

  private applyProblem() {
    const m = this.info!.manifest;
    if (!m.shader_expr) return;
    try {
      this.backend!.setProblem(m.shader_expr, MESH_N);
    } catch (e) {
      this.send({ type: "error", message: String(e) });
      return;
    }
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
    if (!be || !this.info) return;
    const dt = Math.min(0.1, (now - this.lastFrameAt) / 1000);
    this.lastFrameAt = now;
    this.time += dt;
    let changed = (this.mode === "surface" && this.orbit.update(dt)) || this.dirty;

    const got = this.reader?.latest();
    if (got?.fresh) {
      const h = got.frame.header;
      const il = h[H.indexLen];
      this.index.fill(0);
      this.index.set(h.subarray(H.index, H.index + il));
      be.upload("data", got.frame.data, this.dataPoints());
      this.points = h[H.pointsCount];
      be.upload("stress", got.frame.points, this.points);
      this.latency += now - h[H.publishedAt];
      this.latencyN++;
      changed = true;
    }
    // a warning ring pulses while a run is diverging: keep drawing
    if (this.anyWarning()) changed = true;
    if (!changed) return; // nothing moved: skip the draw entirely
    this.dirty = false;
    this.buildItems();
    be.upload("marks", this.marks, this.marks.length / 2);
    this.fillScene(now);
    be.draw(this.scene, this.items, this.theme.bg);
  }

  /** Points used in the data area: the end of the last region the index points into. */
  private dataPoints() {
    const ix = this.index;
    let n = Math.max(ix[IH.minOff] + ix[IH.minN], ix[IH.startsOff] + ix[IH.startsN]);
    for (let i = 0; i < ix[IH.runs]; i++) {
      const r = INDEX_HEAD + RUN_STRIDE * i;
      for (const [o, c] of [[R.trailOff, R.trailN], [R.shapeOff, R.shapeN], [R.okOff, R.okN], [R.badOff, R.badN]]) n = Math.max(n, ix[r + o] + ix[r + c]);
    }
    return n;
  }

  private anyWarning() {
    for (let i = 0; i < this.index[0]; i++) {
      const hc = this.index[INDEX_HEAD + RUN_STRIDE * i + R.health];
      if (hc === 2 || hc === 3) return true;
    }
    return false;
  }

  private item(kind: DrawItem["kind"], buffer: DrawItem["buffer"], first: number, count: number, u: number[]) {
    const k = this.items.length;
    if (!this.uniforms[k]) this.uniforms[k] = new Float32Array(DRAW_FLOATS);
    this.uniforms[k].set(u);
    this.items.push({ kind, buffer, first, count, uniforms: this.uniforms[k] });
  }

  private buildItems() {
    this.items.length = 0;
    const ix = this.index;
    const runs = ix[0];
    const dark = this.theme.dark;
    const halo = dark ? [0.08, 0.09, 0.1] : [1, 1, 1];
    const ink = dark ? [0.92, 0.92, 0.9] : [0.12, 0.12, 0.14];
    const order = [...Array(runs).keys()].filter((i) => i !== this.selected);
    if (this.selected < runs) order.push(this.selected); // selected run on top
    // marks: one marker per run, then the start
    const markAt: number[] = [];
    let m = 0;
    for (let i = 0; i < runs; i++) {
      const r = INDEX_HEAD + RUN_STRIDE * i;
      let u = ix[r + R.mu], v = ix[r + R.mv];
      if (!Number.isFinite(u) || !Number.isFinite(v)) u = v = 0; // diverged: drawn from the trail instead
      const s = Math.max(1, Math.abs(u), Math.abs(v)); // pinned to the box edge when outside
      this.marks[2 * m] = u / s;
      this.marks[2 * m + 1] = v / s;
      markAt[i] = m++;
    }
    const startAt = m;
    if (runs > 0) {
      this.marks[2 * m] = ix[INDEX_HEAD + R.su];
      this.marks[2 * m + 1] = ix[INDEX_HEAD + R.sv];
      m++;
    }
    // the steepest-descent arrow at the selected marker (shaft, then the two barbs), while paused
    let arrowAt = -1;
    const sr = INDEX_HEAD + RUN_STRIDE * this.selected;
    if (this.arrows && this.selected < runs && ix[sr + R.health] === 0) {
      const u = ix[sr + R.mu], v = ix[sr + R.mv], du = ix[sr + R.du], dv = ix[sr + R.dv];
      if (Number.isFinite(u) && (du || dv)) {
        const L = 0.16, b = 0.045;
        const tu = u + L * du, tv = v + L * dv;
        const pts = [u, v, tu, tv, tu - b * (du - 0.6 * dv), tv - b * (dv + 0.6 * du), tu, tv, tu - b * (du + 0.6 * dv), tv - b * (dv - 0.6 * du)];
        arrowAt = m;
        this.marks.set(pts, 2 * m);
        m += 5;
      }
    }

    for (const i of order) {
      const r = INDEX_HEAD + RUN_STRIDE * i;
      const c = RUN_COLORS[i % RUN_COLORS.length].rgb;
      const sel = i === this.selected;
      const t0 = ix[r + R.trailOff], tn = ix[r + R.trailN];
      const width = sel ? 3.2 : 2.4;
      this.item("trail", "data", t0, tn, [...halo, 0.6, width + 2.5, 0.006, 1, tn]);
      this.item("trail", "data", t0, tn, [...c, 1, width, 0.007, 0, tn]);
      const s0 = ix[r + R.shapeOff], sn = ix[r + R.shapeN];
      if (sn > 1) {
        this.item("trail", "data", s0, sn, [...halo, 0.8, 4.5, 0.008, 1, sn]);
        this.item("trail", "data", s0, sn, [...c, 0.9, 1.6, 0.009, 1, sn]);
      }
      // the other walkers (several starting points): small discs; failed ones hollow
      this.item("points", "data", ix[r + R.okOff], ix[r + R.okN], [...c, 0.95, sel ? 4.2 : 3.6, 0.012, 1, 0]);
      this.item("points", "data", ix[r + R.badOff], ix[r + R.badN], [...c, 0.9, 4.5, 0.012, 2, 0]);
      const hc = ix[r + R.health];
      const halted = ix[r + R.halted] > 0;
      const size = sel ? 7.5 : 6.5;
      if (halted || !Number.isFinite(ix[r + R.mu])) {
        if (tn > 0) this.item("points", "data", t0 + tn - 1, 1, [...c, 1, size + 2, 0.014, 2, 1]);
      } else if (hc === 1) {
        this.item("points", "marks", markAt[i], 1, [...c, 1, size + 2, 0.014, 2, 1]);
      } else {
        if (hc === 2 || hc === 3) {
          const pulse = 0.5 + 0.5 * Math.sin(this.time * 6);
          this.item("points", "marks", markAt[i], 1, [...c, 0.35 + 0.4 * pulse, size + 6 + 3 * pulse, 0.013, 2, 1]);
        }
        this.item("points", "marks", markAt[i], 1, [...c, 1, size, 0.014, 1, 1]);
      }
    }
    if (arrowAt >= 0) {
      const c = RUN_COLORS[this.selected % RUN_COLORS.length].rgb;
      this.item("trail", "marks", arrowAt, 5, [...halo, 0.8, 5, 0.016, 1, 5]);
      this.item("trail", "marks", arrowAt, 5, [...c, 1, 2.2, 0.017, 1, 5]);
    }
    // the best point each run has found (with several starts or restarts): a ring around a dot
    for (const i of order) {
      const r = INDEX_HEAD + RUN_STRIDE * i;
      if (!Number.isFinite(ix[r + R.bu])) continue;
      const c = RUN_COLORS[i % RUN_COLORS.length].rgb;
      this.marks[2 * m] = ix[r + R.bu];
      this.marks[2 * m + 1] = ix[r + R.bv];
      this.item("points", "marks", m, 1, [...halo, 0.9, 11, 0.02, 2, 0]);
      this.item("points", "marks", m, 1, [...c, 1, 9.5, 0.021, 2, 0]);
      m++;
    }
    // known minima, the start(s), the stress swarm
    this.item("points", "data", ix[IH.startsOff], ix[IH.startsN], [...ink, 0.45, 3, 0.01, 2, 0]);
    this.item("points", "data", ix[1], ix[2], [...ink, 0.7, 6, 0.004, 2, 0]);
    if (runs > 0) this.item("points", "marks", startAt, 1, [...ink, 0.9, 5, 0.012, 2, 0]);
    if (this.points > 0) this.item("points", "stress", 0, this.points, [0.1, 0.1, 0.12, 0.55, 1.2, 0.004, 0, 0]);
  }

  private fillScene(now: number) {
    const s = this.scene;
    const info = this.info!;
    const aspect = this.w / this.h;
    const contour = this.mode === "contour";
    const cam = contour ? this.contour : this.orbit;
    s.set(cam.viewProj(aspect), 0);
    s.set([...cam.eye(), 1], 16);
    const [[x0, x1], [y0, y1]] = info.view;
    s.set([x0, x1, y0, y1], 20);
    s.set([info.zrange[0], info.zrange[1], contour ? 0 : HEIGHT_SCALE, info.manifest.log_height ? 1 : 0], 24);
    s.set([-0.45, 0.85, 0.35, contour ? 0.85 : 0.42], 28); // key light from the upper left, ambient share
    s.set([this.w * this.dpr, this.h * this.dpr, this.dpr, now / 1000], 32);
    s.set([...this.theme.bg, this.theme.dark ? 1 : 0], 36);
    s.set([24, this.theme.dark ? 0.35 : contour ? 0.4 : 0.28, contour ? 0 : 0.22, 0], 40); // contour levels, strength, fog
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
