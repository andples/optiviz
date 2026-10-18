/** An orbit camera around the surface's centre, with damped motion, and the few matrix helpers it needs.
 *  World space: the surface spans x, z ∈ [-1, 1], y is up. Matrices are column-major (WGSL/GLSL order). */

export type Mat4 = Float32Array;

export function perspective(fovy: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1 / Math.tan(fovy / 2);
  const m = new Float32Array(16);
  m[0] = f / aspect;
  m[5] = f;
  // depth to [0, 1] (WebGPU); the WebGL2 backend remaps z in its vertex shaders
  m[10] = far / (near - far);
  m[11] = -1;
  m[14] = (near * far) / (near - far);
  return m;
}

export function lookAt(eye: number[], target: number[], up: number[]): Mat4 {
  const z = norm(sub(eye, target));
  const x = norm(cross(up, z));
  const y = cross(z, x);
  return new Float32Array([
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -dot(x, eye), -dot(y, eye), -dot(z, eye), 1,
  ]);
}

export function mul(a: Mat4, b: Mat4): Mat4 {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++)
    for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
      o[c * 4 + r] = s;
    }
  return o;
}

const sub = (a: number[], b: number[]) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: number[], b: number[]) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a: number[]) => {
  const l = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
};

export type CameraMode = "orbit" | "free";
export interface MoveKeys { fwd: boolean; back: boolean; left: boolean; right: boolean; up: boolean; down: boolean; fast: boolean }
const NO_KEYS: MoveKeys = { fwd: false, back: false, left: false, right: false, up: false, down: false, fast: false };
const TARGET = [0, -0.05, 0];

/**
 * Two cameras in one: **orbit** circles the surface's centre (drag turns, wheel zooms); **free** flies anywhere
 * (arrow keys move along the view and sideways, Page Up / Page Down rise and fall, Shift is faster, drag looks
 * around, the wheel moves forward). Entering free mode starts exactly where the orbit camera is, looking the
 * same way, so the switch is seamless. All motion is damped.
 */
export class OrbitCamera {
  mode: CameraMode = "orbit";
  yaw = -0.75;
  pitch = 0.62;
  dist = 4.3;
  private goal = { yaw: this.yaw, pitch: this.pitch, dist: this.dist };
  // free camera
  private pos = [0, 0, 0];
  private look = { yaw: 0, pitch: 0 };
  private lookGoal = { yaw: 0, pitch: 0 };
  private vel = [0, 0, 0];
  private keys: MoveKeys = { ...NO_KEYS };

  setMode(mode: CameraMode) {
    if (mode === this.mode) return;
    if (mode === "free") {
      this.pos = this.eye();
      const d = sub(TARGET, this.pos);
      const l = Math.hypot(d[0], d[1], d[2]);
      this.look.yaw = this.lookGoal.yaw = Math.atan2(d[0], d[2]);
      this.look.pitch = this.lookGoal.pitch = Math.asin(d[1] / l);
      this.vel = [0, 0, 0];
    } else this.keys = { ...NO_KEYS };
    this.mode = mode;
  }
  setKeys(keys: MoveKeys) {
    this.keys = keys;
  }
  /** Pointer drag: orbit turns around the centre, free looks around. */
  orbit(dx: number, dy: number) {
    if (this.mode === "free") {
      this.lookGoal.yaw -= dx * 0.004;
      this.lookGoal.pitch = Math.min(1.5, Math.max(-1.5, this.lookGoal.pitch - dy * 0.004));
      return;
    }
    this.goal.yaw -= dx * 0.006;
    this.goal.pitch = Math.min(1.45, Math.max(0.08, this.goal.pitch + dy * 0.006));
  }
  zoom(factor: number) {
    if (this.mode === "free") {
      const d = this.dir();
      const step = (1 - factor) * 2.5;
      this.pos = this.pos.map((p, i) => p + d[i] * step);
      return;
    }
    this.goal.dist = Math.min(9, Math.max(1.4, this.goal.dist * factor));
  }
  private dir() {
    const cp = Math.cos(this.look.pitch);
    return [cp * Math.sin(this.look.yaw), Math.sin(this.look.pitch), cp * Math.cos(this.look.yaw)];
  }
  /** Moves towards the goal; returns true while still moving. */
  update(dt: number): boolean {
    const k = 1 - Math.exp(-dt * 14);
    let moving = false;
    if (this.mode === "free") {
      for (const key of ["yaw", "pitch"] as const) {
        const d = this.lookGoal[key] - this.look[key];
        if (Math.abs(d) > 1e-5) moving = true;
        this.look[key] += d * k;
      }
      const f = this.dir();
      const right = norm(cross(f, [0, 1, 0]));
      const kk = this.keys;
      const speed = kk.fast ? 3.0 : 1.0;
      const want = [0, 0, 0];
      const add = (v: number[], s: number) => v.forEach((c, i) => (want[i] += c * s * speed));
      if (kk.fwd) add(f, 1);
      if (kk.back) add(f, -1);
      if (kk.right) add(right, 1);
      if (kk.left) add(right, -1);
      if (kk.up) add([0, 1, 0], 1);
      if (kk.down) add([0, 1, 0], -1);
      const kv = 1 - Math.exp(-dt * 10);
      for (let i = 0; i < 3; i++) {
        this.vel[i] += (want[i] - this.vel[i]) * kv;
        this.pos[i] += this.vel[i] * dt;
      }
      if (Math.hypot(this.vel[0], this.vel[1], this.vel[2]) > 1e-4) moving = true;
      return moving;
    }
    for (const key of ["yaw", "pitch", "dist"] as const) {
      const d = this.goal[key] - this[key];
      if (Math.abs(d) > 1e-5) moving = true;
      this[key] += d * k;
    }
    return moving;
  }
  /** Camera position */
  eye(): number[] {
    if (this.mode === "free") return [...this.pos];
    const cp = Math.cos(this.pitch);
    return [this.dist * cp * Math.sin(this.yaw), this.dist * Math.sin(this.pitch), this.dist * cp * Math.cos(this.yaw)];
  }
  viewProj(aspect: number): Mat4 {
    const proj = perspective((32 * Math.PI) / 180, aspect, 0.02, 50);
    if (this.mode === "free") {
      const d = this.dir();
      return mul(proj, lookAt(this.pos, [this.pos[0] + d[0], this.pos[1] + d[1], this.pos[2] + d[2]], [0, 1, 0]));
    }
    return mul(proj, lookAt(this.eye(), TARGET, [0, 1, 0]));
  }
}

/** General 4×4 inverse (column-major); null if singular. */
export function invert(m: Mat4): Mat4 | null {
  const a = m;
  const o = new Float32Array(16);
  const b00 = a[0] * a[5] - a[1] * a[4], b01 = a[0] * a[6] - a[2] * a[4], b02 = a[0] * a[7] - a[3] * a[4];
  const b03 = a[1] * a[6] - a[2] * a[5], b04 = a[1] * a[7] - a[3] * a[5], b05 = a[2] * a[7] - a[3] * a[6];
  const b06 = a[8] * a[13] - a[9] * a[12], b07 = a[8] * a[14] - a[10] * a[12], b08 = a[8] * a[15] - a[11] * a[12];
  const b09 = a[9] * a[14] - a[10] * a[13], b10 = a[9] * a[15] - a[11] * a[13], b11 = a[10] * a[15] - a[11] * a[14];
  const det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (!det) return null;
  const d = 1 / det;
  o[0] = (a[5] * b11 - a[6] * b10 + a[7] * b09) * d;
  o[1] = (a[2] * b10 - a[1] * b11 - a[3] * b09) * d;
  o[2] = (a[13] * b05 - a[14] * b04 + a[15] * b03) * d;
  o[3] = (a[10] * b04 - a[9] * b05 - a[11] * b03) * d;
  o[4] = (a[6] * b08 - a[4] * b11 - a[7] * b07) * d;
  o[5] = (a[0] * b11 - a[2] * b08 + a[3] * b07) * d;
  o[6] = (a[14] * b02 - a[12] * b05 - a[15] * b01) * d;
  o[7] = (a[8] * b05 - a[10] * b02 + a[11] * b01) * d;
  o[8] = (a[4] * b10 - a[5] * b08 + a[7] * b06) * d;
  o[9] = (a[1] * b08 - a[0] * b10 - a[3] * b06) * d;
  o[10] = (a[12] * b04 - a[13] * b02 + a[15] * b00) * d;
  o[11] = (a[9] * b02 - a[8] * b04 - a[11] * b00) * d;
  o[12] = (a[5] * b07 - a[4] * b09 - a[6] * b06) * d;
  o[13] = (a[0] * b09 - a[1] * b07 + a[2] * b06) * d;
  o[14] = (a[13] * b01 - a[12] * b03 - a[14] * b00) * d;
  o[15] = (a[8] * b03 - a[9] * b01 + a[10] * b00) * d;
  return o;
}

/** The world-space ray through a point in normalized device coordinates (depth 0..1, WebGPU convention). */
export function rayThrough(viewProj: Mat4, nx: number, ny: number): { o: number[]; d: number[] } | null {
  const inv = invert(viewProj);
  if (!inv) return null;
  const un = (z: number) => {
    const v = [nx, ny, z, 1];
    const r = [0, 1, 2, 3].map((i) => inv[i] * v[0] + inv[4 + i] * v[1] + inv[8 + i] * v[2] + inv[12 + i] * v[3]);
    return [r[0] / r[3], r[1] / r[3], r[2] / r[3]];
  };
  const a = un(0), b = un(1);
  return { o: a, d: norm(sub(b, a)) };
}

/**
 * The contour view's camera: straight down on the surface (screen up = problem y), orthographic, with pan and
 * zoom. Surface coordinates u, v ∈ [-1, 1] fill the shorter side at zoom 1.
 */
export class ContourCamera {
  cx = 0;
  cv = 0;
  half = 1.08;

  /** Drag in CSS pixels; `h` is the canvas height in CSS pixels. */
  pan(dx: number, dy: number, h: number) {
    const s = (2 * this.half) / h;
    this.cx -= dx * s;
    this.cv += dy * s;
  }
  zoom(factor: number) {
    this.half = Math.min(4, Math.max(0.05, this.half * factor));
  }
  reset() {
    this.cx = this.cv = 0;
    this.half = 1.08;
  }
  /** world: x = u, z = −v; looking down −y */
  viewProj(aspect: number): Mat4 {
    const hh = this.half, hw = this.half * aspect;
    const m = new Float32Array(16);
    // x_ndc = (u − cx)/hw; y_ndc = (v − cv)/hh = (−z − cv)/hh; depth from height (top of the box → 0)
    m[0] = 1 / hw;
    m[12] = -this.cx / hw;
    m[9] = -1 / hh;
    m[13] = -this.cv / hh;
    m[6] = -0.1;
    m[14] = 0.5;
    m[15] = 1;
    return m;
  }
  eye(): number[] {
    return [this.cx, 10, -this.cv];
  }
  /** Surface coordinates under a point in NDC. */
  uvAt(nx: number, ny: number, aspect: number): [number, number] {
    return [this.cx + nx * this.half * aspect, this.cv + ny * this.half];
  }
}
