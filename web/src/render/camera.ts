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
