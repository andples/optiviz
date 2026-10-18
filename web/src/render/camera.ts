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

export class OrbitCamera {
  yaw = -0.75;
  pitch = 0.62;
  dist = 4.3;
  private goal = { yaw: this.yaw, pitch: this.pitch, dist: this.dist };

  orbit(dx: number, dy: number) {
    this.goal.yaw -= dx * 0.006;
    this.goal.pitch = Math.min(1.45, Math.max(0.08, this.goal.pitch + dy * 0.006));
  }
  zoom(factor: number) {
    this.goal.dist = Math.min(9, Math.max(1.4, this.goal.dist * factor));
  }
  /** Moves towards the goal; returns true while still moving. */
  update(dt: number): boolean {
    const k = 1 - Math.exp(-dt * 14);
    let moving = false;
    for (const key of ["yaw", "pitch", "dist"] as const) {
      const d = this.goal[key] - this[key];
      if (Math.abs(d) > 1e-5) moving = true;
      this[key] += d * k;
    }
    return moving;
  }
  eye(): number[] {
    const cp = Math.cos(this.pitch);
    return [this.dist * cp * Math.sin(this.yaw), this.dist * Math.sin(this.pitch), this.dist * cp * Math.cos(this.yaw)];
  }
  viewProj(aspect: number): Mat4 {
    return mul(perspective((32 * Math.PI) / 180, aspect, 0.05, 50), lookAt(this.eye(), [0, -0.05, 0], [0, 1, 0]));
  }
}
