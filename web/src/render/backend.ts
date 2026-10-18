/** The small interface both GPU back-ends implement; views are written once against it (DESIGN.md §7.4).
 *  The host uploads point buffers, then hands over a list of draws for the frame. */

/** data: trails, shapes, minima (the engine's frame); marks: markers and start points; stress: test swarm */
export type BufferId = "data" | "marks" | "stress";

export interface DrawItem {
  kind: "trail" | "points";
  buffer: BufferId;
  /** first point in the buffer */
  first: number;
  /** points (a trail of n points draws n − 1 segments) */
  count: number;
  /** the Draw uniforms (DRAW_FLOATS): color rgba; size px, lift, mode, count */
  uniforms: Float32Array;
}

export interface Backend {
  readonly kind: "webgpu" | "webgl2";
  readonly adapter: string;
  /** Rebuilds the shaders for a new f(x, y) and mesh resolution. */
  setProblem(expr: string, meshN: number): void;
  resize(width: number, height: number): void;
  upload(buffer: BufferId, data: Float32Array, count: number): void;
  /** Draws the surface, then the items in order. */
  draw(scene: Float32Array, items: DrawItem[], bg: [number, number, number]): void;
}

export const BUFFER_CAP: Record<BufferId, number> = { data: 131_072, marks: 64, stress: 1 << 20 };
/** Most draw items per frame (uniform slots) */
export const MAX_ITEMS = 96;
