/** The small interface both GPU back-ends implement; views are written once against it (DESIGN.md §7.4). */

export type Layer = "trail" | "points" | "marker" | "minima";

export interface Backend {
  readonly kind: "webgpu" | "webgl2";
  readonly adapter: string;
  /** Rebuilds the shaders for a new f(x, y) and mesh resolution. */
  setProblem(expr: string, meshN: number): void;
  resize(width: number, height: number): void;
  /** Uploads `count` points (u, v pairs) to a layer. */
  upload(layer: Layer, data: Float32Array, count: number): void;
  /** Draws the scene. `scene` has SCENE_FLOATS values; `draws` gives each layer's Draw uniforms (DRAW_FLOATS). */
  draw(scene: Float32Array, draws: Record<Layer, Float32Array>, bg: [number, number, number]): void;
}

export const LAYER_CAP: Record<Layer, number> = { trail: 8200, points: 1 << 20, marker: 4, minima: 16 };
