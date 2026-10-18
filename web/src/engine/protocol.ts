/** Messages between the main thread, the engine worker and the render worker. All small and infrequent; the
 *  per-frame data goes through frames.ts instead. */

import type { TransportInit } from "./frames";

export interface ParamSpec {
  key: string;
  symbol: string;
  description: string;
  default: number;
  min: number;
  max: number;
  log: boolean;
}
export interface ProblemManifest {
  id: string;
  name: string;
  tex: string;
  dim: number;
  bounds: [number, number][];
  minima: number[][];
  shader_expr: string | null;
  log_height: boolean;
  start: number[];
}
export interface Pseudocode {
  source: string;
  lines: { tex: string; indent: number; step: boolean }[];
  deviations: { line: number | null; what: string; why: string }[];
}
export interface OptimizerManifest {
  id: string;
  name: string;
  family: string;
  params: ParamSpec[];
  tex: string;
  pseudocode: Pseudocode;
}

/** Steps per second; 0 = as fast as possible */
export type Speed = number;

export type EngineCmd =
  | { type: "init"; transport: TransportInit; threads: boolean }
  | { type: "run"; problem: string; optimizer: string; x0?: [number, number]; seed: number; params?: Record<string, number> }
  | { type: "param"; key: string; value: number }
  | { type: "play" }
  | { type: "pause" }
  | { type: "step"; n: number }
  | { type: "restart"; x0?: [number, number] }
  | { type: "speed"; speed: Speed }
  | { type: "stress"; points: number }; // 0 turns the stress test off

export interface EngineStatus {
  iter: number;
  f: number;
  gradNorm: number;
  x: number;
  y: number;
  diverged: boolean;
  running: boolean;
  stepsPerSec: number;
  framesPerSec: number; // frames published per second
  stressPoints: number;
  /** mean time to fill + publish one frame, ms */
  publishMs: number;
}

export type EngineMsg =
  | { type: "ready"; threads: boolean; threadCount: number; problems: ProblemManifest[]; optimizers: OptimizerManifest[] }
  | { type: "problem"; manifest: ProblemManifest; zrange: [number, number] }
  | { type: "status"; status: EngineStatus }
  | { type: "error"; message: string };

export interface Theme {
  bg: [number, number, number];
  dark: boolean;
}

export type RenderCmd =
  | { type: "init"; canvas: OffscreenCanvas | HTMLCanvasElement; width: number; height: number; dpr: number; theme: Theme; prefer?: "webgpu" | "webgl2" }
  | { type: "transport"; transport: TransportInit }
  | { type: "problem"; manifest: ProblemManifest; zrange: [number, number] }
  | { type: "resize"; width: number; height: number; dpr: number }
  | { type: "orbit"; dx: number; dy: number }
  | { type: "zoom"; factor: number }
  | { type: "camera"; mode: "orbit" | "free" }
  | { type: "keys"; keys: { fwd: boolean; back: boolean; left: boolean; right: boolean; up: boolean; down: boolean; fast: boolean } }
  | { type: "theme"; theme: Theme };

export interface RenderStats {
  backend: "webgpu" | "webgl2";
  adapter: string;
  fps: number;
  /** CPU time per frame in the render thread, ms (mean over the last half second) */
  frameMs: number;
  /** frame age when drawn: draw time − engine publish time, ms */
  latencyMs: number;
  thread: "worker" | "main";
}

export type RenderMsg =
  | { type: "ready"; backend: "webgpu" | "webgl2"; adapter: string }
  | { type: "failed"; reason: string }
  | { type: "error"; message: string }
  | { type: "stats"; stats: RenderStats };
