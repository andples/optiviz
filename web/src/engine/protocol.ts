/** Messages between the main thread, the engine worker and the render worker. All small and infrequent; the
 *  per-frame data goes through frames.ts instead. Manifest shapes mirror crates/core/src/manifest.rs. */

import type { TransportInit } from "./frames";

export interface ParamSpec {
  key: string;
  symbol: string;
  description: string;
  default: number;
  min: number;
  max: number;
  log: boolean;
  /** discrete choice labels (value = index); empty for continuous */
  choices: string[];
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
  f_min: number | null;
  blurb: string;
  params: ParamSpec[];
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

/** The current problem as the engine has it: coefficients applied, the drawn box and height range. */
export interface ProblemInfo {
  manifest: ProblemManifest;
  coeffs: Record<string, number>;
  view: [[number, number], [number, number]];
  zrange: [number, number];
  start: number[];
}

export type Health = "ok" | "out-of-view" | "diverging" | "unbounded" | "diverged";
export interface Diagnosis {
  health: Health;
  text: string;
  suggest: [string, number] | null;
}
export interface RunStatus {
  opt: string;
  params: Record<string, number>;
  health: Health;
  halted: boolean;
  f: number;
  gradNorm: number;
  x: number[];
  evals: number;
  diagnosis: Diagnosis | null;
  /** starting points (walkers) in this run, how many failed / converged now, restarts so far */
  walkers: number;
  failed: number;
  converged: number;
  restarts: number;
  bestF: number;
  bestX: number[];
}

export type Stop =
  | { kind: "iteration"; iter: number }
  | { kind: "every"; iter: number }
  | { kind: "f-below"; run: number; f: number }
  | { kind: "grad-below"; run: number; grad: number }
  | { kind: "health"; run: number; health: Health }
  | { kind: "line"; run: number; line: number };

export interface Stops {
  at_iter?: number;
  every?: number;
  f_below?: number;
  grad_below?: number;
  on_diverge: boolean;
  line?: number;
}

export interface TraceEntry {
  line: number;
  vars: [string, number[]][];
}

/** Steps per second; 0 = as fast as possible */
export type Speed = number;

export type EngineCmd =
  | { type: "init"; transport: TransportInit; threads: boolean }
  | { type: "problem"; id: string; coeffs: Record<string, number>; keepStart: boolean }
  | { type: "addRun"; opt: string; params?: Record<string, number> }
  | { type: "removeRun"; i: number }
  | { type: "setOptimizer"; i: number; opt: string }
  | { type: "param"; i: number; key: string; value: number }
  | { type: "select"; i: number }
  | { type: "play" }
  | { type: "pause" }
  | { type: "step"; n: number }
  | { type: "restart" }
  | { type: "seek"; k: number }
  | { type: "speed"; speed: Speed }
  | { type: "stops"; stops: Stops }
  | { type: "multistart"; starts: number; onConverged: boolean; onFailed: boolean }
  | { type: "startUv"; u: number; v: number }
  | { type: "start"; x: number; y: number }
  | { type: "pickStart"; o: number[]; d: number[]; hs: number }
  | { type: "fitView" }
  | { type: "resetView" }
  | { type: "inspect"; i: number }
  | { type: "stress"; points: number };

export interface EngineStatus {
  iter: number;
  head: number;
  running: boolean;
  stepsPerSec: number;
  framesPerSec: number;
  stressPoints: number;
  /** mean time to fill + publish one frame, ms */
  publishMs: number;
  runs: RunStatus[];
}

export type EngineMsg =
  | { type: "ready"; threads: boolean; threadCount: number; problems: ProblemManifest[]; optimizers: OptimizerManifest[] }
  | { type: "problem"; info: ProblemInfo }
  | { type: "status"; status: EngineStatus }
  | { type: "metrics"; series: Float64Array[] }
  | { type: "stop"; stop: Stop }
  | { type: "trace"; i: number; iter: number; entries: TraceEntry[] }
  | { type: "error"; message: string };

export interface Theme {
  bg: [number, number, number];
  dark: boolean;
}

export type ViewMode = "surface" | "contour";

export type RenderCmd =
  | { type: "init"; canvas: OffscreenCanvas | HTMLCanvasElement; width: number; height: number; dpr: number; theme: Theme; prefer?: "webgpu" | "webgl2" }
  | { type: "transport"; transport: TransportInit }
  | { type: "problem"; info: ProblemInfo }
  | { type: "resize"; width: number; height: number; dpr: number }
  | { type: "orbit"; dx: number; dy: number }
  | { type: "zoom"; factor: number }
  | { type: "camera"; mode: "orbit" | "free" }
  | { type: "keys"; keys: { fwd: boolean; back: boolean; left: boolean; right: boolean; up: boolean; down: boolean; fast: boolean } }
  | { type: "view"; mode: ViewMode }
  | { type: "select"; i: number }
  | { type: "arrows"; on: boolean }
  | { type: "pick"; x: number; y: number }
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
  | { type: "picked"; uv?: [number, number]; ray?: { o: number[]; d: number[]; hs: number } }
  | { type: "stats"; stats: RenderStats };
