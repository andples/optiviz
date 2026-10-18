# 1. Three threads: UI, engine, render

Date: 2026-10-18 · Status: accepted (M0)

## Context

The UI must never stutter while an algorithm runs flat out, and drawing must not slow the algorithm down.

## Decision

- The **main thread** runs only the Svelte UI.
- The **engine worker** owns the WebAssembly engine (and, in the threaded build, its rayon pool of further
  workers). It steps runs and publishes frames.
- The **render worker** owns the canvas (`OffscreenCanvas`) and the GPU back-end, and draws the newest frame
  at the display's rate.
- Commands are small `postMessage`s; per-frame data goes through the frame transport (ADR 3).

The render host (`web/src/render/host.ts`) is plain code with a `handle(cmd)` entry point, so the same class
runs on the main thread when `OffscreenCanvas` is unavailable, or when the worker can't get a GPU context.

## Consequences

- Measured in M0: render-thread CPU per frame 0.1–0.5 ms, engine publish of a full frame 0.01 ms (trail) to
  1.1 ms (1 M points); the main thread does nothing per frame.
- Camera input is forwarded to the render worker as deltas; damping runs there.
- The main-thread fallback can block the UI under very heavy scenes (seen with 1 M points on a software GPU);
  acceptable for a fallback.
