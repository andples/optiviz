# 3. Frame transport: SharedArrayBuffer triple buffer, postMessage fallback

Date: 2026-10-18 · Status: accepted (M0)

## Context

The engine produces state much faster than the screen shows it, and neither side should ever wait.

## Decision

`web/src/engine/frames.ts`: a frame is a Float64 header (scalars) plus a Float32 data area (trail, points).

- **Triple buffer** over one `SharedArrayBuffer`: three slots; the writer fills its own, publishes with one
  `Atomics.exchange` (setting a "fresh" bit), the reader picks up with one exchange when the bit is set. Lock-free,
  writer never blocks, reader always gets the newest complete frame.
- **Fallback** without cross-origin isolation: the same frames as transferable `ArrayBuffer`s over a
  `MessagePort`, recycled through a pool.

## Consequences

Measured (Chromium, 12-core desktop): publishing a 1 M-point frame (8 MB) costs 1.1 ms with the triple buffer
and 4.2 ms with message passing; a normal 2D run's frame (≤ 8 200 trail points) costs ~0.01 ms. Frame age at
draw time is dominated by the display interval, not the transport.

The engine copies from WebAssembly memory into the slot (one `set` per frame). Rendering straight from the
shared WebAssembly memory would save that copy, but couples the renderer to the engine's memory layout; revisit
if a workload needs it.
