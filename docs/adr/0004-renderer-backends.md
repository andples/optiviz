# 4. WebGPU first, WebGL2 always; shaders evaluate f per pixel

Date: 2026-10-18 · Status: accepted (M0)

## Context

Every modern browser must work (DESIGN.md §7.4), but not all have WebGPU.

## Decision

- A small `Backend` interface (`web/src/render/backend.ts`): set problem, resize, upload a layer, draw.
  Two implementations: WebGPU (`webgpu.ts`, 4× MSAA) and WebGL2 (`webgl2.ts`). WebGPU is tried first.
- Shaders live in `/shaders` as WGSL and GLSL ES 3.00 twins with the same uniform layout (std140). The problem's
  `shader_expr` (from its Rust manifest, valid in both languages) is spliced in for `{{F}}`.
- The surface mesh only provides the silhouette. Height, normal (central differences), colour (viridis) and
  contour lines (`fwidth`) are computed per pixel from f, so shading is smooth at any zoom and mesh size.
- Projection maps depth to [0, 1]; the GLSL shaders remap to [-1, 1].

## Consequences

- crates/gpu-test compiles the same WGSL with native wgpu: shader expressions agree with the Rust f64 reference
  to ≤ 1.04·10⁻⁶ relative error on all six problems, all pipelines validate, and offscreen renders are checked
  and saved.
- Headless Chromium on Linux without a display can't present WebGPU to a canvas at all (the device is lost on the
  first frame, even for an empty clear), so browser tests use `?backend=webgl2`; WebGPU in real browsers is
  checked by hand (owner's MacBook: Zen, Safari).
- GLSL is hand-kept in step with WGSL for now; generating it with naga is an option if the shader count grows.
