# 5. GPU kernels tested natively against the Rust reference

Date: 2026-10-18 · Status: accepted (M0)

## Decision

`crates/gpu-test` runs the WGSL from `/shaders` (and test kernels in `crates/gpu-test/kernels`) through native
wgpu: Vulkan locally (RTX 2060), lavapipe (software Vulkan) in CI with `OPTIVIZ_REQUIRE_GPU=1` and timing
sections off (`OPTIVIZ_PERF=0`). Tests: shader expressions vs `Problem::eval`; render pipelines build for every
problem; offscreen surface renders cover the frame and aren't flat (images in `target/gpu-test/`); the Adam
batch kernel (basins-map spike) ends in the same basin as the f64 Rust run.

## Consequences

Every GPU path has a CPU reference and a tolerance. New GPU algorithms follow the same pattern.
