# 2. Stable Rust with RUSTC_BOOTSTRAP for the threaded WebAssembly build

Date: 2026-10-18 · Status: accepted (M0)

## Context

WebAssembly threads need the standard library compiled with `+atomics,+bulk-memory`, which the prebuilt
`wasm32-unknown-unknown` std isn't. The usual answer is a pinned nightly toolchain with `-Z build-std`.

## Decision

Use **stable** Rust plus the `rust-src` component, and set `RUSTC_BOOTSTRAP=1` only for the threaded build
(`just wasm-threads`), which allows `-Z build-std=panic_abort,std` and the unstable `atomics` target feature.
The link arguments for shared memory are passed explicitly (`--shared-memory --import-memory --max-memory=4 GiB`
and the TLS exports); without them wasm-bindgen emitted unshared memory and the rayon workers failed with
"Memory could not be cloned".

Two builds ship: `pkg-st` (SIMD, single-threaded, works on any page) and `pkg-threads` (SIMD + atomics +
rayon, needs a cross-origin isolated page). The engine worker loads one at start-up.

wasm-bindgen-rayon's worker helper imports its package by directory (`import('../../..')`), which Vite can't
resolve; the justfile rewrites it to the module file after `wasm-bindgen`.

## Consequences

- No nightly download (~100 MB) and no second toolchain to keep in step.
- `RUSTC_BOOTSTRAP` is a supported-but-discouraged escape hatch; if a stable release breaks the threaded build,
  switch the recipe to a pinned nightly (the flags are the same).
