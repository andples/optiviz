# optiviz

Watch optimization algorithms work, at the scale they are really run: lit 3D loss surfaces, swarms, big TSP
tours and real neural-network training, rendered with WebGPU (WebGL2 fallback) and computed in Rust/WebAssembly.

**Status:** M1 done: the 2D continuous experience. Ten problems with editable coefficients, ten optimizers
with cited pseudocode, up to eight runs racing on a lit surface or contour map (each from up to 64 starting points, with optional
random restarts), a replayable timeline, a step
debugger (line-by-line stepping, stopping points, the state at each line), divergence diagnosis, share links.
How to use it: [docs/guide.md](docs/guide.md); what M1 contains: [docs/milestones/m1.md](docs/milestones/m1.md).
See [DESIGN.md](DESIGN.md) for goals, architecture and milestones (§10 has what's next), [docs/perf/m0.md](docs/perf/m0.md) for
measurements and [docs/adr](docs/adr) for decisions.

## Run it

Needs Rust (stable, with `rust-src` and the `wasm32-unknown-unknown` target; `rust-toolchain.toml` asks rustup for
them), [`just`](https://github.com/casey/just), `wasm-bindgen-cli` 0.2.129 (`cargo install wasm-bindgen-cli --version 0.2.129`),
Node 26 and pnpm.

```sh
cd web && pnpm install && cd ..
just dev        # builds both WASM engines, serves http://localhost:5173 (cross-origin isolated)
```

Useful URL parameters: `?backend=webgl2` (skip WebGPU), `?render=main` (render on the main thread),
`?shared=0` (no SharedArrayBuffer: message-passing frames, single-threaded engine), `?stress=1000000` (a million
extra points per frame). The **i** button shows what the browser got and live performance. Keys (all optional): space play/pause,
`.` one iteration, `,` one pseudocode line, `r` restart, arrows fly the free camera, Esc back.

## Test

```sh
just test       # Rust unit tests (core with and without threads, wasm crate)
just test-gpu   # WGSL from /shaders through native wgpu, against the Rust reference; prints PERF lines
just lint       # rustfmt, clippy, svelte-check
just e2e        # production build + Playwright (CHROMIUM=/usr/bin/chromium to use a system Chromium)
just bench      # native benchmarks; `cd web && pnpm bench` runs the same workloads in WASM
```

## Layout

```
crates/core      problems, optimizers (with sourced pseudocode), rng, multistart: the f64 reference
crates/wasm      wasm-bindgen bindings for the engine worker; `threads` feature adds the rayon pool
crates/bench     native benchmarks of the same workloads
crates/gpu-test  native wgpu harness for /shaders and the test kernels
shaders/         WGSL + GLSL twins, shared by the browser and gpu-test
web/             Svelte 5 UI, engine worker, render worker (WebGPU / WebGL2 back-ends), Playwright tests
docs/            ADRs and performance notes
```

## Security

See [docs/security.md](docs/security.md) for the audit and the headers hosting must send.
