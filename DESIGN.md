# optiviz: design and plan

A web app for **watching optimization algorithms work, at the scale they are really run.** You pick a problem
and one or more algorithms, press play, and see the search as it happens: a lit 3D loss surface with optimizers
moving across it, a swarm closing in on a basin, a 200 000-city tour untangling, CMA-ES's covariance ellipse
stretching along a valley, the central path of an interior point method crossing a polytope. The numbers
underneath (loss, gradient norm, step size, temperature, diversity) update live, and you can scrub back
through the run.

Status: **planning** (2026-10-18). Nothing is built yet. The working name `optiviz` is a placeholder.

---

## 1. Goals and non-goals

### Goals

1. **Real scale, not toy scale.** The engine is fast enough that what you see is the actual algorithm at a
   useful size: a million particles, a hundred thousand cities, a few thousand dimensions, a small neural network
   trained for real. Toy sizes are a setting, not the ceiling.
2. **Beautiful, legible visuals.** A loss landscape is drawn as a proper surface: shaded, lit, with contour
   lines, a sense of depth, smooth camera, crisp trajectories. Every view is chosen to show what the algorithm is
   *doing*, not decoration.
3. **Intuitive, minimal, scientific.** The surface (or whatever the main view is) fills the screen; everything
   else is quiet and only there when it matters. It should feel like a beautifully made figure from a maths
   paper that you can play with, not like a developer tool. Mouse/trackpad and keyboard both work fully;
   neither is required. The UI never stutters, however hard the engine is working.
4. **Expandable.** A new algorithm or problem is one file implementing a trait plus a small manifest. Its
   parameters and live state show up in the UI automatically. Adding the 40th optimizer should be as easy as the
   4th.
5. **Honest and reproducible.** Seeded, deterministic runs. Correct implementations checked against reference
   results. Each run can be shared as a link that reproduces it exactly.

### Non-goals (for now)

- Not a solver product. We don't compete with Gurobi, LKH or Optuna; we show how algorithms behave.
- No accounts and no backend. A static site; everything runs in the browser.
- Phones are not a first target. It should work on a tablet or phone, but design for a laptop or desktop with a
  real GPU.
- No f64 on the GPU (WGSL doesn't support it portably). GPU paths are f32; CPU paths are f64 by default.

### Relationship to the site's ASCII surface

`~/Projects/site` has a gradient-descent screensaver drawn in characters (`src/core/client/surface.ts`). It
sets the *vision*: watching an optimizer walk a landscape is mesmerizing and teaches intuition. optiviz takes
that idea much further: real rendering, many algorithm families, real scale, analysis tools. It is **not** a
port of it: no code is shared and it does not copy the site's terminal look (no ASCII, no vim-style UI, no
monospace chrome). What it keeps is the idea: a landscape you watch an algorithm explore, seeds you can share,
presets. A later, optional step could let the site link to optiviz.

---

## 2. Who it's for and what they do

| User | Wants | Key features |
|---|---|---|
| Learner | Build intuition for why Adam beats SGD here, why SA needs cooling, what a saddle point does | Step mode with pseudocode highlighting, guided presets, the 3D surface, explanations |
| Practitioner / researcher | Compare algorithms honestly on a problem; tune hyperparameters; see failure modes | Arena (side by side, same seed), sweeps, real-scale runs, metric charts, export |
| Me (the owner) | A showcase project that is technically impressive and nice to use | All of it, plus clean code and benchmarks |

The **primary target** (to confirm, see §14) is learner + practitioner: visual first, but with real scale and
rigour underneath.

---

## 3. Problem families and algorithms

The registry is open-ended. These are the families planned, roughly in build order. **Bold** = in the first
release of that family.

### 3.1 Continuous, low-dimensional (2D/3D landscapes): the flagship

- Problems: **Rosenbrock, Rastrigin, Ackley, Himmelblau, Beale, saddle, Styblinski-Tang, six-hump camel,
  random Gaussian hills (seeded)**, Goldstein-Price, Eggholder, Schwefel, Lévi, user-typed formula (§6.6),
  noisy versions (additive gradient noise, minibatch-like noise).
- First order: **GD, momentum (heavy ball), Nesterov, AdaGrad, RMSProp, Adam, AdamW**, AMSGrad, NAdam, Lion,
  Adafactor, Sophia-like, Shampoo (small), signSGD.
- Second order / quasi-Newton: **Newton (with damping), L-BFGS**, BFGS, conjugate gradient (FR/PR), trust region
  (dogleg), Gauss-Newton / Levenberg-Marquardt for least-squares problems.
- Derivative-free, local: **Nelder-Mead**, pattern search, Powell, coordinate descent.
- Line searches as their own pluggable piece: fixed step, backtracking Armijo, strong Wolfe.
- Schedules as pluggable pieces: constant, step, cosine, warmup, 1cycle.

### 3.2 Population and black-box methods

- **Random search, PSO, differential evolution, CMA-ES, simple GA, (1+1)-ES**, simulated annealing (continuous),
  natural evolution strategies, cross-entropy method, Bayesian optimization (GP + EI/UCB, small budgets), TPE.
- Runs on any continuous problem (2D for viewing, N-D for scale).

### 3.3 Combinatorial

- **TSP**: nearest neighbour, greedy edge, space-filling curve, Christofides-style (approx.); **2-opt, Or-opt**
  with neighbour lists and don't-look bits; **simulated annealing; ant colony (ACO)**; GA with order/EAX-like
  crossover; Or-3opt / Lin-Kernighan style moves later. Instances: random uniform/clustered, TSPLIB files,
  drawn by hand, images (stippling → tour art).
- Later: graph colouring (tabu, DSATUR), max-cut, knapsack (DP vs greedy vs B&B tree), SAT (WalkSAT, a small
  CDCL with the implication graph drawn), vehicle routing, job-shop scheduling (Gantt view), N-queens.

### 3.4 High-dimensional continuous and machine learning

- N-D test functions (Rosenbrock-N, Rastrigin-N, ill-conditioned quadratics with a chosen spectrum).
- **Train a small network for real** in the browser on the GPU: an MLP on two-spirals / moons / MNIST, a small
  CNN later. Compare optimizers on the same initialisation and data order.
- Views: loss curves, 2D projections of the trajectory (PCA), filter-normalised loss slices around the
  current point (Li et al. 2018), per-layer gradient and update norms, Hessian top eigenvalues via power
  iteration (sharpness), decision boundary for 2D datasets.

### 3.5 Constrained, linear and convex

- **LP**: simplex (vertex walk on a 2D/3D polytope), **interior point** (central path), with a tableau view.
- Projected gradient, penalty/barrier methods, augmented Lagrangian, ADMM, Frank-Wolfe, proximal gradient
  (lasso: sparsity appearing).
- Feasible region drawn on the surface; constraint violations visible.

### 3.6 Later

Multi-objective (NSGA-II, MOEA/D with a live Pareto front), hyperparameter search over the ML family, quantum-
inspired / QAOA toy, user-written algorithms in the browser (§6.7).

---

## 4. Views (what you see)

Views are generic over a problem's **kind**, not written per algorithm. An algorithm publishes state
**channels** (§6.2); a view draws whichever channels it understands.

| View | For | Detail |
|---|---|---|
| **Surface 3D** | 2D continuous | A lit, shaded heightfield evaluated on the GPU, with smooth normals, soft ambient occlusion, contour lines baked in the shader, a choice of colour ramps (perceptual: viridis/magma/cividis plus a theme ramp), log-height option, fog for depth. Optimizers as glowing markers with trails that sit on the surface; the gradient vector and step arrows as optional overlays. Orbit / free / follow cameras with smooth damping. |
| **Contour 2D** | 2D continuous | Top-down, infinitely sharp because the function is evaluated per pixel in the fragment shader; zoom without limit (f32 caveat: switch to a recentred coordinate frame when zoomed in deep). Iso-lines, gradient field streamlines, critical points marked. |
| **Basins map** | 2D continuous | Run the optimizer from *every pixel* at once on the GPU (1–4 M starts), colour each pixel by which minimum it reached and shade by steps taken. Shows the fractal-like basin boundaries of momentum and Adam. This is the "at scale" signature view. |
| **Swarm** | population methods | Particles as instanced points on the surface or contour view, velocity streaks, personal/global bests, CMA-ES covariance ellipses, DE difference vectors on hover. |
| **Tour** | TSP / routing | GPU line rendering of 100k+ segment tours with AA, the move being applied highlighted, ACO pheromone matrix as a heat overlay on candidate edges, the improvement curve. |
| **Projection** | N-D continuous, ML | PCA / random 2D slice of the trajectory, with the loss surface on that plane rendered as Surface 3D. |
| **Network** | ML | Decision boundary, weight histograms, per-layer norms. |
| **Polytope** | LP / constrained | 3D polytope with the simplex vertex path or the central path. |
| **Charts** | everything | Loss / best-so-far, gradient norm, step size, temperature, population diversity, constraint violation. uPlot-class speed (millions of points). Log axes. Every run in the arena on the same axes. |
| **Pseudocode** | everything | The algorithm's pseudocode with the current line highlighted in step mode, live variable values next to it. |

### Interaction model

- **Run controls:** play / pause, step one iteration, step one sub-step (a line search trial, one particle
  update), speed (iterations per frame up to "flat out"), restart with the same seed, new seed. All are buttons
  in the bottom bar; keys (space, arrows, `r`, `n`) are shortcuts, never the only way.
- **Timeline:** a scrubber along the bottom; drag back to any earlier iteration and the views show that state
  (history is recorded, §6.5). Branch: change a parameter at iteration 300 and continue from there as a new run.
- **Live tweaking:** change learning rate, temperature, population size, etc. while running; the change is
  marked on the timeline and on the charts.
- **Arena:** up to ~8 runs at once on the same problem and seed, drawn together (colour per run) or in a grid
  of synchronised views. Leaderboard in the side panel.
- **Sweep:** vary one or two parameters over a grid and show a heatmap of final loss / steps-to-tolerance
  (runs in parallel on CPU threads or GPU).
- **Place starts:** click the surface to drop a start point; drag a start and the whole trajectory re-runs
  live (feasible because 2D runs take microseconds).
- **Share:** the URL hash encodes problem, algorithms, parameters and seed (compact, versioned); a share button
  copies it.
- **Export:** PNG / video (WebCodecs → MP4/WebM) of a view, CSV/JSON of metrics, the run config.

---

## 5. UI and visual design

The guiding rule: **the view is the interface.** The surface takes the whole window; controls float over it,
small and translucent, and get out of the way. Nothing is on screen that isn't needed right now.

### Layout

```
┌───────────────────────────────────────────────────────────────────────────────┐
│ f(x,y) = (1−x)² + 100(y−x²)²  ▾                         Adam · L-BFGS  +   ⋯  │  thin top strip
│                                                                    ┌────────┐ │
│                                                                    │ Adam   │ │  inspector:
│                                                                    │ η 0.01 │ │  only when a run
│                    the surface, full window                        │ β₁ 0.9 │ │  is selected,
│               (or contour / basins / tour / ...)                   └────────┘ │  collapsible
│                                                                               │
│                                                      ┌───────────────────────┐│
│                                                      │ f(xₖ)   log scale ╲__ ││  one small chart,
│                                                      └───────────────────────┘│  expandable
│  ▶  ⏸  ⏭   ━━━━━━━━━━━━━━●━━━━━━━━━━━━━━━   k = 1 284      f = 3.2·10⁻⁷       │  thin bottom strip
└───────────────────────────────────────────────────────────────────────────────┘
```

- **Top strip:** the problem, written as its formula (typeset), which opens the problem picker; the runs on the
  scene, each with its colour, and `+` to add one; a `⋯` menu for views, export, settings.
- **Inspector:** appears when you click a run, shows its parameters with their mathematical symbols (η, β₁, T,
  σ, λ) and short descriptions on hover. Generated from each algorithm's parameter schema: sliders with log
  scales where sensible, segmented controls for choices. Collapsible to nothing.
- **Bottom strip:** play/pause/step, the timeline scrubber, the iteration count and the current value. One small
  convergence chart floats above it; click to expand into a full metrics panel (gradient norm, step size,
  temperature, diversity...), then collapse again.
- **Hide everything** (one button or `h`) for a clean figure, e.g. to take a screenshot or just watch.
- Picking problems and algorithms is a visual gallery (a thumbnail of each landscape, a one-line description and
  the formula), not a list of names. A search box at the top of it.
- **Both input styles are first class.** Everything is clickable, draggable, and works on a trackpad and touch
  (drag to orbit, pinch to zoom, two-finger pan). Keyboard shortcuts exist for frequent actions, shown in
  tooltips and on `?`, but are never required. A search-everything box (`⌘K` or the search icon) helps power
  users without being the main path.

### Visual language: a scientific figure, alive

- **Mathematical typography.** Formulas, parameters and values are typeset properly with KaTeX: `f(x, y)`,
  `∇f`, `η`, `‖∇f(xₖ)‖`, numbers in scientific notation (`3.2·10⁻⁷`) rather than `3.2e-07`. A clean
  humanist/neo-grotesque sans for UI text, a math font (KaTeX's Latin Modern/Computer Modern) for maths. Tabular
  figures so live numbers don't jitter.
- **Surfaces like good figures.** Soft studio lighting, perceptually uniform colour maps (viridis, cividis,
  magma, a diverging map for signed functions) with an optional legend/colour bar, thin labelled contour lines,
  optional axes with tick labels in maths font, a subtle grid on the floor. It should look at home next to a
  figure in a textbook, only better.
- **Two themes, equally good:** a light "paper" theme (off-white background, dark ink, like a printed figure)
  and a dark theme for watching. All colours are tokens; run colours come from a validated categorical palette
  that holds up in both themes and for colour-blind viewers.
- **Quiet chrome:** translucent floating panels, thin rules, small type, no heavy cards or toolbars. The canvas
  is the hero.
- **Purposeful motion:** camera damping, trails fading, no gratuitous animation. Respects
  `prefers-reduced-motion`.

---

## 6. Architecture

### 6.1 The shape of it

```
 main thread (UI only)           engine worker                      render worker
 ┌──────────────────┐   cmds    ┌─────────────────────────┐        ┌───────────────────────────┐
 │ Svelte 5 UI      │ ────────▶ │ Rust core (WASM, SIMD)  │        │ OffscreenCanvas + WebGPU  │
 │ panels, palette, │           │ rayon thread pool       │ SAB    │ views, shaders            │
 │ inspector,charts │ ◀──────── │ problems, optimizers    │ ─────▶ │ GPU compute kernels       │
 │ (no heavy work)  │  metrics  │ history recorder        │ frames │ (basins, swarm, NN, eval) │
 └──────────────────┘           └─────────────────────────┘        └───────────────────────────┘
                 ▲                                                       │ input events forwarded
                 └───────────────────────────────────────────────────────┘
```

- **Three kinds of thread, none blocking another.** The main thread only runs the UI, so it stays at 60–120 fps
  however hard the engine works. The engine worker runs the algorithms in Rust/WASM with its own thread
  pool. The render worker owns the canvas (`OffscreenCanvas`) and the WebGPU device, draws the views and also
  runs GPU compute kernels, so GPU results never leave the GPU when they only need to be drawn.
- **Simulation and drawing are decoupled.** The engine runs as fast as it can (or at a requested rate) and
  publishes state into shared memory; the render worker draws the latest state at the display's frame rate.
  "Flat out" mode might do 100 000 iterations between two frames; step mode does one.
- **Zero-copy hand-off.** State goes through `SharedArrayBuffer` triple buffers (writer never waits, reader
  always gets a complete, recent frame) with a sequence number per buffer. Metrics go through an SPSC ring
  buffer. Commands go the other way via `postMessage` (they're small and rare).
- **GPU-native algorithms** (basins map, big swarms, NN training, million-point landscape evaluation) run in the
  render worker as WGSL compute shaders; the engine worker coordinates them through the same command channel and
  reads back only small summaries (best value, metrics) asynchronously.

### 6.2 The plugin model

Everything is defined in Rust (types, maths, state), described to the UI by a manifest, and drawn by generic
views.

```rust
/// What a problem is: its kind decides which views can draw it.
pub trait Problem {
    type Point;                          // e.g. [f64; 2], Vec<f64>, Permutation, BitVec
    fn manifest(&self) -> ProblemManifest;  // name, kind, dimension, bounds, known optima, params schema
    fn eval(&self, x: &Self::Point) -> f64;
    fn grad(&self, x: &Self::Point, g: &mut Self::Point) -> f64 { /* default: autodiff or finite differences */ }
}

/// An optimizer is a state machine stepped one iteration (or one sub-step) at a time.
pub trait Optimizer<P: Problem> {
    fn manifest() -> OptimizerManifest;  // name, family, params schema, channels it publishes, pseudocode
    fn init(&mut self, p: &P, rng: &mut Rng);
    fn step(&mut self, p: &P, rng: &mut Rng) -> StepInfo;   // StepInfo: evals used, sub-step / line marker
    fn publish(&self, out: &mut ChannelWriter);             // write channels into the shared frame
    fn set_param(&mut self, key: &str, v: ParamValue);      // live tweaking
    fn snapshot(&self) -> Snapshot; fn restore(&mut self, s: &Snapshot);  // timeline scrubbing, branching
}
```

- **Params schema** (typed: float with range and log flag, int, enum, bool) → the inspector is generated.
- **Channels** are typed, named arrays: `positions: f32[N×D]`, `best: f32[D]`, `velocity`, `covariance: f32[D×D]`,
  `tour: u32[N]`, `pheromone: f32[E]`, scalars like `loss`, `grad_norm`, `temperature`. Views declare which
  channel names and shapes they draw. A new optimizer that publishes `positions` gets the swarm view for free.
- **Pseudocode** is part of the manifest, with line ids; `StepInfo` says which line just ran.
- Rust → TypeScript types are generated (`ts-rs` or `specta`) so manifests and channels are typed on both
  sides and can't drift.
- A GPU-native algorithm implements the same manifest but its `step` dispatches WGSL kernels; the WGSL lives
  next to the Rust file.

### 6.3 Determinism and correctness

- One seeded PRNG per run (`rand_xoshiro`/PCG), derived per thread with jump-ahead so results don't depend on
  thread scheduling. Parallel reductions use a fixed order. Same seed + same config = same run, CPU path
  bit-identical across machines; GPU path identical on the same device and within tolerance across devices
  (float atomics and drivers differ; documented).
- Each algorithm has tests against known results: converges on convex quadratics at the theoretical rate,
  reaches known optima on standard functions, matches reference implementations (SciPy, `cma`, published TSPLIB
  tour lengths) on fixed seeds within tolerance.

### 6.4 Precision

CPU: f64 everywhere. GPU: f32. Surface rendering in f32 with a re-centred coordinate frame for deep zoom
(evaluate `f(c + δ)` with `c` in f64 on the CPU folded into constants). Views read f32 channels.

### 6.5 History (timeline scrubbing)

- Every iteration's small state (scalars, best point, a few positions) is kept in a compact ring buffer.
- Large state (a million particles, a tour of 200k cities) is kept as periodic **keyframes** (full snapshot
  every k iterations, adaptive to memory) plus **deterministic replay**: to show iteration 7 345, restore the
  nearest keyframe and re-run (fast, since runs are deterministic).
- A memory budget (default 512 MB) with the keyframe interval adapting to it.

### 6.6 User-typed functions

`f(x, y) = sin(x)*cos(y) + 0.1*(x^2+y^2)` → a small parser → expression graph → (a) WGSL source for the
surface/basins/GPU paths, (b) for the CPU, a generated tiny WebAssembly module compiled at runtime (fast, near
native), with forward-mode automatic differentiation generating the gradient code. A bytecode interpreter is
the fallback. Same graph also gives symbolic gradient display.

### 6.7 Later: user-written algorithms

An in-browser editor where you write an optimizer in TypeScript (or WGSL for the GPU) against a small API
(`step(state, f, grad, rng)`), hot-reloaded into the same views. Sandboxed in its own worker. This is what
makes the project a playground and not just a gallery.

---

## 7. Stack (the big decision)

### 7.1 Decision

| Layer | Choice | Why |
|---|---|---|
| Algorithm core | **Rust** (2024 edition), `no_std`-friendly core crate | Near-C speed, SIMD, real threads, great testing; same code runs native (tests, benchmarks, reference runs) and in the browser. The owner already ships Rust (ttyp, ctui). |
| CPU in browser | **WebAssembly** with `simd128` + **threads** (`wasm-bindgen-rayon` over `SharedArrayBuffer`) | 1.2–2× of native for numeric loops, multi-core. Needs nightly Rust for `-Z build-std` with `+atomics,+bulk-memory`, pinned in `rust-toolchain.toml`. |
| GPU | **WebGPU** (raw API from TypeScript) with **WGSL** shaders, compute + render | The only real way to get GPU compute in a browser; supported in Chrome/Edge, Safari 26+, and Firefox (Windows since 141; Linux/macOS rolling out; verify in M0). Raw API instead of three.js/Babylon: our views are custom (heightfields, instanced particles, huge line sets) and we want full control of buffers to avoid copies. |
| GPU testing | **wgpu** (native Rust) running the **same WGSL files** | Kernel unit tests on the CPU-side test runner and headless CI, compared against the Rust f64 reference. |
| UI | **Svelte 5** (runes) + **TypeScript**, built with **Vite** | Fine-grained reactivity without a virtual DOM, small bundles, the owner knows it (jigswap). Static SPA, no SvelteKit needed (no routes, no SSR). |
| Charts | **uPlot** (themed), or our own WebGPU line renderer for >1M points | uPlot draws a million points at 60 fps in canvas2D; we only go custom if it's the bottleneck. |
| Maths typesetting | **KaTeX** (formulas, parameter symbols, axis labels; rendered once, cached as textures for in-canvas labels) | Fast, no layout reflow, proper maths fonts. |
| Types across the boundary | `ts-rs`/`specta` generated TS types, `wasm-bindgen` | Manifests/channels typed on both sides. |
| Tooling | `pnpm`, `cargo`, `just`, `wasm-bindgen-cli` + `wasm-opt`, Biome (lint/format TS), clippy/rustfmt | Fast, simple. |
| Tests | `cargo test` + `proptest`, `criterion` benches, Vitest, Playwright (Chromium with WebGPU, screenshot tests) | |
| Hosting | Static files with `COOP: same-origin` + `COEP: require-corp` headers (needed for SharedArrayBuffer). Cloudflare Pages, or the owner's own server behind a tunnel like the site. | Decide at M6. |

### 7.2 Alternatives considered

- **All TypeScript + WebGPU.** Simplest build, one language. But CPU-bound algorithms (TSP local search, CMA-ES
  eigendecompositions, L-BFGS, BO with GPs) are 2–5× slower than WASM SIMD, no control over memory layout, and
  no native benchmarking or reference runs. Rejected for the core; TS stays for UI and GPU orchestration.
- **All Rust (Leptos/egui + wgpu in WASM).** One language end to end, but egui doesn't give the polished,
  accessible web UI we want; Leptos is fine but the DOM UI isn't where the speed matters, and sharing one
  `GPUDevice` between Rust (wgpu-in-WASM) and a separate render worker is awkward. Also larger bundles. Kept as
  a fallback if the TS/Rust split hurts.
- **C++ / Emscripten.** Mature, fast, but Rust's tooling, safety and testing are better for a long-lived project.
- **Zig / AssemblyScript.** Smaller ecosystems for numerics; rejected.
- **Pyodide / JAX-in-browser.** Huge downloads, slow start, not real-time. Rejected.
- **three.js / Babylon.js for rendering.** Fast to start, but generic scene graphs fight our data layout
  (we want to render straight from compute buffers). Rejected; we may borrow ideas (camera controls, tone
  mapping).
- **React / Solid.** Solid is as good as Svelte for speed; React's re-render model is a poor fit for a UI
  showing live numbers. Svelte chosen for familiarity.

### 7.3 Risks in the stack (to retire in M0)

1. WASM threads need nightly Rust + `build-std`, and COOP/COEP headers (which block third-party embeds without
   CORP headers). → Spike in M0; fall back to one WASM instance per worker with manual work splitting if
   rayon is painful.
2. WebGPU in a worker with `OffscreenCanvas` across browsers (Safari, Firefox Linux). → Spike; fallback is
   rendering on the main thread (engine still in a worker).
3. Browsers without WebGPU. → v1 shows a clear message; CPU algorithms still run and charts still work. A WebGL2
   fallback for the surface and contour views is a later milestone if needed.
4. GPU timeouts (TDR) on long compute dispatches. → Chunk dispatches (e.g. 64 iterations per dispatch for the
   basins map), measure with timestamp queries.
5. Readback latency (`mapAsync`). → Never await readbacks in the frame loop; double-buffer staging buffers.

---

## 8. Performance targets

Measured on the owner's desktop GPU and on an integrated laptop GPU (the "low" column). These are the
contract; the bench suite tracks them.

| Scenario | Target (desktop) | Target (low) |
|---|---|---|
| UI frame time while engine runs flat out | < 4 ms main thread, never drop frames | < 8 ms |
| Surface 3D at 4K, 2048² heightfield, contours, AO | 120 fps | 60 fps at 1080p |
| 2D optimizer, flat out | > 10 M iterations/s (GD on Rosenbrock, one core) | > 3 M |
| Drag a start point, whole 2 000-step trajectory re-run | < 1 ms | < 3 ms |
| Basins map, 2048² starts × 500 Adam steps | < 0.5 s | < 3 s |
| PSO, 1 M particles, 30-D Rastrigin (GPU) | > 60 iterations/s | > 10 |
| CMA-ES, 1 000-D (CPU threads) | > 50 generations/s | > 10 |
| TSP 100k cities: greedy + 2-opt/Or-opt to < 6% above best known | < 10 s in the browser | < 30 s |
| MLP 2-layer (784-256-10) on MNIST, batch 256 (GPU) | > 2 000 steps/s | > 300 |
| Cold load to first frame | < 1.5 s on broadband, < 1 MB JS+WASM gzipped for the shell | |

Native vs WASM parity is tracked: the same `criterion` bench is run natively and in headless Chromium, and the
ratio must stay under ~1.5×.

---

## 9. Repository layout (planned)

```
optiviz/
├── DESIGN.md  README.md  CLAUDE.md  justfile  rust-toolchain.toml  Cargo.toml (workspace)
├── crates/
│   ├── core/        # traits, manifests, channels, rng, problems/, optimizers/ (f64, no I/O)
│   ├── expr/        # formula parser, autodiff, WGSL + WASM codegen (§6.6)
│   ├── wasm/        # wasm-bindgen bindings, rayon pool, SAB channel writer, history recorder
│   ├── gpu-test/    # wgpu harness running shaders/*.wgsl against core's reference
│   └── bench/       # native CLI: run any config headless, criterion benches, reference outputs
├── shaders/         # *.wgsl (render + compute), shared by web and gpu-test
├── web/
│   ├── src/ui/      # Svelte components: shell, palette, inspector, timeline, charts
│   ├── src/engine/  # engine worker bootstrap, command protocol, SAB ring/triple buffers
│   ├── src/render/  # render worker, WebGPU device, views/, cameras, GPU algorithms/
│   └── src/gen/     # generated TS types from Rust
└── docs/            # adr/ (architecture decisions), algorithm notes, perf reports
```

---

## 10. Milestones

Each milestone ends with something that runs and is committed; benches and tests land with the code.

### M0: Foundations and spikes (retire the risks)
- Workspace, toolchain pinning, `just` recipes, CI (fmt, clippy, tests, wasm build, Playwright smoke).
- Spike: Rust → WASM with SIMD + rayon threads in a worker, COOP/COEP dev server, measured native vs WASM.
- Spike: render worker with OffscreenCanvas + WebGPU drawing a lit heightfield; check Chrome, Firefox, Safari.
- Spike: SAB triple buffer from engine worker → render worker, 1M points/frame, no stalls.
- Spike: wgpu native running a WGSL kernel and matching the Rust reference.
- Write ADRs for the results. **Exit:** a heightfield with one Adam marker moving on it, engine in WASM,
  render in a worker, 120 fps, numbers in a perf note.

### M1: The flagship vertical slice (2D continuous)
- Problem + optimizer traits, manifests, channels, generated TS types.
- 8 test functions, 7 first-order optimizers + L-BFGS + Newton + Nelder-Mead, line searches, schedules.
- Surface 3D (lighting, AO, contours, ramps, cameras) and Contour 2D views, trails, gradient arrows.
- Minimal shell: formula strip + problem gallery, generated inspector, timeline with scrubbing, the small
  chart and its expanded panel, light and dark themes, KaTeX typesetting, mouse/touch and keyboard. Live
  parameter tweaks. Arena with several runs. URL sharing. Click/drag start points.
- **Exit:** feels finished for 2D continuous; someone can learn the difference between Adam and momentum in
  two minutes.

### M2: Scale and populations
- Basins map on the GPU. Many-start mode.
- PSO, DE, CMA-ES, GA, (1+1)-ES, SA, random search on CPU (threads) and the big ones on GPU.
- Swarm view, covariance ellipses. N-D problems with projection view.
- User-typed functions (§6.6).
- Sweeps (parameter heatmaps).

### M3: Combinatorial (TSP first)
- Permutation problem kind, TSP instances (random, clustered, TSPLIB loader, hand-drawn, image stippling).
- Constructive heuristics, 2-opt/Or-opt with neighbour lists (k-d tree), SA, ACO, GA.
- Tour view with GPU lines for 100k+ cities, move highlighting, pheromone overlay.
- History with keyframes + deterministic replay for big state.

### M4: Machine learning
- GPU kernels for MLP forward/backward (tiled matmul in WGSL), datasets (spirals, moons, MNIST lazily fetched).
- Optimizer comparison on real training, loss slices, PCA trajectory, sharpness, decision boundary.

### M5: Constrained and LP
- Simplex with tableau and polytope view, interior point central path, projected gradient, barrier, ADMM,
  proximal methods.

### M6: Polish and ship
- Step mode with pseudocode highlighting everywhere, guided presets ("why Adam overshoots", "SA cooling too
  fast"), explanations per algorithm.
- Video/PNG export, accessibility pass, light theme polish, tablet layout, WebGL2 fallback if needed.
- Deploy with COOP/COEP; decide public vs private, domain, and whether the site links to it.

### Later
More combinatorial problems (SAT, colouring, VRP, scheduling), multi-objective, user-written algorithms,
Bayesian optimization at scale, embeddable scenes (`<optiviz-scene>` web component) for blog posts.

---

## 11. Testing strategy

- **Rust unit + property tests** per problem (gradient matches finite differences; known optima) and per
  optimizer (convergence on quadratics, reference comparisons, determinism across thread counts).
- **Kernel tests**: every WGSL compute kernel run through wgpu and compared with the Rust f64 reference within
  stated tolerance.
- **Golden runs**: fixed config + seed → stored metrics trace; CI fails on drift.
- **Benches**: criterion natively, plus browser benches in headless Chromium; perf targets in §8 checked and
  trends recorded.
- **UI**: Vitest for logic, Playwright e2e (load, pick, run, scrub, share link round trip) and screenshot
  diffs of each view on a fixed seed.

---

## 12. Conventions

- Commit after each working step; small, descriptive commits.
- Every algorithm file carries: a short description, references (paper/book), the pseudocode in the manifest,
  and its tests next to it.
- No dependency added without a reason noted in the commit; keep the shell bundle small.
- ADRs in `docs/adr/` for decisions that would be expensive to reverse.

---

## 13. Open questions for the owner

1. **Audience emphasis**: the plan now assumes a mathematical/scientific reader first (proper notation,
   figure-quality visuals), with the scale and rigour a practitioner needs underneath. Is learner vs
   researcher still worth weighting one way?
2. **Which families matter most** after 2D continuous? The plan's order: populations → TSP → ML → LP.
3. **Public or private** when it ships, and on which domain? (Repo is private for now.)
4. **Name**: `optiviz` is a placeholder.
5. **Firefox on Linux**: is it your main browser? If so WebGPU support there decides a lot in M0 (Chromium is
   the safe target).
6. **Default theme**: light "paper" or dark? Both will exist.
7. **Site integration**: should the site's `S` surface ever hand off to optiviz ("open this landscape in
   optiviz"), or stay separate?
