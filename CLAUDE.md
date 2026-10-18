# optiviz: handoff notes

- **What and why:** `DESIGN.md` (goals, UI direction, stack, architecture, perf targets, milestones).
- **This file:** current state, gotchas, what to do next.
- **Guide / milestone notes:** docs/guide.md (how to use the app), docs/milestones/m1.md (what M1 built).

Repo: private GitHub `andples/optiviz`, branch `main`. `optiviz` is a working name.

## Status (2026-10-18)

M0 and M1 are done (DESIGN.md §10 lists what M1 contains and what was deferred). Architecture in short:
crates/core has problems (with coefficients), optimizers (with pseudocode + tracing), the divergence monitor and
`Session` (runs, timeline, stops); crates/wasm wraps Session as `Engine` and lays out frames; web/ has the
engine worker, the render worker (WebGPU / WebGL2 back-ends drawing a list of items), and the Svelte UI around
`ui/state.svelte.ts`. Tests: `just test` (44), `just test-gpu` (4), `just e2e` (12).

Open: the owner's check in real browsers (Zen + Safari on the MacBook): WebGPU, fps (press **i**); then fill
DESIGN.md §7.4's support table. Next is M2.

## Gotchas

- Headless Chromium on this Linux box can't present WebGPU to a canvas (device lost on the first frame, even an
  empty clear). Browser tests therefore use `?backend=webgl2`; WebGPU is verified natively by `just test-gpu`
  (RTX 2060, Vulkan) and by hand in real browsers. Its WebGL2 fps (SwiftShader) mean nothing.
- The threaded WASM build uses `RUSTC_BOOTSTRAP=1 -Z build-std` on stable and explicit shared-memory link args
  (ADR 0002); without the link args wasm-bindgen emits unshared memory ("Memory could not be cloned").
- `just wasm-threads` patches wasm-bindgen-rayon's `import('../../..')` (Vite can't resolve directory imports).
- Svelte `$state` proxies can't be posted to workers: use `$state.snapshot`.
- Adding an optimizer: implement `Optimizer` in crates/core/src/optimizers, register it in `IDS`/`by_id`, give it
  cited pseudocode, `tr!` calls with its line indices, and deviations; the generic tests in optimizers/mod.rs then
  check it. The UI needs nothing.
- svelte-check 4.7 needs TypeScript 6 (not 7).
- Previews: run on localhost and tell the owner the port.
  `just dev` serves :5173; `pnpm preview` serves the production build on :4173.

## Direction from the owner (keep to it)

- The view (surface) is the screen. Minimal chrome, nothing that isn't really needed.
- Scientific/mathematical feel (typeset maths, figure-quality visuals), not a terminal/developer look.
- Mouse/touch and keyboard both first class; keys are shortcuts, never required.
- Real scale: algorithms run at sizes people actually use, not toy demos.
- Learners and researchers served equally; where a decision can't serve both, split into two modes
  (DESIGN.md §2, record splits in docs/modes.md). Don't quietly favour one.
- Light theme by default, dark available.
- Must work in every modern browser (owner uses Zen on a MacBook): WebGPU + WebGL2 back-ends, threads optional
  (DESIGN.md §7.4).
- Families after 2D are not chosen yet; DESIGN.md §3 is the menu. Ask before starting one.
- Pseudocode must match the cited literature; every difference in the code is listed with its reason
  (DESIGN.md §4.1). The tool is also a step debugger (§4.2). Divergence handling: §4.3 / ADR 0006.
- Problems have editable coefficients (owner's request); new problems must expose theirs too.
- The site's ASCII surface (`~/Projects/site`, `S`) is inspiration for the idea only; share no code or look.

## Next

M2 (DESIGN.md §10 lists the eight steps in order): real-browser check, basins map on the GPU, many-start mode,
user-typed functions, the rest of the debugger (what-if branches etc.), sweeps, schedules, first presets.
