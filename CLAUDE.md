# optiviz: handoff notes

- **What and why:** `DESIGN.md` (goals, UI direction, stack, architecture, perf targets, milestones).
- **This file:** current state, gotchas, what to do next.

Repo: private GitHub `andples/optiviz`, branch `main`. `optiviz` is a working name.

## Status (2026-10-18)

M0 (foundations and spikes) is done and committed: Rust core (6 problems, GD/momentum/Adam with sourced
pseudocode), two WASM builds (single-threaded and threaded), engine + render workers, SAB triple-buffer frames
with a postMessage fallback, WebGPU and WebGL2 back-ends sharing /shaders, a minimal light UI with KaTeX, native
GPU kernel tests, Playwright tests (6 pass), CI workflow (not yet run on GitHub), ADRs 0001–0005, perf note.
README has run/test commands. Next is M1 (DESIGN.md §10).

Open from M0: the owner's check in real browsers (Zen + Safari on the MacBook): does WebGPU come up (press **i**),
how many fps; then fill DESIGN.md §7.4's support table.

## Gotchas

- Headless Chromium on this Linux box can't present WebGPU to a canvas (device lost on the first frame, even an
  empty clear). Browser tests therefore use `?backend=webgl2`; WebGPU is verified natively by `just test-gpu`
  (RTX 2060, Vulkan) and by hand in real browsers. Its WebGL2 fps (SwiftShader) mean nothing.
- The threaded WASM build uses `RUSTC_BOOTSTRAP=1 -Z build-std` on stable and explicit shared-memory link args
  (ADR 0002); without the link args wasm-bindgen emits unshared memory ("Memory could not be cloned").
- `just wasm-threads` patches wasm-bindgen-rayon's `import('../../..')` (Vite can't resolve directory imports).
- Svelte `$state` proxies can't be posted to workers: use `$state.snapshot`.
- svelte-check 4.7 needs TypeScript 6 (not 7).
- Previews: run on localhost and tell the owner the port; their Mac reaches it through their tunnel.
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
  (DESIGN.md §4.1). The tool is also a step debugger (§4.2).
- The site's ASCII surface (`~/Projects/site`, `S`) is inspiration for the idea only; share no code or look.

## Next

M1 (DESIGN.md §10): the rest of the 2D optimizers with sourced pseudocode (§4.1), contour view, timeline and
history, charts, arena, URL sharing, click-to-start, and the step-debugger basics (§4.2).
