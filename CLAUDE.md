# optiviz: handoff notes

- **What and why:** `DESIGN.md` (goals, UI direction, stack, architecture, perf targets, milestones M0–M6).
- **This file:** current state, gotchas, what to do next.

Repo: private GitHub `andples/optiviz`, branch `main`. `optiviz` is a working name.

## Status (2026-10-18)

Planning only. No code yet. Open questions for the owner are in DESIGN.md §13.

## Direction from the owner (keep to it)

- The view (surface) is the screen. Minimal chrome, nothing that isn't really needed.
- Scientific/mathematical feel (typeset maths, figure-quality visuals), not a terminal/developer look.
- Mouse/touch and keyboard both first class; keys are shortcuts, never required.
- Real scale: algorithms run at sizes people actually use, not toy demos.
- The site's ASCII surface (`~/Projects/site`, `S`) is inspiration for the idea only; share no code or look.

## Next

M0 spikes (DESIGN.md §10): WASM threads + SIMD, WebGPU render worker with OffscreenCanvas, SAB triple buffer,
wgpu native kernel tests. Write ADRs in `docs/adr/` from the results.
