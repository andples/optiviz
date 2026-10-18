# 6. What happens when a point diverges

Date: 2026-10-18 · Status: accepted (M1)

## Context

Optimizers run away: a step size above the stability limit, a function unbounded below (a saddle), adaptive
methods thrown out of a narrow valley, overflow to ±∞ or NaN. The app must neither crash nor silently draw
nonsense (lines to infinity, NaN markers), and it should help the person understand what happened.

## Decision

Every step feeds the run's `Monitor` (crates/core/src/monitor.rs), which classifies the run:

| Health | Rule | What happens |
|---|---|---|
| ok | | |
| out of view | x outside the drawn box, nothing else wrong | Marker drawn as a hollow ring pinned to the box edge; trail segments outside the box aren't drawn; run continues; "Zoom out to follow" offered |
| diverging | f has risen 1000·(1+\|f₀\|) above where it started, or x is 20 box-diameters away, or the step length grew ×1.2 or more for 8 steps in a row | A default stopping point: the session pauses once per run (until restart) with a diagnosis; marker gets a pulsing ring |
| unbounded | x far away, f falling steadily and far below f₀ | Same stop, but the diagnosis says the landscape has no bottom (not the optimizer's fault) |
| diverged | x, f or ∇f not finite | The run halts for good, keeping its last finite point (hollow ring there); other runs continue; always stops |

The **diagnosis** (`monitor::diagnose`) uses the largest Hessian eigenvalue L at the last finite point: for
gradient descent "stable only for η < 2/L, L ≈ …", for heavy ball / Nesterov η < 2(1+β)/L, for adaptive methods
"moves ≈ η per coordinate per step"; it suggests a value (half the limit, or η/4), which the notice offers as
"Restart with η = …". Other actions: zoom the view out to take in every trail (`Session::fit_view`), continue
anyway, dismiss. Turning the stopping point off (Stops panel) lets runs diverge without pausing.

## Consequences

- Tests: a stable run stays ok; η = 0.11 on the κ = 20 bowl is caught within 200 steps and the suggestion is
  η = 0.05 (= ½·2/L); the saddle is reported as unbounded; a non-finite run halts while the others go on.
- The thresholds are heuristics. They are deliberately loose (false alarms are worse than a late warning,
  because the stop interrupts); revisit when the N-D families arrive.
