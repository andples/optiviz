# Using optiviz

How to use the app as of M1. Everything is reachable by mouse or touch; keyboard shortcuts are optional.

## The screen

The landscape fills the window. Around it, small floating panels:

| Where | What |
|---|---|
| Top left | The problem and its formula. Click it to change problem or its coefficients. |
| Top right | Surface / Contour view · free camera · share link · light/dark · **i** (what your browser got, performance) |
| Right | The runs racing on the problem, and the selected run's parameters |
| Left (optional) | The selected run's algorithm: cited pseudocode, live values, where the code differs from the source |
| Bottom right | Convergence chart (**+** for ‖∇f‖ and step length too) |
| Bottom | Play / one iteration / one line / restart · speed · timeline · readout · Stops |

## Problems

Ten 2D test problems: Rosenbrock, Himmelblau, Rastrigin, Ackley, Beale, an elliptic bowl, Styblinski–Tang, the
six-hump camel, a saddle, and random hills. Each has a one-line description of what makes it interesting.

**Coefficients.** In the problem panel every coefficient has a slider and an exact-value field (click the
number to type). The surface, gradients and known minima follow; when a change moves the minima, they are found
numerically and shown as rings. "textbook values" resets them. Examples worth trying: Rosenbrock's b (how steep
the valley walls are), the bowl's κ and rotation θ (Adam vs momentum on a rotated ellipse), the saddle's c
(negative turns it into a bowl), the hills' seed.

**Start point.** Click anywhere on the surface (or contour map) to start every run from there.

## Runs

Up to eight runs race in lockstep from the same start, each in its own colour. **+ add a run** picks an
optimizer; × removes one; click a run to select it (its trail is drawn on top and its parameters show below).

Optimizers: gradient descent (fixed step, backtracking or strong-Wolfe line search), momentum (heavy ball),
Nesterov, AdaGrad, RMSProp, Adam, AdamW, Newton's method (with Hessian modification), L-BFGS, Nelder–Mead (its
simplex is drawn). Parameters use the mathematical symbols (η, β₁, ρ, …) with a slider (log scale where it
makes sense) and an exact-value field; changes apply from the current iteration on.

## Several starting points and random restarts

At the top of the runs panel:

- **Starting points** (1, 4, 16, 64): every run starts from the clicked point plus random points in the view.
  All runs use the same random points, so the comparison stays fair. The random points show as small rings;
  each run's other walkers show as small discs in its colour.
- **Random restart** (never, converged, failed, both): a point that has converged (‖∇f‖ or its step
  effectively zero for 10 iterations) or failed (diverging, unbounded, not finite) jumps to a new random point
  and starts again. Each run keeps the best point any of its walkers found, shown as a ring in its colour; the
  chart then plots the best value so far.

Under each run: how many points, restarts so far, how many are failing, and the best f. With several points or
restarts on failure, divergence doesn't pause the run (it would interrupt constantly); failing points are
shown hollow and counted instead.

## Watching and stepping

- **Speed**: 10, 60 or 600 iterations per second, or as fast as the engine goes.
- **Timeline**: drag back to any earlier iteration; every run is replayed exactly. Changing a parameter while
  scrubbed back starts a new branch from there.
- **One iteration** (`.`) and **one line** (`,`): with the algorithm panel open, line stepping walks through the
  pseudocode lines the step executed, showing the values each computed (vectors as (a, b), Hessians as matrices).
  The surface shows the iterate after the step being walked through.
- While paused, an arrow at the selected run's marker shows the steepest-descent direction.

## Stopping points

**Stops** at the bottom opens the list: stop when a run starts diverging (on by default), when iteration k is
reached, every n iterations, when f(xₖ) or ‖∇f(xₖ)‖ falls below a value, or when the selected run executes a
chosen pseudocode line (click the line's number in the algorithm panel; click again to clear). When one fires,
a notice says which run and why, with Continue and One step.

## When a run diverges

Each run is watched every step (details: [ADR 0006](adr/0006-divergence.md)):

- **Outside the view**: the marker becomes a hollow ring pinned to the edge; **Zoom out to follow** enlarges the
  drawn area to take in every trail (**Reset view** goes back).
- **Diverging**: the run pauses and the notice explains why from the local curvature, e.g. "gradient descent is
  stable only for η < 2/L; here L ≈ 20, so η must be below 0.1", with **Restart with η = …**.
- **Unbounded below**: f keeps falling with no bottom (the saddle): the landscape's fault, not the optimizer's.
- **Diverged**: the numbers stopped being finite; that run halts at its last finite point, the others go on.

## Cameras

Surface view: drag to orbit, scroll or pinch to zoom. The arrow keys switch to a **free camera** (↑↓ forward and
back, ←→ sideways, Page Up/Down rise and fall, Shift faster, drag to look); Esc or the chip returns to orbit.
Contour view: drag to pan, scroll to zoom.

## Sharing

The share button copies a link that recreates the setup: problem, coefficients, start, runs with their
parameters, and the view.

## Keys

space play/pause · `.` one iteration · `,` one line · `r` restart · arrows free camera · Esc back / close.

## URL parameters (testing)

`?backend=webgl2` skip WebGPU · `?render=main` draw on the main thread · `?shared=0` no SharedArrayBuffer
(message-passing frames, one engine thread) · `?stress=1000000` a million extra points per frame.
