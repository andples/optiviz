//! The engine worker's view of the core: a `Run` steps one optimizer on one problem and keeps the trail the
//! render worker draws; plus manifests as JSON and the M0 benchmark workloads.
//!
//! Frames are read straight out of WebAssembly memory (`trail_ptr`, `points_ptr`) by the worker, which copies
//! them into the shared triple buffer (web/src/engine/frames.ts). Scalars come from getters.

use optiviz_core::{Optimizer, Problem, Rng, StepInfo, multistart, optimizers, problems};
use wasm_bindgen::prelude::*;

#[cfg(feature = "threads")]
pub use wasm_bindgen_rayon::init_thread_pool;

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

/// True when built with the rayon thread pool (the page must still call `initThreadPool`).
#[wasm_bindgen]
pub fn threads_enabled() -> bool {
    cfg!(feature = "threads")
}

/// The 2D manifests of every problem, as a JSON array.
#[wasm_bindgen]
pub fn problems_json() -> String {
    let all: Vec<_> = problems::IDS.iter().map(|id| problems::by_id(id, 2).unwrap().manifest()).collect();
    serde_json::to_string(&all).unwrap()
}

/// Every optimizer manifest, as a JSON array.
#[wasm_bindgen]
pub fn optimizers_json() -> String {
    let all: Vec<_> = optimizers::IDS.iter().map(|id| optimizers::by_id(id).unwrap().manifest()).collect();
    serde_json::to_string(&all).unwrap()
}

/// [min, max] of the drawn height (f, or log(1 + f) for `log_height` problems) over an n×n grid of the view
/// box: the render worker scales the surface by it.
#[wasm_bindgen]
pub fn surface_range(problem_id: &str, n: usize) -> Vec<f64> {
    let p = problems::by_id(problem_id, 2).expect("problem id");
    let m = p.manifest();
    let (bx, by) = (m.bounds[0], m.bounds[1]);
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for i in 0..n {
        for j in 0..n {
            let x = bx.0 + (bx.1 - bx.0) * i as f64 / (n - 1) as f64;
            let y = by.0 + (by.1 - by.0) * j as f64 / (n - 1) as f64;
            let f = p.eval(&[x, y]);
            let h = if m.log_height { f.max(0.0).ln_1p() } else { f };
            lo = lo.min(h);
            hi = hi.max(h);
        }
    }
    vec![lo, hi]
}

/// The most trail points kept; past it the trail is thinned to every other point (and the stride doubles),
/// so a run of any length keeps its whole shape.
const TRAIL_MAX: usize = 8192;

#[wasm_bindgen]
pub struct Run {
    problem: Box<dyn Problem>,
    opt: Box<dyn Optimizer>,
    rng: Rng,
    iter: u64,
    last: StepInfo,
    diverged: bool,
    /// x, y pairs (f32, problem coordinates)
    trail: Vec<f32>,
    stride: u64,
}

#[wasm_bindgen]
impl Run {
    #[wasm_bindgen(constructor)]
    pub fn new(problem_id: &str, opt_id: &str, x0: &[f64], seed: u64) -> Result<Run, JsError> {
        let problem = problems::by_id(problem_id, 2).ok_or_else(|| JsError::new("unknown problem"))?;
        let opt = optimizers::by_id(opt_id).ok_or_else(|| JsError::new("unknown optimizer"))?;
        let mut r = Run {
            problem,
            opt,
            rng: Rng::new(seed),
            iter: 0,
            last: StepInfo::default(),
            diverged: false,
            trail: vec![],
            stride: 1,
        };
        r.restart(x0);
        Ok(r)
    }

    pub fn restart(&mut self, x0: &[f64]) {
        self.opt.init(x0);
        self.iter = 0;
        self.diverged = false;
        self.stride = 1;
        self.trail.clear();
        self.trail.extend([x0[0] as f32, x0[1] as f32]);
        let mut g = [0.0; 2];
        let f = self.problem.grad(x0, &mut g);
        self.last = StepInfo { f, grad_norm: g[0].hypot(g[1]), evals: 1 };
    }

    pub fn set_param(&mut self, key: &str, v: f64) -> bool {
        self.opt.set_param(key, v)
    }

    /// Runs up to `n` steps, stopping early if the iterate stops being finite. Returns the steps done.
    pub fn step(&mut self, n: u32) -> u32 {
        for k in 0..n {
            if self.diverged {
                return k;
            }
            self.last = self.opt.step(self.problem.as_ref(), &mut self.rng);
            self.iter += 1;
            let x = self.opt.x();
            if !x.iter().all(|v| v.is_finite()) {
                self.diverged = true;
                return k + 1;
            }
            if self.iter.is_multiple_of(self.stride) {
                self.trail.extend([x[0] as f32, x[1] as f32]);
                if self.trail.len() / 2 > TRAIL_MAX {
                    let thinned: Vec<f32> = self.trail.chunks(2).step_by(2).flatten().copied().collect();
                    self.trail = thinned;
                    self.stride *= 2;
                }
            }
        }
        n
    }

    pub fn iter(&self) -> f64 {
        self.iter as f64
    }
    /// f at the start of the last step
    pub fn f(&self) -> f64 {
        self.last.f
    }
    pub fn grad_norm(&self) -> f64 {
        self.last.grad_norm
    }
    pub fn diverged(&self) -> bool {
        self.diverged
    }
    pub fn x(&self) -> f64 {
        self.opt.x()[0]
    }
    pub fn y(&self) -> f64 {
        self.opt.x()[1]
    }
    /// f at the current iterate
    pub fn f_now(&self) -> f64 {
        self.problem.eval(self.opt.x())
    }
    pub fn trail_ptr(&self) -> *const f32 {
        self.trail.as_ptr()
    }
    /// Number of f32 values (2 per point)
    pub fn trail_len(&self) -> usize {
        self.trail.len()
    }
}

/* ---------- M0 benchmarks: the same workloads run natively in crates/bench ---------- */

/// GD on 2D Rosenbrock for `iters` steps; returns the final f so nothing is optimized away.
#[wasm_bindgen]
pub fn bench_gd_rosenbrock(iters: u32) -> f64 {
    let p = problems::by_id("rosenbrock", 2).unwrap();
    let mut o = optimizers::by_id("gd").unwrap();
    o.init(&[-1.2, 1.0]);
    let mut rng = Rng::new(0);
    for _ in 0..iters {
        o.step(p.as_ref(), &mut rng);
    }
    p.eval(o.x())
}

/// Adam from `n` random starts on `dim`-D Rastrigin for `iters` steps (parallel in the threads build).
#[wasm_bindgen]
pub fn bench_many(n: usize, dim: usize, iters: usize) -> f64 {
    let mut xs = multistart::random_starts("rastrigin", dim, n, 1);
    multistart::run_many("rastrigin", "adam", dim, &mut xs, iters)
}

/// A synthetic swarm for the frame-transport stress test: `n` points on a moving flow field, rewritten in
/// place every call.
#[wasm_bindgen]
pub struct Points {
    xy: Vec<f32>,
    t: f32,
}

#[wasm_bindgen]
impl Points {
    #[wasm_bindgen(constructor)]
    pub fn new(n: usize) -> Points {
        let mut rng = Rng::new(42);
        let xy = (0..2 * n).map(|_| rng.range(-1.0, 1.0) as f32).collect();
        Points { xy, t: 0.0 }
    }
    pub fn advance(&mut self, dt: f32) {
        self.t += dt;
        let t = self.t;
        for p in self.xy.as_chunks_mut::<2>().0 {
            let [x, y] = *p;
            let (u, v) = ((3.0 * y + t).sin(), (3.0 * x - t).cos());
            p[0] = (x + dt * 0.3 * u).clamp(-1.0, 1.0);
            p[1] = (y + dt * 0.3 * v).clamp(-1.0, 1.0);
        }
    }
    pub fn ptr(&self) -> *const f32 {
        self.xy.as_ptr()
    }
    pub fn len(&self) -> usize {
        self.xy.len()
    }
    pub fn is_empty(&self) -> bool {
        self.xy.is_empty()
    }
}
