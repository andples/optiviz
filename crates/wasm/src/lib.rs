//! The engine worker's view of the core: an `Engine` wraps a `Session` (runs, timeline, stopping points,
//! health) and lays out each frame for the render worker; plus manifests as JSON and the benchmark workloads.
//!
//! Frames: `build_frame` fills two buffers the worker copies straight out of WebAssembly memory into the shared
//! triple buffer (web/src/engine/frames.ts):
//! - `index` (f64): [runs, minima offset, minima count, 5 reserved] then per run `RUN_STRIDE` values:
//!   trail offset, trail count, shape offset, shape count, marker u, v, start u, v, health, halted, and the
//!   steepest-descent direction at the marker (unit vector in u, v)
//! - `frame` (f32): u, v pairs (surface coordinates over the view box): every trail, every shape (closed), minima

use optiviz_core::monitor::Health;
use optiviz_core::session::{MAX_RUNS, Session, Stop, Stops};
use optiviz_core::{Rng, multistart, optimizers, problems};
use wasm_bindgen::prelude::*;

#[cfg(feature = "threads")]
pub use wasm_bindgen_rayon::init_thread_pool;

pub const RUN_STRIDE: usize = 12;
pub const INDEX_HEAD: usize = 8;
/// Longest drawn trail segment, in surface units (the view box is 2 wide), and extra points allowed per run.
const DRAPE_STEP: f32 = 0.012;
const DRAPE_BUDGET: usize = 6000;

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

/// True when built with the rayon thread pool (the page must still call `initThreadPool`).
#[wasm_bindgen]
pub fn threads_enabled() -> bool {
    cfg!(feature = "threads")
}

/// The 2D manifests of every problem with textbook coefficients, as a JSON array.
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

fn health_code(h: Health) -> f64 {
    match h {
        Health::Ok => 0.0,
        Health::OutOfView => 1.0,
        Health::Diverging => 2.0,
        Health::Unbounded => 3.0,
        Health::Diverged => 4.0,
    }
}

fn pairs(json: &str) -> Vec<(String, f64)> {
    serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(json)
        .map(|m| m.into_iter().filter_map(|(k, v)| v.as_f64().map(|v| (k, v))).collect())
        .unwrap_or_default()
}

#[wasm_bindgen]
pub struct Engine {
    s: Session,
    stop: Option<Stop>,
    frame: Vec<f32>,
    index: Vec<f64>,
}

#[wasm_bindgen]
impl Engine {
    #[wasm_bindgen(constructor)]
    pub fn new(problem_id: &str, seed: u64) -> Result<Engine, JsError> {
        let s = Session::new(problem_id, seed).ok_or_else(|| JsError::new("unknown problem"))?;
        Ok(Engine { s, stop: None, frame: vec![], index: vec![] })
    }

    /* ---- problem ---- */

    /// Current problem, view box and height range, as JSON for the renderer and the UI.
    pub fn problem_json(&self) -> String {
        serde_json::json!({
            "manifest": self.s.manifest,
            "coeffs": self.s.coeffs.iter().map(|(k, v)| (k.clone(), serde_json::json!(v))).collect::<serde_json::Map<_, _>>(),
            "view": self.s.view,
            "zrange": [self.s.zrange.0, self.s.zrange.1],
            "start": self.s.start,
        })
        .to_string()
    }

    /// Switches problem or coefficients (`coeffs` is a JSON object of key → value).
    pub fn set_problem(&mut self, id: &str, coeffs: &str, keep_start: bool) -> bool {
        self.stop = None;
        self.s.set_problem(id, &pairs(coeffs), keep_start)
    }

    pub fn fit_view(&mut self) {
        self.s.fit_view();
    }
    pub fn reset_view(&mut self) {
        self.s.reset_view();
    }

    /* ---- runs ---- */

    pub fn add_run(&mut self, opt_id: &str, params: &str) -> i32 {
        self.s.add_run(opt_id, &pairs(params)).map_or(-1, |i| i as i32)
    }
    pub fn remove_run(&mut self, i: usize) {
        self.s.remove_run(i);
    }
    pub fn set_optimizer(&mut self, i: usize, opt_id: &str) -> bool {
        self.s.set_optimizer(i, opt_id)
    }
    pub fn set_param(&mut self, i: usize, key: &str, v: f64) -> bool {
        self.s.set_param(i, key, v)
    }
    pub fn select(&mut self, i: usize) {
        self.s.selected = i.min(self.s.runs.len().saturating_sub(1));
    }
    pub fn run_count(&self) -> usize {
        self.s.runs.len()
    }
    pub fn max_runs(&self) -> usize {
        MAX_RUNS
    }

    /// Every run's state for the UI, as JSON.
    pub fn runs_json(&self) -> String {
        let runs: Vec<_> = self
            .s
            .runs
            .iter()
            .map(|r| {
                serde_json::json!({
                    "opt": r.opt_id,
                    "params": r.params.iter().map(|(k, v)| (k.clone(), serde_json::json!(v))).collect::<serde_json::Map<_, _>>(),
                    "health": r.health,
                    "halted": r.halted,
                    "f": r.f_now,
                    "gradNorm": r.last.grad_norm,
                    "x": r.x(),
                    "evals": r.evals,
                    "diagnosis": r.diagnosis,
                })
            })
            .collect();
        serde_json::to_string(&runs).unwrap()
    }

    /// Run i's metric rows (iter, f, ‖∇f‖, step length), flattened.
    pub fn metrics(&self, i: usize) -> Vec<f64> {
        self.s.runs.get(i).map(|r| r.metrics().iter().flatten().copied().collect()).unwrap_or_default()
    }

    /// The trace of run i's last step, as JSON (for the debugger's state panel).
    pub fn inspect(&self, i: usize) -> String {
        serde_json::to_string(&self.s.inspect(i).unwrap_or_default()).unwrap()
    }

    /* ---- start point ---- */

    pub fn set_start(&mut self, x: f64, y: f64) {
        self.s.set_start([x, y]);
    }
    /// Sets the start from surface coordinates (the contour view).
    pub fn set_start_uv(&mut self, u: f64, v: f64) {
        let (x, y) = self.s.from_uv(u, v);
        self.s.set_start([x, y]);
    }
    /// Sets the start where a world-space ray meets the surface; false if it misses.
    #[allow(clippy::too_many_arguments)]
    pub fn pick_start(&mut self, ox: f64, oy: f64, oz: f64, dx: f64, dy: f64, dz: f64, height_scale: f64) -> bool {
        match self.s.pick([ox, oy, oz], [dx, dy, dz], height_scale) {
            Some(p) => {
                self.s.set_start(p);
                true
            }
            None => false,
        }
    }

    /* ---- timeline ---- */

    pub fn restart(&mut self) {
        self.stop = None;
        self.s.restart();
    }
    /// Up to n iterations of every run; returns those done (fewer if a stopping point fired).
    pub fn step(&mut self, n: u32) -> u32 {
        let (done, stop) = self.s.step(n as u64);
        if stop.is_some() {
            self.stop = stop;
        }
        done as u32
    }
    /// The stopping point that fired last, as JSON ("" if none); reading clears it.
    pub fn take_stop(&mut self) -> String {
        self.stop.take().map(|s| serde_json::to_string(&s).unwrap()).unwrap_or_default()
    }
    pub fn seek(&mut self, k: f64) {
        self.s.seek(k.max(0.0) as u64);
    }
    pub fn iter(&self) -> f64 {
        self.s.iter as f64
    }
    pub fn head(&self) -> f64 {
        self.s.head as f64
    }
    pub fn all_halted(&self) -> bool {
        self.s.runs.iter().all(|r| r.halted)
    }

    /// Stopping points from JSON: {at_iter, every, f_below, grad_below, on_diverge, line} (missing = off).
    pub fn set_stops(&mut self, json: &str) {
        let v: serde_json::Value = serde_json::from_str(json).unwrap_or_default();
        let num = |k: &str| v.get(k).and_then(|x| x.as_f64());
        self.s.stops = Stops {
            at_iter: num("at_iter").map(|x| x as u64),
            every: num("every").map(|x| x as u64),
            f_below: num("f_below"),
            grad_below: num("grad_below"),
            on_diverge: v.get("on_diverge").and_then(|x| x.as_bool()).unwrap_or(true),
            line: num("line").map(|x| x as usize),
        };
    }

    /* ---- frames ---- */

    /// Lays out the current state (see the module docs). Then read `frame_ptr/len` and `index_ptr/len`.
    pub fn build_frame(&mut self) {
        let s = &self.s;
        self.frame.clear();
        self.index.clear();
        self.index.resize(INDEX_HEAD + RUN_STRIDE * s.runs.len(), 0.0);
        self.index[0] = s.runs.len() as f64;
        let uv = |x: f64, y: f64| {
            let (u, v) = s.to_uv(x, y);
            [u as f32, v as f32]
        };
        let mut runs_idx = vec![];
        for r in &s.runs {
            let t0 = self.frame.len() / 2;
            // long segments are subdivided so they drape over the surface instead of cutting through it
            // (straight in x, y; each piece is lifted to the surface by the shader); a budget per run caps it
            let mut budget = DRAPE_BUDGET;
            let mut prev: Option<[f32; 2]> = None;
            for p in r.trail().chunks(2) {
                let q = uv(p[0], p[1]);
                if let Some(a) = prev {
                    let d = ((q[0] - a[0]).powi(2) + (q[1] - a[1]).powi(2)).sqrt();
                    let inside = a.iter().chain(&q).all(|v| v.abs() <= 1.0);
                    let n = if inside { ((d / DRAPE_STEP).ceil() as usize).clamp(1, 64).min(budget + 1) } else { 1 };
                    budget -= n - 1;
                    for k in 1..n {
                        let t = k as f32 / n as f32;
                        self.frame.extend([a[0] + t * (q[0] - a[0]), a[1] + t * (q[1] - a[1])]);
                    }
                }
                self.frame.extend(q);
                prev = Some(q);
            }
            let tn = self.frame.len() / 2 - t0;
            let s0 = self.frame.len() / 2;
            if let Some(sh) = r.shape() {
                for p in sh.chunks(2) {
                    self.frame.extend(uv(p[0], p[1]));
                }
                self.frame.extend(uv(sh[0], sh[1]));
            }
            let sn = self.frame.len() / 2 - s0;
            let x = r.x();
            let (mu, mv) = if x.iter().all(|v| v.is_finite()) { s.to_uv(x[0], x[1]) } else { (f64::NAN, f64::NAN) };
            let (su, sv) = s.to_uv(s.start[0], s.start[1]);
            // steepest-descent direction at x, as a unit vector in surface coordinates (for the arrow)
            let (mut du, mut dv) = (0.0, 0.0);
            if x.iter().all(|v| v.is_finite()) {
                let mut g = [0.0; 2];
                s.problem().grad(x, &mut g);
                let [(x0, x1), (y0, y1)] = s.view;
                let (a, b) = (-g[0] * 2.0 / (x1 - x0), -g[1] * 2.0 / (y1 - y0));
                let n = a.hypot(b);
                if n > 0.0 && n.is_finite() {
                    (du, dv) = (a / n, b / n);
                }
            }
            runs_idx.push([
                t0 as f64,
                tn as f64,
                s0 as f64,
                sn as f64,
                mu,
                mv,
                su,
                sv,
                health_code(r.health),
                if r.halted { 1.0 } else { 0.0 },
                du,
                dv,
            ]);
        }
        let m0 = self.frame.len() / 2;
        for m in &s.manifest.minima {
            self.frame.extend(uv(m[0], m[1]));
        }
        self.index[1] = m0 as f64;
        self.index[2] = (self.frame.len() / 2 - m0) as f64;
        for (i, r) in runs_idx.iter().enumerate() {
            self.index[INDEX_HEAD + RUN_STRIDE * i..INDEX_HEAD + RUN_STRIDE * (i + 1)].copy_from_slice(r);
        }
    }
    pub fn frame_ptr(&self) -> *const f32 {
        self.frame.as_ptr()
    }
    pub fn frame_len(&self) -> usize {
        self.frame.len()
    }
    pub fn index_ptr(&self) -> *const f64 {
        self.index.as_ptr()
    }
    pub fn index_len(&self) -> usize {
        self.index.len()
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
        o.step(p.as_ref(), &mut rng, None);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_layout_is_consistent() {
        let mut e = Engine::new("himmelblau", 1).unwrap();
        e.add_run("adam", "{}");
        e.add_run("nelder-mead", "{}");
        e.step(20);
        e.build_frame();
        assert_eq!(e.index[0], 2.0);
        let n_minima = e.index[2] as usize;
        assert_eq!(n_minima, 4);
        let total = e.frame.len() / 2;
        for i in 0..2 {
            let r = &e.index[INDEX_HEAD + RUN_STRIDE * i..];
            assert!(r[0] + r[1] <= total as f64 && r[2] + r[3] <= total as f64);
        }
        // Nelder–Mead draws a closed triangle: 3 vertices + the first again
        assert_eq!(e.index[INDEX_HEAD + RUN_STRIDE + 3], 4.0);
        assert_eq!(e.index[1] as usize + n_minima, total);
    }

    #[test]
    fn stops_and_problem_json_roundtrip() {
        let mut e = Engine::new("bowl", 1).unwrap();
        e.add_run("gd", r#"{"lr": 0.11}"#);
        e.set_stops(r#"{"on_diverge": true}"#);
        e.step(10_000);
        let stop = e.take_stop();
        assert!(stop.contains("diverging"), "{stop}");
        assert!(e.runs_json().contains("2/L"));
        assert!(e.set_problem("bowl", r#"{"kappa": 50}"#, true));
        let pj: serde_json::Value = serde_json::from_str(&e.problem_json()).unwrap();
        assert_eq!(pj["coeffs"]["kappa"], 50.0);
    }
}
