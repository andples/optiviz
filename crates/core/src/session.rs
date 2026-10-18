//! A session: one problem (with its coefficients and view box), several runs racing on it from the same start,
//! a shared timeline, stopping points and run health. Everything the engine worker does that isn't plumbing,
//! here so it can be tested natively.
//!
//! **Timeline.** All runs advance in lockstep, so iteration k means the same for each. Runs are deterministic,
//! so seeking to any k replays every run from its start, applying the parameter changes logged on the way
//! (DESIGN.md §6.5: 2D runs replay at millions of steps per second). Changing a parameter while scrubbed back
//! branches: the logged future is dropped.
//!
//! **Divergence** (DESIGN.md §4.3): every step feeds the run's `Monitor`. A diverged run (non-finite) halts for
//! good; a diverging or unbounded run triggers a stopping point once (if that stop is on, the default), with a
//! diagnosis; leaving the view box is only shown.

use crate::monitor::{Diagnosis, Health, Monitor, Obs, diagnose};
use crate::optimizers::{self, Optimizer};
use crate::problems::{self, Problem};
use crate::trace::Trace;
use crate::{ProblemManifest, Rng, StepInfo};
use serde::Serialize;

pub const MAX_RUNS: usize = 8;
/// Trail points kept per run; past it the trail is thinned (every other point, stride doubles).
pub const TRAIL_MAX: usize = 8192;
/// Metric rows kept per run, thinned the same way.
pub const METRICS_MAX: usize = 4096;

/// Stopping points (breakpoints). Any run meeting one pauses the session.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Stops {
    pub at_iter: Option<u64>,
    pub every: Option<u64>,
    pub f_below: Option<f64>,
    pub grad_below: Option<f64>,
    /// Stop when a run starts diverging or turns out unbounded (on by default)
    pub on_diverge: bool,
    /// Stop when the selected run executes this pseudocode line
    pub line: Option<usize>,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum Stop {
    Iteration { iter: u64 },
    Every { iter: u64 },
    FBelow { run: usize, f: f64 },
    GradBelow { run: usize, grad: f64 },
    Health { run: usize, health: Health },
    Line { run: usize, line: usize },
}

pub struct Run {
    pub opt_id: String,
    opt: Box<dyn Optimizer>,
    /// current parameter values (key, value) and the log of changes (at iteration, key, value)
    pub params: Vec<(String, f64)>,
    log: Vec<(u64, String, f64)>,
    rng: Rng,
    seed: u64,
    monitor: Monitor,
    pub health: Health,
    /// health stopping point already reported (reset on restart)
    reported: bool,
    pub halted: bool,
    pub last: StepInfo,
    pub f_now: f64,
    pub evals: u64,
    x_prev: Vec<f64>,
    /// last finite iterate, for the diagnosis
    x_ok: Vec<f64>,
    trail: Vec<f64>,
    stride: u64,
    /// rows of (iter, f, ‖∇f‖, step length)
    metrics: Vec<[f64; 4]>,
    mstride: u64,
    pub diagnosis: Option<Diagnosis>,
    /// the trace of the selected run's last step, when line stops are on
    pub last_trace: Option<Trace>,
}

impl Run {
    fn new(opt_id: &str, overrides: &[(String, f64)], seed: u64) -> Option<Run> {
        let opt = optimizers::by_id(opt_id)?;
        let params = opt.manifest().params.iter().map(|s| (s.key.to_string(), s.default)).collect();
        let mut r = Run {
            opt_id: opt_id.into(),
            opt,
            params,
            log: vec![],
            rng: Rng::new(seed),
            seed,
            monitor: Monitor::new(&[0.0, 0.0], 0.0, &[(-1.0, 1.0), (-1.0, 1.0)]),
            health: Health::Ok,
            reported: false,
            halted: false,
            last: StepInfo::default(),
            f_now: f64::NAN,
            evals: 0,
            x_prev: vec![],
            x_ok: vec![],
            trail: vec![],
            stride: 1,
            metrics: vec![],
            mstride: 1,
            diagnosis: None,
            last_trace: None,
        };
        for (k, v) in overrides {
            r.set(k, *v);
        }
        Some(r)
    }

    /// Sets a parameter, clamped to its declared range; non-finite values and unknown keys are refused.
    /// (Values can come from a share link, so they are untrusted.)
    fn set(&mut self, key: &str, v: f64) -> bool {
        let Some(spec) = self.opt.manifest().params.into_iter().find(|s| s.key == key) else { return false };
        if !v.is_finite() {
            return false;
        }
        let v = v.clamp(spec.min, spec.max);
        if !self.opt.set_param(key, v) {
            return false;
        }
        if let Some(p) = self.params.iter_mut().find(|(k, _)| k == key) {
            p.1 = v;
        }
        true
    }

    pub fn get(&self, key: &str) -> Option<f64> {
        self.params.iter().find(|(k, _)| k == key).map(|p| p.1)
    }

    /// Back to the start with the parameter values in force at iteration 0.
    fn reset(&mut self, p: &dyn Problem, x0: &[f64], bounds: &[(f64, f64)]) {
        // undo logged changes: re-apply initial values (the first logged value's predecessor is unknown, so the
        // log keeps the value at 0 explicitly; see `set_param`)
        let initial: Vec<(String, f64)> = self
            .params
            .iter()
            .map(|(k, v)| {
                let first = self.log.iter().find(|(_, lk, _)| lk == k);
                (k.clone(), first.map_or(*v, |_| self.value_at(k, 0).unwrap_or(*v)))
            })
            .collect();
        for (k, v) in initial {
            self.set(&k, v);
        }
        self.opt.init(x0);
        self.rng = Rng::new(self.seed);
        let f0 = p.eval(x0);
        self.monitor = Monitor::new(x0, f0, bounds);
        self.health = Health::Ok;
        self.reported = false;
        self.halted = false;
        let mut g = vec![0.0; x0.len()];
        p.grad(x0, &mut g);
        self.last = StepInfo { f: f0, grad_norm: g.iter().map(|v| v * v).sum::<f64>().sqrt(), evals: 0 };
        self.f_now = f0;
        self.evals = 0;
        self.x_prev = x0.to_vec();
        self.x_ok = x0.to_vec();
        self.trail = x0.to_vec();
        self.stride = 1;
        self.metrics = vec![[0.0, f0, self.last.grad_norm, 0.0]];
        self.mstride = 1;
        self.diagnosis = None;
        self.last_trace = None;
    }

    /// The value of `key` in force at iteration `k` according to the log (None if never logged).
    fn value_at(&self, key: &str, k: u64) -> Option<f64> {
        self.log.iter().rfind(|(at, lk, _)| lk == key && *at <= k).map(|e| e.2)
    }

    fn apply_log_at(&mut self, k: u64) {
        let changes: Vec<(String, f64)> =
            self.log.iter().filter(|(at, _, _)| *at == k).map(|(_, key, v)| (key.clone(), *v)).collect();
        for (key, v) in changes {
            self.set(&key, v);
        }
    }

    /// One step at iteration `k` (→ k+1). Returns the health after it.
    fn step(&mut self, p: &dyn Problem, k: u64, trace: bool) -> Health {
        self.apply_log_at(k);
        let mut t = trace.then(Trace::default);
        self.last = self.opt.step(p, &mut self.rng, t.as_mut());
        self.last_trace = t;
        self.evals += self.last.evals as u64;
        let x = self.opt.x().to_vec();
        let f = p.eval(&x);
        self.f_now = f;
        let step_len = x.iter().zip(&self.x_prev).map(|(a, b)| (a - b) * (a - b)).sum::<f64>().sqrt();
        let gn = if self.last.grad_norm.is_nan() {
            // derivative-free methods don't compute it; the readout and the monitor still want it
            let mut g = vec![0.0; x.len()];
            p.grad(&x, &mut g);
            g.iter().map(|v| v * v).sum::<f64>().sqrt()
        } else {
            self.last.grad_norm
        };
        self.last.grad_norm = gn;
        let h = self.monitor.observe(&Obs { x: &x, f, grad_norm: gn, step_len });
        self.health = h;
        if h == Health::Diverged {
            self.halted = true;
            return h;
        }
        self.x_ok = x.clone();
        let it = k + 1;
        if it.is_multiple_of(self.stride) {
            self.trail.extend_from_slice(&x);
            if self.trail.len() / 2 > TRAIL_MAX {
                self.trail = self.trail.chunks(2).step_by(2).flatten().copied().collect();
                self.stride *= 2;
            }
        }
        if it.is_multiple_of(self.mstride) {
            self.metrics.push([it as f64, f, gn, step_len]);
            if self.metrics.len() > METRICS_MAX {
                self.metrics = self.metrics.iter().step_by(2).copied().collect();
                self.mstride *= 2;
            }
        }
        self.x_prev = x;
        h
    }

    pub fn x(&self) -> &[f64] {
        self.opt.x()
    }
    pub fn trail(&self) -> &[f64] {
        &self.trail
    }
    pub fn metrics(&self) -> &[[f64; 4]] {
        &self.metrics
    }
    pub fn shape(&self) -> Option<Vec<f64>> {
        self.opt.shape()
    }
    pub fn manifest(&self) -> crate::OptimizerManifest {
        self.opt.manifest()
    }
}

pub struct Session {
    pub problem_id: String,
    pub coeffs: Vec<(String, f64)>,
    problem: Box<dyn Problem>,
    pub manifest: ProblemManifest,
    /// the drawn box, (x range, y range); starts as the problem's bounds
    pub view: [(f64, f64); 2],
    pub zrange: (f64, f64),
    pub runs: Vec<Run>,
    pub stops: Stops,
    /// timeline position, and the furthest position reached
    pub iter: u64,
    pub head: u64,
    pub selected: usize,
    pub start: Vec<f64>,
    seed: u64,
}

fn coeff_refs(c: &[(String, f64)]) -> Vec<(&str, f64)> {
    c.iter().map(|(k, v)| (k.as_str(), *v)).collect()
}

impl Session {
    pub fn new(problem_id: &str, seed: u64) -> Option<Session> {
        let problem = problems::by_id(problem_id, 2)?;
        let manifest = problem.manifest();
        let view = [manifest.bounds[0], manifest.bounds[1]];
        let zrange = problems::height_range(problem.as_ref(), view, 129);
        Some(Session {
            problem_id: problem_id.into(),
            coeffs: vec![],
            start: manifest.start.clone(),
            problem,
            manifest,
            view,
            zrange,
            runs: vec![],
            stops: Stops { on_diverge: true, ..Default::default() },
            iter: 0,
            head: 0,
            selected: 0,
            seed,
        })
    }

    pub fn problem(&self) -> &dyn Problem {
        self.problem.as_ref()
    }

    /// Switches problem (or coefficients); keeps the runs, restarts them from the problem's default start.
    pub fn set_problem(&mut self, id: &str, coeffs: &[(String, f64)], keep_start: bool) -> bool {
        let Some(p) = problems::build(id, 2, &coeff_refs(coeffs)) else { return false };
        let same = id == self.problem_id;
        self.problem = p;
        self.manifest = self.problem.manifest();
        self.problem_id = id.into();
        self.coeffs = coeffs.to_vec();
        if !same {
            self.view = [self.manifest.bounds[0], self.manifest.bounds[1]];
        }
        if !(same && keep_start) {
            self.start = self.manifest.start.clone();
        }
        self.zrange = problems::height_range(self.problem.as_ref(), self.view, 129);
        self.restart();
        true
    }

    pub fn add_run(&mut self, opt_id: &str, params: &[(String, f64)]) -> Option<usize> {
        if self.runs.len() >= MAX_RUNS {
            return None;
        }
        let seed = self.seed.wrapping_add(self.runs.len() as u64 * 0x9e37_79b9);
        let mut r = Run::new(opt_id, params, seed)?;
        let bounds = self.view.to_vec();
        r.reset(self.problem.as_ref(), &self.start, &bounds);
        // a run added mid-session catches up with the others
        for k in 0..self.iter {
            if r.halted {
                break;
            }
            r.step(self.problem.as_ref(), k, false);
        }
        self.runs.push(r);
        Some(self.runs.len() - 1)
    }

    pub fn remove_run(&mut self, i: usize) {
        if i < self.runs.len() {
            self.runs.remove(i);
            self.selected = self.selected.min(self.runs.len().saturating_sub(1));
        }
    }

    /// Changes a run's optimizer (keeping parameters that share a key) and restarts the session.
    pub fn set_optimizer(&mut self, i: usize, opt_id: &str) -> bool {
        let Some(old) = self.runs.get(i) else { return false };
        let keep: Vec<(String, f64)> = old.params.clone();
        let seed = old.seed;
        let Some(mut r) = Run::new(opt_id, &[], seed) else { return false };
        for (k, v) in keep {
            if k == "lr" {
                // step sizes mean different things across methods; keep the new method's default
                continue;
            }
            r.set(&k, v);
        }
        self.runs[i] = r;
        self.restart();
        true
    }

    /// Sets a parameter on a run from the current iteration on. Scrubbed back, this drops the logged future.
    pub fn set_param(&mut self, i: usize, key: &str, v: f64) -> bool {
        let at = self.iter;
        if self.iter < self.head {
            self.head = self.iter;
        }
        let Some(r) = self.runs.get_mut(i) else { return false };
        let before = r.get(key);
        if !r.set(key, v) {
            return false;
        }
        r.log.retain(|(k, _, _)| *k < at);
        // remember the value before the first change so replays from 0 can restore it
        if r.log.iter().all(|(_, lk, _)| lk != key)
            && let Some(b) = before
        {
            r.log.push((0, key.into(), b));
        }
        if at == 0 {
            r.log.retain(|(_, lk, _)| lk != key);
        }
        r.log.push((at, key.into(), v));
        r.reported = false;
        true
    }

    pub fn set_start(&mut self, x0: [f64; 2]) {
        self.start = x0.to_vec();
        self.restart();
    }

    pub fn restart(&mut self) {
        let bounds = self.view.to_vec();
        for r in &mut self.runs {
            r.reset(self.problem.as_ref(), &self.start, &bounds);
        }
        self.iter = 0;
        self.head = 0;
    }

    /// Advances every live run by up to `n` iterations; stops early at a stopping point. Returns (iterations
    /// done, the stop if one fired).
    pub fn step(&mut self, n: u64) -> (u64, Option<Stop>) {
        for done in 0..n {
            if self.runs.iter().all(|r| r.halted) {
                return (done, None);
            }
            let k = self.iter;
            let mut stop = None;
            for (i, r) in self.runs.iter_mut().enumerate() {
                if r.halted {
                    continue;
                }
                let trace = self.stops.line.is_some() && i == self.selected;
                let h = r.step(self.problem.as_ref(), k, trace);
                if matches!(h, Health::Diverged | Health::Diverging | Health::Unbounded) && !r.reported {
                    r.reported = true;
                    let get = |key: &str| r.get(key);
                    r.diagnosis = Some(diagnose(h, self.problem.as_ref(), r.opt.as_ref(), &r.x_ok, &get));
                    if self.stops.on_diverge || h == Health::Diverged {
                        stop.get_or_insert(Stop::Health { run: i, health: h });
                    }
                }
                if let Some(fb) = self.stops.f_below
                    && r.f_now < fb
                {
                    stop.get_or_insert(Stop::FBelow { run: i, f: r.f_now });
                }
                if let Some(gb) = self.stops.grad_below
                    && r.last.grad_norm < gb
                {
                    stop.get_or_insert(Stop::GradBelow { run: i, grad: r.last.grad_norm });
                }
                if let (Some(line), Some(t)) = (self.stops.line, &r.last_trace)
                    && t.entries.iter().any(|e| e.line == line)
                {
                    stop.get_or_insert(Stop::Line { run: i, line });
                }
            }
            self.iter += 1;
            self.head = self.iter;
            if self.stops.at_iter == Some(self.iter) {
                stop.get_or_insert(Stop::Iteration { iter: self.iter });
            }
            if let Some(e) = self.stops.every
                && e > 0
                && self.iter.is_multiple_of(e)
            {
                stop.get_or_insert(Stop::Every { iter: self.iter });
            }
            if stop.is_some() {
                return (done + 1, stop);
            }
        }
        (n, None)
    }

    /// Moves the timeline to iteration k (≤ head) by replaying every run from its start.
    pub fn seek(&mut self, k: u64) {
        let k = k.min(self.head);
        let head = self.head;
        let bounds = self.view.to_vec();
        for r in &mut self.runs {
            r.reset(self.problem.as_ref(), &self.start, &bounds);
            for j in 0..k {
                if r.halted {
                    break;
                }
                r.step(self.problem.as_ref(), j, false);
                if matches!(r.health, Health::Diverging | Health::Unbounded | Health::Diverged) {
                    r.reported = true;
                }
            }
        }
        self.iter = k;
        self.head = head;
    }

    /// The trace of run i's step from iteration iter−1 to iter (replayed on a copy), for inspection at a stop.
    pub fn inspect(&self, i: usize) -> Option<Trace> {
        let r0 = self.runs.get(i)?;
        if self.iter == 0 {
            return Some(Trace::default());
        }
        let mut r = Run::new(&r0.opt_id, &[], r0.seed)?;
        r.log = r0.log.clone();
        r.params = r0.params.clone();
        for (k, v) in r0.params.clone() {
            r.set(&k, v);
        }
        r.reset(self.problem.as_ref(), &self.start, &self.view);
        for j in 0..self.iter - 1 {
            if r.halted {
                return Some(Trace::default());
            }
            r.step(self.problem.as_ref(), j, false);
        }
        r.step(self.problem.as_ref(), self.iter - 1, true);
        r.last_trace
    }

    /// Enlarges the view box to take in every run's trail (finite points), so runaway runs stay visible.
    pub fn fit_view(&mut self) {
        let [mut bx, mut by] = [self.manifest.bounds[0], self.manifest.bounds[1]];
        for r in &self.runs {
            for p in r.trail.chunks(2) {
                if p[0].is_finite() && p[1].is_finite() && p[0].abs() < 1e12 && p[1].abs() < 1e12 {
                    bx = (bx.0.min(p[0]), bx.1.max(p[0]));
                    by = (by.0.min(p[1]), by.1.max(p[1]));
                }
            }
        }
        // square-ish margins
        let (wx, wy) = (bx.1 - bx.0, by.1 - by.0);
        let m = 0.08 * wx.max(wy);
        self.view = [(bx.0 - m, bx.1 + m), (by.0 - m, by.1 + m)];
        self.zrange = problems::height_range(self.problem.as_ref(), self.view, 129);
        for r in &mut self.runs {
            r.monitor.set_bounds(&self.view);
        }
    }

    pub fn reset_view(&mut self) {
        self.view = [self.manifest.bounds[0], self.manifest.bounds[1]];
        self.zrange = problems::height_range(self.problem.as_ref(), self.view, 129);
        for r in &mut self.runs {
            r.monitor.set_bounds(&self.view);
        }
    }

    /// Problem coordinates → surface coordinates (u, v ∈ [-1, 1] over the view box).
    pub fn to_uv(&self, x: f64, y: f64) -> (f64, f64) {
        let [(x0, x1), (y0, y1)] = self.view;
        (2.0 * (x - x0) / (x1 - x0) - 1.0, 2.0 * (y - y0) / (y1 - y0) - 1.0)
    }
    pub fn from_uv(&self, u: f64, v: f64) -> (f64, f64) {
        let [(x0, x1), (y0, y1)] = self.view;
        (x0 + (u + 1.0) * 0.5 * (x1 - x0), y0 + (v + 1.0) * 0.5 * (y1 - y0))
    }

    /// The drawn height of the surface at (u, v), in world units (see shaders/common.wgsl `worldPos`).
    fn world_y(&self, u: f64, v: f64, height_scale: f64) -> f64 {
        let (x, y) = self.from_uv(u, v);
        let f = self.problem.eval(&[x, y]);
        let h = if self.manifest.log_height { f.max(0.0).ln_1p() } else { f };
        let t = (h - self.zrange.0) / (self.zrange.1 - self.zrange.0).max(1e-20);
        (t - 0.5) * height_scale
    }

    /// Where a ray (world space: x = u, y up, z = −v) first meets the drawn surface, in problem coordinates.
    pub fn pick(&self, o: [f64; 3], d: [f64; 3], height_scale: f64) -> Option<[f64; 2]> {
        let at = |s: f64| [o[0] + s * d[0], o[1] + s * d[1], o[2] + s * d[2]];
        let above = |p: [f64; 3]| p[1] - self.world_y(p[0], -p[2], height_scale);
        // march across the box |u|, |v| ≤ 1, then refine by bisection
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        let (mut s0, s_end) = (0.0, 12.0 / len);
        let steps = 2000;
        let ds = (s_end - s0) / steps as f64;
        let mut prev = None;
        for _ in 0..steps {
            let p = at(s0);
            let inside = p[0].abs() <= 1.0 && p[2].abs() <= 1.0;
            if inside {
                let a = above(p);
                if let Some((sp, ap)) = prev
                    && ap > 0.0
                    && a <= 0.0
                {
                    let (mut lo, mut hi) = (sp, s0);
                    for _ in 0..40 {
                        let mid = 0.5 * (lo + hi);
                        if above(at(mid)) > 0.0 { lo = mid } else { hi = mid }
                    }
                    let q = at(0.5 * (lo + hi));
                    let (x, y) = self.from_uv(q[0], -q[2]);
                    return Some([x, y]);
                }
                prev = Some((s0, a));
            } else {
                prev = None;
            }
            s0 += ds;
        }
        None
    }
}

impl Monitor {
    pub(crate) fn set_bounds(&mut self, b: &[(f64, f64)]) {
        self.bounds = b.to_vec();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session(problem: &str, opts: &[&str]) -> Session {
        let mut s = Session::new(problem, 1).unwrap();
        for o in opts {
            s.add_run(o, &[]).unwrap();
        }
        s
    }

    #[test]
    fn runs_advance_in_lockstep() {
        let mut s = session("rosenbrock", &["gd", "adam", "lbfgs"]);
        let (n, stop) = s.step(100);
        assert_eq!((n, stop), (100, None));
        assert_eq!(s.iter, 100);
        assert!(s.runs.iter().all(|r| r.metrics().last().unwrap()[0] == 100.0));
    }

    #[test]
    fn seeking_back_replays_exactly_including_parameter_changes() {
        let mut s = session("himmelblau", &["adam", "momentum"]);
        s.step(40);
        s.set_param(0, "lr", 0.01);
        s.step(60);
        let at100: Vec<Vec<f64>> = s.runs.iter().map(|r| r.x().to_vec()).collect();
        s.step(50);
        s.seek(100);
        assert_eq!(s.iter, 100);
        assert_eq!(s.head, 150);
        let replayed: Vec<Vec<f64>> = s.runs.iter().map(|r| r.x().to_vec()).collect();
        assert_eq!(at100, replayed);
        // and from 0 the original learning rate is back
        s.seek(0);
        assert_eq!(s.runs[0].get("lr"), Some(0.05));
        s.seek(41);
        assert_eq!(s.runs[0].get("lr"), Some(0.01));
    }

    #[test]
    fn changing_a_parameter_while_scrubbed_back_branches() {
        let mut s = session("bowl", &["gd"]);
        s.step(100);
        s.seek(30);
        s.set_param(0, "lr", 0.02);
        assert_eq!(s.head, 30);
        s.step(10);
        assert_eq!(s.iter, 40);
        assert_eq!(s.head, 40);
    }

    #[test]
    fn divergence_stops_once_with_a_diagnosis_and_a_suggestion() {
        let mut s = session("bowl", &["gd"]);
        s.set_param(0, "lr", 0.11); // > 2/L = 0.1
        let (_, stop) = s.step(10_000);
        assert!(matches!(stop, Some(Stop::Health { run: 0, health: Health::Diverging })), "{stop:?}");
        let d = s.runs[0].diagnosis.clone().unwrap();
        assert_eq!(d.suggest.unwrap().0, "lr");
        // continuing does not fire the same stop again
        let (_, again) = s.step(5);
        assert_eq!(again, None);
    }

    #[test]
    fn non_finite_runs_halt_and_the_others_continue() {
        let mut s = session("rosenbrock", &["gd", "adam"]);
        s.set_param(0, "lr", 0.5);
        s.stops.on_diverge = false;
        let mut halted = false;
        for _ in 0..100 {
            s.step(100);
            halted |= s.runs[0].halted;
        }
        assert!(halted, "gd with η = 0.5 on Rosenbrock overflows");
        assert!(s.runs[0].x_ok.iter().all(|v| v.is_finite()));
        assert!(!s.runs[1].halted);
        assert_eq!(s.iter, 10_000);
    }

    #[test]
    fn stopping_points_fire() {
        let mut s = session("bowl", &["adam"]);
        s.stops.at_iter = Some(25);
        assert_eq!(s.step(1000), (25, Some(Stop::Iteration { iter: 25 })));
        s.stops.at_iter = None;
        s.stops.grad_below = Some(1e-3);
        let (_, stop) = s.step(100_000);
        assert!(matches!(stop, Some(Stop::GradBelow { .. })));
        s.stops.grad_below = None;
        s.stops.line = Some(12); // Adam's update line
        let (n, stop) = s.step(10);
        assert_eq!((n, stop), (1, Some(Stop::Line { run: 0, line: 12 })));
    }

    #[test]
    fn inspect_replays_the_last_step_with_its_trace() {
        let mut s = session("rosenbrock", &["adam"]);
        s.step(7);
        let t = s.inspect(0).unwrap();
        let theta = t.entries.iter().find(|e| e.line == 12).unwrap();
        let x = theta.vars.iter().find(|v| v.0 == r"\theta_t").unwrap();
        assert_eq!(x.1, s.runs[0].x());
    }

    #[test]
    fn fit_view_takes_in_a_runaway_trail() {
        let mut s = session("saddle", &["gd"]);
        s.set_param(0, "lr", 0.05);
        s.stops.on_diverge = false;
        s.step(60);
        let y = s.runs[0].x()[1];
        assert!(y.abs() > 2.0, "left the box: y = {y}");
        s.fit_view();
        let [_, (y0, y1)] = s.view;
        assert!(y0 < y && y < y1);
        s.reset_view();
        assert_eq!(s.view[1], (-2.0, 2.0));
    }

    #[test]
    fn pick_finds_the_surface_point_under_a_vertical_ray() {
        let s = session("bowl", &[]);
        let (u, v) = s.to_uv(0.5, -0.8);
        let p = s.pick([u, 5.0, -v], [0.0, -1.0, 0.0], 0.62).unwrap();
        assert!((p[0] - 0.5).abs() < 1e-6 && (p[1] + 0.8).abs() < 1e-6, "{p:?}");
    }

    #[test]
    fn parameters_from_outside_are_clamped_and_non_finite_refused() {
        let mut s = session("bowl", &["adam"]);
        assert!(s.set_param(0, "lr", 1e300));
        assert_eq!(s.runs[0].get("lr"), Some(2.0));
        assert!(!s.set_param(0, "lr", f64::NAN));
        assert!(!s.set_param(0, "nope", 1.0));
        assert_eq!(s.runs[0].get("lr"), Some(2.0));
    }

    #[test]
    fn changing_coefficients_keeps_runs_and_start() {
        let mut s = session("rosenbrock", &["adam", "gd"]);
        s.set_start([0.5, 0.5]);
        s.step(10);
        assert!(s.set_problem("rosenbrock", &[("a".into(), 0.5)], true));
        assert_eq!(s.runs.len(), 2);
        assert_eq!(s.start, vec![0.5, 0.5]);
        assert_eq!(s.iter, 0);
        assert_eq!(s.manifest.minima[0], vec![0.5, 0.25]);
    }
}
