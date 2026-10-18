//! A session: one problem (with its coefficients and view box), several runs racing on it, a shared timeline,
//! stopping points and run health. Everything the engine worker does that isn't plumbing, here so it can be
//! tested natively.
//!
//! **Runs and walkers.** A run is one optimizer with its parameters. It has one walker (one iterate) per starting
//! point: the clicked start, plus random starts in the view box when the multistart setting asks for more
//! (owner's request, 2026-10-19). Every run uses the same set of starts, so runs stay comparable.
//!
//! **Random restarts** (multistart; Rinnooy Kan & Timmer 1987, Martí et al. 2013): a walker that converges
//! (‖∇f‖ or its step effectively zero for 10 iterations) or fails (diverging, unbounded, not finite) can jump to
//! a new random point in the view box and start over; the run keeps the best point any walker found. Each
//! walker's sequence of restart points depends only on the session seed and the walker's index, so it is the
//! same for every run.
//!
//! **Timeline.** All runs advance in lockstep, so iteration k means the same for each. Everything is
//! deterministic, so seeking to any k replays every run from its starts, applying the parameter changes logged
//! on the way. Changing a parameter while scrubbed back branches: the logged future is dropped.
//!
//! **Divergence** (DESIGN.md §4.3): every step feeds the walker's `Monitor`. With a single walker and no
//! restarts on failure, a diverging or unbounded run triggers a stopping point once, with a diagnosis, and a
//! non-finite walker halts. With several walkers or restarts, failures are counted and shown, not stopped on.

use crate::monitor::{Diagnosis, Health, Monitor, Obs, diagnose};
use crate::optimizers::{self, Optimizer};
use crate::problems::{self, Problem};
use crate::trace::Trace;
use crate::{ProblemManifest, Rng, StepInfo};
use serde::Serialize;

pub const MAX_RUNS: usize = 8;
pub const MAX_STARTS: usize = 64;
/// Trail points kept per run (shared by its walkers); past it a trail is thinned (every other point, stride
/// doubles).
pub const TRAIL_MAX: usize = 8192;
/// Metric rows kept per run, thinned the same way.
pub const METRICS_MAX: usize = 4096;
/// Iterations of no movement before a walker counts as converged.
const STILL_FOR: u32 = 10;

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

/// How many starting points, and when walkers restart from a random point.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Multistart {
    pub starts: usize,
    pub on_converged: bool,
    pub on_failed: bool,
}

impl Default for Multistart {
    fn default() -> Self {
        Multistart { starts: 1, on_converged: false, on_failed: false }
    }
}

impl Multistart {
    fn active(&self) -> bool {
        self.starts > 1 || self.on_converged || self.on_failed
    }
}

/// One iterate of a run: its own optimizer state, health, trail and best point.
struct Walker {
    opt: Box<dyn Optimizer>,
    rng: Rng,
    /// draws this walker's restart points
    jump: Rng,
    monitor: Monitor,
    health: Health,
    reported: bool,
    halted: bool,
    last: StepInfo,
    f_now: f64,
    evals: u64,
    x_prev: Vec<f64>,
    /// last finite iterate, for the diagnosis
    x_ok: Vec<f64>,
    /// x, y pairs; NaN pairs separate the segments between restarts
    trail: Vec<f64>,
    stride: u64,
    last_trace: Option<Trace>,
    diagnosis: Option<Diagnosis>,
    restarts: u32,
    best_x: Vec<f64>,
    best_f: f64,
    still: u32,
}

impl Walker {
    fn begin(&mut self, p: &dyn Problem, x0: &[f64], bounds: &[(f64, f64)]) {
        self.opt.init(x0);
        let f0 = p.eval(x0);
        self.monitor = Monitor::new(x0, f0, bounds);
        self.health = Health::Ok;
        self.reported = false;
        self.halted = false;
        let mut g = vec![0.0; x0.len()];
        p.grad(x0, &mut g);
        self.last = StepInfo { f: f0, grad_norm: g.iter().map(|v| v * v).sum::<f64>().sqrt(), evals: 0 };
        self.f_now = f0;
        self.x_prev = x0.to_vec();
        self.x_ok = x0.to_vec();
        self.still = 0;
        if !self.trail.is_empty() {
            self.trail.extend([f64::NAN, f64::NAN]);
        }
        self.trail.extend_from_slice(x0);
        if f0 < self.best_f {
            self.best_f = f0;
            self.best_x = x0.to_vec();
        }
    }

    fn x(&self) -> &[f64] {
        self.opt.x()
    }

    /// Thins the trail to every other point, keeping the restart separators.
    fn thin(&mut self) {
        let mut out = Vec::with_capacity(self.trail.len() / 2 + 8);
        for (i, p) in self.trail.chunks(2).enumerate() {
            if i % 2 == 0 || p[0].is_nan() {
                out.extend_from_slice(p);
            }
        }
        self.trail = out;
        self.stride *= 2;
    }
}

pub struct Run {
    pub opt_id: String,
    /// current parameter values (key, value) and the log of changes (at iteration, key, value)
    pub params: Vec<(String, f64)>,
    log: Vec<(u64, String, f64)>,
    seed: u64,
    walkers: Vec<Walker>,
    /// rows of (iter, f, ‖∇f‖, step length); f is the best found so far when multistart is active
    metrics: Vec<[f64; 4]>,
    mstride: u64,
    multi: bool,
}

/// A run's state for the UI: what the primary walker (the clicked start) is doing, plus the walkers' totals.
#[derive(Clone, Debug, Serialize)]
pub struct RunSummary {
    pub health: Health,
    pub halted: bool,
    /// the lowest current f among live walkers (just f for a single walker)
    pub f: f64,
    pub grad_norm: f64,
    pub evals: u64,
    pub walkers: usize,
    pub failed: usize,
    pub converged: usize,
    pub restarts: u32,
    pub best_f: f64,
    pub best_x: Vec<f64>,
}

impl Run {
    fn new(opt_id: &str, overrides: &[(String, f64)], seed: u64) -> Option<Run> {
        let opt = optimizers::by_id(opt_id)?;
        let params = opt.manifest().params.iter().map(|s| (s.key.to_string(), s.default)).collect();
        let mut r = Run {
            opt_id: opt_id.into(),
            params,
            log: vec![],
            seed,
            walkers: vec![],
            metrics: vec![],
            mstride: 1,
            multi: false,
        };
        for (k, v) in overrides {
            r.set(k, *v);
        }
        Some(r)
    }

    /// Sets a parameter, clamped to its declared range; non-finite values and unknown keys are refused.
    /// (Values can come from a share link, so they are untrusted.)
    fn set(&mut self, key: &str, v: f64) -> bool {
        let Some(m) = optimizers::by_id(&self.opt_id) else { return false };
        let Some(spec) = m.manifest().params.into_iter().find(|s| s.key == key) else { return false };
        if !v.is_finite() {
            return false;
        }
        let v = v.clamp(spec.min, spec.max);
        for w in &mut self.walkers {
            w.opt.set_param(key, v);
        }
        if let Some(p) = self.params.iter_mut().find(|(k, _)| k == key) {
            p.1 = v;
        }
        true
    }

    pub fn get(&self, key: &str) -> Option<f64> {
        self.params.iter().find(|(k, _)| k == key).map(|p| p.1)
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

    /// Back to iteration 0: parameters as they were then, one fresh walker per start.
    fn reset(&mut self, p: &dyn Problem, starts: &[Vec<f64>], bounds: &[(f64, f64)], ms: Multistart, seed: u64) {
        let initial: Vec<(String, f64)> =
            self.params.iter().map(|(k, v)| (k.clone(), self.value_at(k, 0).unwrap_or(*v))).collect();
        for (k, v) in &initial {
            if let Some(q) = self.params.iter_mut().find(|(pk, _)| pk == k) {
                q.1 = *v;
            }
        }
        self.multi = ms.active();
        self.walkers = starts
            .iter()
            .enumerate()
            .map(|(j, x0)| {
                let mut opt = optimizers::by_id(&self.opt_id).expect("optimizer id");
                for (k, v) in &self.params {
                    opt.set_param(k, *v);
                }
                let mut w = Walker {
                    opt,
                    rng: Rng::new(self.seed.wrapping_add(j as u64)),
                    jump: Rng::new(seed ^ 0x6a75_6d70).stream(j),
                    monitor: Monitor::new(x0, 0.0, bounds),
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
                    last_trace: None,
                    diagnosis: None,
                    restarts: 0,
                    best_x: x0.clone(),
                    best_f: f64::INFINITY,
                    still: 0,
                };
                w.begin(p, x0, bounds);
                w
            })
            .collect();
        let s = self.summary();
        self.metrics = vec![[0.0, if self.multi { s.best_f } else { s.f }, s.grad_norm, 0.0]];
        self.mstride = 1;
    }

    /// One iteration of every live walker (k → k+1), restarting walkers as the policy says. Returns the primary
    /// walker's health.
    fn step(&mut self, p: &dyn Problem, k: u64, trace: bool, ms: Multistart, bounds: &[(f64, f64)]) -> Health {
        self.apply_log_at(k);
        let scale = bounds.iter().map(|(a, b)| (b - a) * (b - a)).sum::<f64>().sqrt();
        let cap = (TRAIL_MAX / self.walkers.len().max(1)).max(256);
        let it = k + 1;
        for (j, w) in self.walkers.iter_mut().enumerate() {
            if w.halted {
                continue;
            }
            let mut t = (trace && j == 0).then(Trace::default);
            w.last = w.opt.step(p, &mut w.rng, t.as_mut());
            w.last_trace = t;
            w.evals += w.last.evals as u64;
            let x = w.opt.x().to_vec();
            let f = p.eval(&x);
            w.f_now = f;
            let step_len = x.iter().zip(&w.x_prev).map(|(a, b)| (a - b) * (a - b)).sum::<f64>().sqrt();
            if w.last.grad_norm.is_nan() {
                // derivative-free methods don't compute it; the readout, the monitor and restarts want it
                let mut g = vec![0.0; x.len()];
                p.grad(&x, &mut g);
                w.last.grad_norm = g.iter().map(|v| v * v).sum::<f64>().sqrt();
            }
            let gn = w.last.grad_norm;
            let h = w.monitor.observe(&Obs { x: &x, f, grad_norm: gn, step_len });
            w.health = h;
            let failed = matches!(h, Health::Diverging | Health::Unbounded | Health::Diverged);
            if h != Health::Diverged {
                w.x_ok = x.clone();
                if f < w.best_f {
                    w.best_f = f;
                    w.best_x = x.clone();
                }
                if it.is_multiple_of(w.stride) {
                    w.trail.extend_from_slice(&x);
                    if w.trail.len() / 2 > cap {
                        w.thin();
                    }
                }
            }
            let still = gn <= 1e-7 * f.abs().max(1.0) || step_len <= 1e-10 * scale;
            w.still = if still && h != Health::Diverged { w.still + 1 } else { 0 };
            let converged = w.still >= STILL_FOR;
            if (converged && ms.on_converged) || (failed && ms.on_failed) {
                let x0: Vec<f64> = bounds.iter().map(|&(lo, hi)| w.jump.range(lo, hi)).collect();
                w.restarts += 1;
                w.begin(p, &x0, bounds);
            } else if h == Health::Diverged {
                w.halted = true;
            }
            w.x_prev = w.opt.x().to_vec();
        }
        if it.is_multiple_of(self.mstride) {
            let s = self.summary();
            let step_len = self.walkers.first().map_or(0.0, |w| {
                let n = w.trail.len();
                if n >= 4 {
                    let (a, b) = (&w.trail[n - 4..n - 2], &w.trail[n - 2..]);
                    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
                } else {
                    0.0
                }
            });
            self.metrics.push([it as f64, if self.multi { s.best_f } else { s.f }, s.grad_norm, step_len]);
            if self.metrics.len() > METRICS_MAX {
                self.metrics = self.metrics.iter().step_by(2).copied().collect();
                self.mstride *= 2;
            }
        }
        self.walkers[0].health
    }

    /// The live walker with the lowest current f.
    fn lead(&self) -> Option<&Walker> {
        self.walkers.iter().filter(|w| !w.halted && w.f_now.is_finite()).min_by(|a, b| a.f_now.total_cmp(&b.f_now))
    }

    pub fn summary(&self) -> RunSummary {
        let w0 = &self.walkers[0];
        let lead = self.lead().unwrap_or(w0);
        let (best_f, best_x) = self
            .walkers
            .iter()
            .filter(|w| w.best_f.is_finite())
            .min_by(|a, b| a.best_f.total_cmp(&b.best_f))
            .map_or((f64::NAN, w0.x_ok.clone()), |w| (w.best_f, w.best_x.clone()));
        RunSummary {
            health: w0.health,
            halted: self.walkers.iter().all(|w| w.halted),
            f: lead.f_now,
            grad_norm: lead.last.grad_norm,
            evals: self.walkers.iter().map(|w| w.evals).sum(),
            walkers: self.walkers.len(),
            failed: self
                .walkers
                .iter()
                .filter(|w| matches!(w.health, Health::Diverging | Health::Unbounded | Health::Diverged))
                .count(),
            converged: self.walkers.iter().filter(|w| w.still >= STILL_FOR).count(),
            restarts: self.walkers.iter().map(|w| w.restarts).sum(),
            best_f,
            best_x,
        }
    }

    /// The primary walker's iterate (the clicked start's run).
    pub fn x(&self) -> &[f64] {
        self.walkers[0].x()
    }
    pub fn halted(&self) -> bool {
        self.walkers.iter().all(|w| w.halted)
    }
    pub fn health(&self) -> Health {
        self.walkers[0].health
    }
    pub fn diagnosis(&self) -> Option<&Diagnosis> {
        self.walkers.iter().find_map(|w| w.diagnosis.as_ref())
    }
    /// Every walker's trail, separated by NaN pairs (restarts inside a trail are separated the same way).
    pub fn trails(&self) -> Vec<f64> {
        let mut out = vec![];
        for w in &self.walkers {
            if !out.is_empty() {
                out.extend([f64::NAN, f64::NAN]);
            }
            out.extend_from_slice(&w.trail);
        }
        out
    }
    /// Every walker's current point and health.
    pub fn points(&self) -> Vec<(Vec<f64>, Health, bool)> {
        self.walkers.iter().map(|w| (w.x().to_vec(), w.health, w.halted)).collect()
    }
    pub fn metrics(&self) -> &[[f64; 4]] {
        &self.metrics
    }
    /// Shapes to draw (Nelder–Mead's simplices), closed, separated by NaN pairs.
    pub fn shapes(&self) -> Vec<f64> {
        let mut out = vec![];
        for w in &self.walkers {
            if let Some(sh) = w.opt.shape() {
                if !out.is_empty() {
                    out.extend([f64::NAN, f64::NAN]);
                }
                out.extend_from_slice(&sh);
                out.extend_from_slice(&sh[..2]);
            }
        }
        out
    }
    pub fn manifest(&self) -> crate::OptimizerManifest {
        optimizers::by_id(&self.opt_id).expect("optimizer id").manifest()
    }
    fn last_trace(&self) -> Option<&Trace> {
        self.walkers[0].last_trace.as_ref()
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
    pub multistart: Multistart,
    /// timeline position, and the furthest position reached
    pub iter: u64,
    pub head: u64,
    pub selected: usize,
    /// the clicked start; `starts[0]` is the same point, the rest are random in the view box
    pub start: Vec<f64>,
    pub starts: Vec<Vec<f64>>,
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
        let start = manifest.start.clone();
        let mut s = Session {
            problem_id: problem_id.into(),
            coeffs: vec![],
            start: start.clone(),
            starts: vec![start],
            problem,
            manifest,
            view,
            zrange,
            runs: vec![],
            stops: Stops { on_diverge: true, ..Default::default() },
            multistart: Multistart::default(),
            iter: 0,
            head: 0,
            selected: 0,
            seed,
        };
        s.make_starts();
        Some(s)
    }

    pub fn problem(&self) -> &dyn Problem {
        self.problem.as_ref()
    }

    /// The clicked start, then `starts − 1` uniform random points in the view box (from the session seed, so
    /// the same for every run and every replay).
    fn make_starts(&mut self) {
        let mut r = Rng::new(self.seed ^ 0x7374_6172);
        let n = self.multistart.starts.clamp(1, MAX_STARTS);
        self.starts = std::iter::once(self.start.clone())
            .chain((1..n).map(|_| self.view.iter().map(|&(lo, hi)| r.range(lo, hi)).collect()))
            .collect();
    }

    /// Switches problem (or coefficients); keeps the runs, restarts them.
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
        self.make_starts();
        self.restart();
        true
    }

    /// Number of starting points and the restart policy; restarts the session.
    pub fn set_multistart(&mut self, ms: Multistart) {
        self.multistart = Multistart { starts: ms.starts.clamp(1, MAX_STARTS), ..ms };
        self.make_starts();
        self.restart();
    }

    pub fn add_run(&mut self, opt_id: &str, params: &[(String, f64)]) -> Option<usize> {
        if self.runs.len() >= MAX_RUNS {
            return None;
        }
        let seed = self.seed.wrapping_add(self.runs.len() as u64 * 0x9e37_79b9);
        let mut r = Run::new(opt_id, params, seed)?;
        let bounds = self.view.to_vec();
        r.reset(self.problem.as_ref(), &self.starts, &bounds, self.multistart, self.seed);
        // a run added mid-session catches up with the others
        for k in 0..self.iter {
            if r.halted() {
                break;
            }
            r.step(self.problem.as_ref(), k, false, self.multistart, &bounds);
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

    /// Changes a run's optimizer (keeping parameters that share a key, except the step size) and restarts.
    pub fn set_optimizer(&mut self, i: usize, opt_id: &str) -> bool {
        let Some(old) = self.runs.get(i) else { return false };
        let keep: Vec<(String, f64)> = old.params.iter().filter(|(k, _)| k != "lr").cloned().collect();
        let seed = old.seed;
        let Some(mut r) = Run::new(opt_id, &[], seed) else { return false };
        for (k, v) in keep {
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
        let v = r.get(key).unwrap_or(v);
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
        for w in &mut r.walkers {
            w.reported = false;
        }
        true
    }

    pub fn set_start(&mut self, x0: [f64; 2]) {
        self.start = x0.to_vec();
        self.make_starts();
        self.restart();
    }

    pub fn restart(&mut self) {
        let bounds = self.view.to_vec();
        for r in &mut self.runs {
            r.reset(self.problem.as_ref(), &self.starts, &bounds, self.multistart, self.seed);
        }
        self.iter = 0;
        self.head = 0;
    }

    /// Advances every live run by up to `n` iterations; stops early at a stopping point. Returns (iterations
    /// done, the stop if one fired).
    pub fn step(&mut self, n: u64) -> (u64, Option<Stop>) {
        let bounds = self.view.to_vec();
        let ms = self.multistart;
        // health stops only make sense for a single walker that isn't restarted on failure
        let health_stops = ms.starts == 1 && !ms.on_failed;
        for done in 0..n {
            if self.runs.iter().all(|r| r.halted()) {
                return (done, None);
            }
            let k = self.iter;
            let mut stop = None;
            for (i, r) in self.runs.iter_mut().enumerate() {
                if r.halted() {
                    continue;
                }
                let trace = self.stops.line.is_some() && i == self.selected;
                let h = r.step(self.problem.as_ref(), k, trace, ms, &bounds);
                if health_stops && matches!(h, Health::Diverged | Health::Diverging | Health::Unbounded) {
                    let w = &mut r.walkers[0];
                    if !w.reported {
                        w.reported = true;
                        let params = r.params.clone();
                        let get = |key: &str| params.iter().find(|(k, _)| k == key).map(|p| p.1);
                        w.diagnosis = Some(diagnose(h, self.problem.as_ref(), w.opt.as_ref(), &w.x_ok, &get));
                        if self.stops.on_diverge || h == Health::Diverged {
                            stop.get_or_insert(Stop::Health { run: i, health: h });
                        }
                    }
                }
                let s = r.summary();
                if let Some(fb) = self.stops.f_below
                    && s.f < fb
                {
                    stop.get_or_insert(Stop::FBelow { run: i, f: s.f });
                }
                if let Some(gb) = self.stops.grad_below
                    && s.grad_norm < gb
                {
                    stop.get_or_insert(Stop::GradBelow { run: i, grad: s.grad_norm });
                }
                if let (Some(line), Some(t)) = (self.stops.line, r.last_trace())
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

    /// Moves the timeline to iteration k (≤ head) by replaying every run from its starts.
    pub fn seek(&mut self, k: u64) {
        let k = k.min(self.head);
        let head = self.head;
        let bounds = self.view.to_vec();
        let ms = self.multistart;
        for r in &mut self.runs {
            r.reset(self.problem.as_ref(), &self.starts, &bounds, ms, self.seed);
            for j in 0..k {
                if r.halted() {
                    break;
                }
                r.step(self.problem.as_ref(), j, false, ms, &bounds);
                for w in &mut r.walkers {
                    if matches!(w.health, Health::Diverging | Health::Unbounded | Health::Diverged) {
                        w.reported = true;
                    }
                }
            }
        }
        self.iter = k;
        self.head = head;
    }

    /// The trace of run i's primary walker from iteration iter−1 to iter (replayed on a copy), for inspection.
    pub fn inspect(&self, i: usize) -> Option<Trace> {
        let r0 = self.runs.get(i)?;
        if self.iter == 0 {
            return Some(Trace::default());
        }
        let mut r = Run::new(&r0.opt_id, &[], r0.seed)?;
        r.log = r0.log.clone();
        r.params = r0.params.clone();
        let bounds = self.view.to_vec();
        let ms = self.multistart;
        r.reset(self.problem.as_ref(), &self.starts[..1], &bounds, ms, self.seed);
        for j in 0..self.iter - 1 {
            if r.halted() {
                return Some(Trace::default());
            }
            r.step(self.problem.as_ref(), j, false, ms, &bounds);
        }
        r.step(self.problem.as_ref(), self.iter - 1, true, ms, &bounds);
        r.walkers[0].last_trace.clone()
    }

    /// Enlarges the view box to take in every trail (finite points), so runaway runs stay visible.
    pub fn fit_view(&mut self) {
        let [mut bx, mut by] = [self.manifest.bounds[0], self.manifest.bounds[1]];
        for r in &self.runs {
            for w in &r.walkers {
                for p in w.trail.chunks(2) {
                    if p[0].is_finite() && p[1].is_finite() && p[0].abs() < 1e12 && p[1].abs() < 1e12 {
                        bx = (bx.0.min(p[0]), bx.1.max(p[0]));
                        by = (by.0.min(p[1]), by.1.max(p[1]));
                    }
                }
            }
        }
        let (wx, wy) = (bx.1 - bx.0, by.1 - by.0);
        let m = 0.08 * wx.max(wy);
        self.view = [(bx.0 - m, bx.1 + m), (by.0 - m, by.1 + m)];
        self.after_view_change();
    }

    pub fn reset_view(&mut self) {
        self.view = [self.manifest.bounds[0], self.manifest.bounds[1]];
        self.after_view_change();
    }

    fn after_view_change(&mut self) {
        self.zrange = problems::height_range(self.problem.as_ref(), self.view, 129);
        for r in &mut self.runs {
            for w in &mut r.walkers {
                w.monitor.set_bounds(&self.view);
            }
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
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        let (mut s0, s_end) = (0.0, 12.0 / len);
        let steps = 2000;
        let ds = (s_end - s0) / steps as f64;
        let mut prev = None;
        for _ in 0..steps {
            let p = at(s0);
            if p[0].abs() <= 1.0 && p[2].abs() <= 1.0 {
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

    fn xs(s: &Session) -> Vec<Vec<f64>> {
        s.runs.iter().map(|r| r.x().to_vec()).collect()
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
        let at100 = xs(&s);
        s.step(50);
        s.seek(100);
        assert_eq!(s.iter, 100);
        assert_eq!(s.head, 150);
        assert_eq!(at100, xs(&s));
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
        assert_eq!((s.iter, s.head), (40, 40));
    }

    #[test]
    fn divergence_stops_once_with_a_diagnosis_and_a_suggestion() {
        let mut s = session("bowl", &["gd"]);
        s.set_param(0, "lr", 0.11); // > 2/L = 0.1
        let (_, stop) = s.step(10_000);
        assert!(matches!(stop, Some(Stop::Health { run: 0, health: Health::Diverging })), "{stop:?}");
        let d = s.runs[0].diagnosis().cloned().unwrap();
        assert_eq!(d.suggest.unwrap().0, "lr");
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
            halted |= s.runs[0].halted();
        }
        assert!(halted, "gd with η = 0.5 on Rosenbrock overflows");
        assert!(s.runs[0].walkers[0].x_ok.iter().all(|v| v.is_finite()));
        assert!(!s.runs[1].halted());
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
        assert_eq!(s.step(10), (1, Some(Stop::Line { run: 0, line: 12 })));
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

    /* ---- multistart and random restarts ---- */

    #[test]
    fn several_starts_are_shared_by_every_run_and_inside_the_view() {
        let mut s = session("himmelblau", &["adam", "gd"]);
        s.set_multistart(Multistart { starts: 16, ..Default::default() });
        assert_eq!(s.starts.len(), 16);
        assert_eq!(s.starts[0], s.start);
        for r in &s.runs {
            assert_eq!(r.walkers.len(), 16);
            for (w, x0) in r.walkers.iter().zip(&s.starts) {
                assert_eq!(&w.trail[..2], &x0[..]);
            }
        }
        for x in &s.starts {
            assert!(x.iter().zip(&s.view).all(|(v, (lo, hi))| lo <= v && v <= hi));
        }
    }

    #[test]
    fn many_starts_on_himmelblau_find_all_four_minima() {
        let mut s = session("himmelblau", &["adam"]);
        s.set_multistart(Multistart { starts: 32, ..Default::default() });
        s.step(3000);
        let minima = s.manifest.minima.clone();
        let pts = s.runs[0].points();
        for m in &minima {
            assert!(pts.iter().any(|(x, _, _)| (x[0] - m[0]).hypot(x[1] - m[1]) < 1e-2), "no walker at {m:?}");
        }
        assert!(s.runs[0].summary().best_f < 1e-6);
    }

    #[test]
    fn restarts_on_convergence_keep_searching_and_keep_the_best() {
        let mut s = session("rastrigin", &["gd"]);
        s.set_param(0, "lr", 0.002);
        s.set_multistart(Multistart { starts: 1, on_converged: true, on_failed: false });
        s.step(20_000);
        let sum = s.runs[0].summary();
        assert!(sum.restarts > 5, "restarted {} times", sum.restarts);
        // the best ever is at least as good as a single descent from the start would get
        let mut single = session("rastrigin", &["gd"]);
        single.set_param(0, "lr", 0.002);
        single.step(20_000);
        assert!(sum.best_f <= single.runs[0].summary().f + 1e-12);
        // and the best curve in the metrics never goes up
        let m = s.runs[0].metrics();
        assert!(m.windows(2).all(|w| w[1][1] <= w[0][1] + 1e-12));
        // restarts are separated in the trail
        assert!(s.runs[0].trails().iter().any(|v| v.is_nan()));
    }

    #[test]
    fn restarts_on_failure_replace_divergence_stops() {
        let mut s = session("bowl", &["gd"]);
        s.set_param(0, "lr", 0.11);
        s.set_multistart(Multistart { starts: 1, on_converged: false, on_failed: true });
        let (n, stop) = s.step(2000);
        assert_eq!((n, stop), (2000, None));
        assert!(s.runs[0].summary().restarts > 0);
    }

    #[test]
    fn multistart_replays_exactly() {
        let mut s = session("ackley", &["adam", "nelder-mead"]);
        s.set_multistart(Multistart { starts: 8, on_converged: true, on_failed: true });
        s.step(500);
        let before: Vec<_> = s.runs.iter().map(|r| (r.trails(), r.summary().best_f)).collect();
        s.step(100);
        s.seek(500);
        let after: Vec<_> = s.runs.iter().map(|r| (r.trails(), r.summary().best_f)).collect();
        assert_eq!(format!("{before:?}"), format!("{after:?}"));
    }
}
