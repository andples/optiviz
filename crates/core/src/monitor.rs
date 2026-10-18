//! Watching a run's health: is it fine, outside the view, running away, or broken? And when it runs away, why,
//! and what would fix it. See DESIGN.md §4.3 and docs/adr/0006-divergence.md.
//!
//! - **Diverged**: x, f or ∇f is no longer finite. The run cannot continue; it always stops, keeping the last
//!   finite state.
//! - **Diverging**: still finite but clearly running away: f has risen far above where it started, or x has
//!   travelled far beyond the landscape, or the step length keeps growing geometrically. A default stopping
//!   point; the run pauses with an explanation and a suggested fix.
//! - **Unbounded**: f keeps falling while x runs off to infinity: the function has no minimum in that direction
//!   (a saddle, a downhill ramp). Not the optimizer's fault; reported as such.
//! - **OutOfView**: x left the drawn box but nothing else is wrong. Shown, not stopped.

use crate::{Problem, optimizers::Optimizer};
use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Health {
    Ok,
    OutOfView,
    Diverging,
    Unbounded,
    Diverged,
}

/// One step's observation.
pub struct Obs<'a> {
    pub x: &'a [f64],
    pub f: f64,
    pub grad_norm: f64,
    pub step_len: f64,
}

pub struct Monitor {
    x0: Vec<f64>,
    f0: f64,
    f_best: f64,
    /// view box diameter: the yardstick for "far"
    scale: f64,
    pub(crate) bounds: Vec<(f64, f64)>,
    grow: u32,
    fall: u32,
    last_step: f64,
    last_f: f64,
}

impl Monitor {
    pub fn new(x0: &[f64], f0: f64, bounds: &[(f64, f64)]) -> Self {
        let scale = bounds.iter().map(|(a, b)| (b - a) * (b - a)).sum::<f64>().sqrt();
        Monitor {
            x0: x0.to_vec(),
            f0,
            f_best: f0,
            scale,
            bounds: bounds.to_vec(),
            grow: 0,
            fall: 0,
            last_step: 0.0,
            last_f: f0,
        }
    }

    pub fn observe(&mut self, o: &Obs) -> Health {
        let finite = o.x.iter().all(|v| v.is_finite()) && o.f.is_finite() && !o.grad_norm.is_infinite();
        if !finite {
            return Health::Diverged;
        }
        let dist = o.x.iter().zip(&self.x0).map(|(a, b)| (a - b) * (a - b)).sum::<f64>().sqrt();
        // geometric growth of the step: ×1.2 or more for 8 steps in a row, and already a sizeable step
        if o.step_len > self.last_step * 1.2 && o.step_len > 0.02 * self.scale {
            self.grow += 1;
        } else {
            self.grow = 0;
        }
        if o.f < self.last_f {
            self.fall += 1;
        } else {
            self.fall = 0;
        }
        self.last_step = o.step_len;
        self.last_f = o.f;
        self.f_best = self.f_best.min(o.f);
        let far = dist > 20.0 * self.scale;
        // unbounded below: steadily falling f, far away, and f well below anything seen near the start
        if far && self.fall >= 5 && o.f < self.f0 - 10.0 * (1.0 + self.f0.abs()) {
            return Health::Unbounded;
        }
        let risen = o.f > self.f0.max(self.f_best) + 1e3 * (1.0 + self.f0.abs());
        if risen || far || self.grow >= 8 {
            return Health::Diverging;
        }
        let outside = o.x.iter().zip(&self.bounds).any(|(v, (lo, hi))| v < lo || v > hi);
        if outside { Health::OutOfView } else { Health::Ok }
    }
}

/// Why a run is diverging and what to try, in prose with inline TeX (`$...$`).
#[derive(Clone, Debug, Serialize)]
pub struct Diagnosis {
    pub health: Health,
    pub text: String,
    /// A parameter change that should fix it: (key, value)
    pub suggest: Option<(String, f64)>,
}

/// Largest absolute eigenvalue of the Hessian at x (2D: closed form; N-D: power iteration).
pub fn curvature(p: &dyn Problem, x: &[f64]) -> f64 {
    let n = x.len();
    let mut h = vec![0.0; n * n];
    p.hess(x, &mut h);
    if n == 2 {
        let (a, b, d) = (h[0], h[1], h[3]);
        let m = 0.5 * (a + d);
        let r = (0.25 * (a - d) * (a - d) + b * b).sqrt();
        return (m + r).abs().max((m - r).abs());
    }
    let mut v = vec![1.0 / (n as f64).sqrt(); n];
    let mut lam = 0.0;
    for _ in 0..50 {
        let w: Vec<f64> = (0..n).map(|i| (0..n).map(|j| h[i * n + j] * v[j]).sum()).collect();
        lam = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if lam == 0.0 {
            break;
        }
        v = w.iter().map(|x| x / lam).collect();
    }
    lam
}

fn sci(v: f64) -> String {
    if v == 0.0 || (1e-3..1e4).contains(&v.abs()) {
        let s = format!("{v:.3}");
        let s = s.trim_end_matches('0').trim_end_matches('.');
        return s.to_string();
    }
    let e = v.abs().log10().floor() as i32;
    format!("{:.2}\\cdot 10^{{{e}}}", v / 10f64.powi(e))
}

/// Explains a Diverging / Diverged / Unbounded run. `x_ok` is the last finite iterate; `params` the
/// optimizer's current values by key.
pub fn diagnose(
    health: Health,
    p: &dyn Problem,
    opt: &dyn Optimizer,
    x_ok: &[f64],
    get: &dyn Fn(&str) -> Option<f64>,
) -> Diagnosis {
    let id = opt.manifest().id;
    if health == Health::Unbounded {
        return Diagnosis {
            health,
            text: "$f$ keeps decreasing as $x$ moves away: the function is unbounded below in this direction, so there is no minimum to find. The optimizer is doing its job; the landscape has no bottom.".into(),
            suggest: None,
        };
    }
    let l = curvature(p, x_ok);
    let lr = get("lr");
    let mut text = String::new();
    let mut suggest = None;
    if health == Health::Diverged {
        text += "The iterate stopped being a finite number (overflow to $\\pm\\infty$ or NaN). ";
    } else {
        text += "The run is running away from the landscape. ";
    }
    match (id, lr) {
        ("gd", Some(eta)) if get("ls") == Some(0.0) => {
            let limit = 2.0 / l;
            text += &format!(
                "With a fixed step, gradient descent is only stable where $\\eta < 2/L$, with $L$ the largest curvature of $f$. Near the last good point $L \\approx {}$, so $\\eta$ must be below ${}$; it is ${}$.",
                sci(l),
                sci(limit),
                sci(eta)
            );
            if eta >= limit {
                suggest = Some(("lr".into(), 0.5 * limit));
            }
        }
        ("momentum" | "nesterov", Some(eta)) => {
            let beta = get("beta").or_else(|| get("mu")).unwrap_or(0.9);
            let limit = 2.0 * (1.0 + beta) / l;
            text += &format!(
                "With momentum $\\beta = {}$ the step is stable on a quadratic only while $\\eta < 2(1+\\beta)/L$. Near the last good point $L \\approx {}$, so $\\eta$ must be below ${}$; it is ${}$.",
                sci(beta),
                sci(l),
                sci(limit),
                sci(eta)
            );
            if eta >= limit {
                suggest = Some(("lr".into(), 0.5 * limit));
            }
        }
        ("adam" | "adamw" | "rmsprop" | "adagrad", Some(eta)) => {
            text += &format!(
                "Adaptive methods move each coordinate by roughly $\\eta$ per step whatever the gradient's size, so a large $\\eta = {}$ overshoots narrow valleys and can be thrown out of the landscape. Try a smaller $\\eta$.",
                sci(eta)
            );
            suggest = Some(("lr".into(), eta / 4.0));
        }
        ("newton", _) => {
            text += &format!(
                "Newton's step uses the local quadratic model, which is poor far from a minimum (curvature here $\\approx {}$). The backtracking search should prevent this; it suggests non-finite values in the landscape.",
                sci(l)
            );
        }
        _ => {
            text += &format!("The local curvature near the last good point is about ${}$.", sci(l));
        }
    }
    Diagnosis { health, text, suggest }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Rng, optimizers, problems};

    fn watch(problem: &str, opt: &str, lr: f64, steps: usize) -> (Health, usize) {
        let p = problems::by_id(problem, 2).unwrap();
        let m = p.manifest();
        let mut o = optimizers::by_id(opt).unwrap();
        o.set_param("lr", lr);
        o.init(&m.start);
        let mut mon = Monitor::new(&m.start, p.eval(&m.start), &m.bounds);
        let mut rng = Rng::new(0);
        let mut prev = m.start.clone();
        for k in 0..steps {
            let info = o.step(p.as_ref(), &mut rng, None);
            let x = o.x().to_vec();
            let sl = x.iter().zip(&prev).map(|(a, b)| (a - b) * (a - b)).sum::<f64>().sqrt();
            let h = mon.observe(&Obs { x: &x, f: p.eval(&x), grad_norm: info.grad_norm, step_len: sl });
            if matches!(h, Health::Diverging | Health::Diverged | Health::Unbounded) {
                return (h, k);
            }
            prev = x;
        }
        (Health::Ok, steps)
    }

    #[test]
    fn a_stable_run_stays_ok() {
        assert_eq!(watch("bowl", "gd", 0.04, 2000).0, Health::Ok);
        assert_eq!(watch("rosenbrock", "adam", 0.05, 2000).0, Health::Ok);
    }

    #[test]
    fn too_large_a_step_is_caught_early_as_diverging() {
        // bowl: L = 20, so η = 0.11 > 2/L = 0.1 diverges geometrically
        let (h, k) = watch("bowl", "gd", 0.11, 5000);
        assert_eq!(h, Health::Diverging);
        assert!(k < 200, "caught at step {k}");
    }

    #[test]
    fn a_saddle_is_reported_as_unbounded_not_as_a_bad_step_size() {
        let (h, _) = watch("saddle", "gd", 0.05, 20000);
        assert_eq!(h, Health::Unbounded);
    }

    #[test]
    fn the_diagnosis_suggests_a_stable_step_size() {
        let p = problems::by_id("bowl", 2).unwrap();
        let o = optimizers::by_id("gd").unwrap();
        let get = |k: &str| match k {
            "lr" => Some(0.11),
            "ls" => Some(0.0),
            _ => None,
        };
        let d = diagnose(Health::Diverging, p.as_ref(), o.as_ref(), &[0.1, 0.1], &get);
        let (key, v) = d.suggest.unwrap();
        assert_eq!(key, "lr");
        assert!((v - 0.05).abs() < 1e-6, "{v}"); // half of 2/L with L = 20
        assert!(d.text.contains("2/L"));
    }
}
