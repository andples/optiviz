//! Trust-region Newton method (Nocedal & Wright 2006, Algorithm 4.1), with three ways of solving the subproblem:
//! the Cauchy point (§4.1, Algorithm 4.2), the dogleg (§4.1, eqs. 4.15–4.16) and the exact solution (§4.3,
//! Theorem 4.1).
//!
//! Drawn on the surface (`shape`): the trust region around the current iterate, the last step's path (the dogleg
//! goes x → Cauchy point → Newton point), and a cross where a rejected trial step landed.

use super::{Optimizer, dot, no_stopping_test, norm};
use crate::manifest::{Deviation, OptimizerManifest, ParamSpec, Pseudocode, lines};
use crate::trace::Trace;
use crate::{Problem, Rng, StepInfo, tr};

pub struct TrustRegion {
    x: Vec<f64>,
    delta: f64,
    pub delta0: f64,
    pub delta_max: f64,
    pub eta: f64,
    /// 0 Cauchy point, 1 dogleg, 2 exact
    pub solver: f64,
    /// what the last step tried, for drawing: its centre, path (centre → … → trial), and whether it was accepted
    last_center: Vec<f64>,
    last_delta: f64,
    last_path: Vec<f64>,
    last_accepted: bool,
}

impl Default for TrustRegion {
    fn default() -> Self {
        Self {
            x: vec![],
            delta: 0.5,
            delta0: 0.5,
            delta_max: 4.0,
            eta: 0.1,
            solver: 1.0,
            last_center: vec![],
            last_delta: 0.5,
            last_path: vec![],
            last_accepted: true,
        }
    }
}

/// m(p) − f = gᵀp + ½ pᵀBp
fn model(g: &[f64], b: &[f64], p: &[f64]) -> f64 {
    let n = g.len();
    let mut bp = 0.0;
    for i in 0..n {
        for j in 0..n {
            bp += p[i] * b[i * n + j] * p[j];
        }
    }
    dot(g, p) + 0.5 * bp
}

fn mat_vec(b: &[f64], v: &[f64]) -> Vec<f64> {
    let n = v.len();
    (0..n).map(|i| (0..n).map(|j| b[i * n + j] * v[j]).sum()).collect()
}

/// Cauchy point (N&W Algorithm 4.2): p = −τ (Δ/‖g‖) g, τ = 1 if gᵀBg ≤ 0, else min(‖g‖³ / (Δ gᵀBg), 1).
fn cauchy(g: &[f64], b: &[f64], delta: f64) -> Vec<f64> {
    let gn = norm(g);
    if gn == 0.0 {
        return vec![0.0; g.len()];
    }
    let gbg = dot(g, &mat_vec(b, g));
    let tau = if gbg <= 0.0 { 1.0 } else { (gn.powi(3) / (delta * gbg)).min(1.0) };
    g.iter().map(|gi| -tau * delta / gn * gi).collect()
}

/// Symmetric eigendecomposition by cyclic Jacobi rotations: (eigenvalues, eigenvectors as columns, row-major).
fn eigen_sym(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = a.to_vec();
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _ in 0..100 {
        let off: f64 = (0..n)
            .flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| a[i * n + j].powi(2))
            .sum();
        if off < 1e-30 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let (c, s) = (1.0 / (t * t + 1.0).sqrt(), t / (t * t + 1.0).sqrt());
                for k in 0..n {
                    let (akp, akq) = (a[k * n + p], a[k * n + q]);
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p * n + k], a[q * n + k]);
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let (vkp, vkq) = (v[k * n + p], v[k * n + q]);
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    ((0..n).map(|i| a[i * n + i]).collect(), v)
}

/// The exact subproblem solution (N&W Theorem 4.1): p = −(B + λI)⁻¹ g with λ ≥ 0, B + λI positive semidefinite,
/// λ(Δ − ‖p‖) = 0; λ found from the eigendecomposition by bisection on ‖p(λ)‖ = Δ, the hard case (N&W eq. 4.45)
/// handled by moving along the lowest eigenvector to the boundary. Returns (p, λ).
fn exact(g: &[f64], b: &[f64], delta: f64) -> (Vec<f64>, f64) {
    let n = g.len();
    let (lam, q) = eigen_sym(b, n);
    let col = |k: usize| -> Vec<f64> { (0..n).map(|i| q[i * n + k]).collect() };
    let qg: Vec<f64> = (0..n).map(|k| dot(&col(k), g)).collect();
    let p_of = |l: f64| -> Vec<f64> {
        let mut p = vec![0.0; n];
        for k in 0..n {
            let ck = -qg[k] / (lam[k] + l);
            for (pi, qi) in p.iter_mut().zip(col(k)) {
                *pi += ck * qi;
            }
        }
        p
    };
    let lmin = lam.iter().copied().fold(f64::INFINITY, f64::min);
    let kmin = lam.iter().position(|&l| l == lmin).unwrap_or(0);
    // interior Newton step if B is positive definite and the step fits
    if lmin > 0.0 {
        let p = p_of(0.0);
        if norm(&p) <= delta {
            return (p, 0.0);
        }
    }
    let lo0 = (-lmin).max(0.0);
    let gn = norm(g);
    // hard case: g has (almost) no component along the lowest eigenvector and the step at λ = −λ_min is short
    if qg[kmin].abs() <= 1e-12 * gn.max(1e-300) {
        let mut p = vec![0.0; n];
        for k in 0..n {
            if k != kmin && (lam[k] - lmin).abs() > 1e-14 {
                let ck = -qg[k] / (lam[k] - lmin);
                for (pi, qi) in p.iter_mut().zip(col(k)) {
                    *pi += ck * qi;
                }
            }
        }
        let pn = norm(&p);
        if pn <= delta {
            let tau = (delta * delta - pn * pn).max(0.0).sqrt();
            for (pi, qi) in p.iter_mut().zip(col(kmin)) {
                *pi += tau * qi;
            }
            return (p, lo0);
        }
    }
    // bisection on ‖p(λ)‖ = Δ over (lo0, hi]
    let mut lo = lo0 + 1e-14 * (1.0 + lo0);
    let mut hi = lo0 + gn / delta + 1.0;
    while norm(&p_of(hi)) > delta {
        hi *= 2.0;
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if norm(&p_of(mid)) > delta {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo < 1e-15 * hi.max(1.0) {
            break;
        }
    }
    (p_of(hi), hi)
}

impl Optimizer for TrustRegion {
    fn manifest(&self) -> OptimizerManifest {
        OptimizerManifest {
            id: "trust-region",
            name: "Trust region",
            family: "second-order",
            params: vec![
                ParamSpec::choice(
                    "solver",
                    r"\text{subproblem}",
                    "how the step inside the region is chosen",
                    1,
                    &["Cauchy point", "dogleg", "exact"],
                ),
                ParamSpec::real("delta0", r"\Delta_0", "initial trust-region radius", 0.5, 1e-3, 4.0, true),
                ParamSpec::real("delta_max", r"\hat\Delta", "largest radius allowed", 4.0, 0.01, 20.0, true),
                ParamSpec::real(
                    "eta",
                    r"\eta",
                    "a step is accepted when the actual decrease is more than this share of the predicted one",
                    0.1,
                    0.0,
                    0.249,
                    false,
                ),
            ],
            tex: r"p_k = \arg\min_{\lVert p \rVert \le \Delta_k} f_k + g_k^\top p + \tfrac12 p^\top B_k p,\quad \rho_k = \frac{f(x_k) - f(x_k + p_k)}{m_k(0) - m_k(p_k)}",
            pseudocode: Pseudocode {
                source: "Nocedal & Wright (2006), Numerical Optimization, 2nd ed., Algorithm 4.1 (trust region); subproblem by the Cauchy point (Algorithm 4.2), the dogleg (§4.1) or exactly (§4.3, Theorem 4.1)",
                lines: lines(&[
                    (
                        0,
                        false,
                        r"\text{Given } \hat\Delta > 0,\ \Delta_0 \in (0, \hat\Delta), \text{ and } \eta \in [0, \tfrac14)",
                    ),
                    (0, false, r"\textbf{for } k = 0, 1, 2, \dots"),
                    (
                        1,
                        true,
                        r"\text{Obtain } p_k \text{ by (approximately) solving } \min_{\lVert p \rVert \le \Delta_k} m_k(p) = f_k + g_k^\top p + \tfrac12 p^\top B_k p \quad (4.3)",
                    ),
                    (1, true, r"\text{Evaluate } \rho_k = \frac{f(x_k) - f(x_k + p_k)}{m_k(0) - m_k(p_k)} \quad (4.4)"),
                    (1, true, r"\textbf{if } \rho_k < \tfrac14:\ \Delta_{k+1} = \tfrac14 \Delta_k"),
                    (
                        1,
                        true,
                        r"\textbf{else if } \rho_k > \tfrac34 \text{ and } \lVert p_k \rVert = \Delta_k:\ \Delta_{k+1} = \min(2\Delta_k, \hat\Delta)",
                    ),
                    (1, true, r"\textbf{else } \Delta_{k+1} = \Delta_k"),
                    (1, true, r"\textbf{if } \rho_k > \eta:\ x_{k+1} = x_k + p_k"),
                    (1, true, r"\textbf{else } x_{k+1} = x_k"),
                    (0, false, r"\textbf{end (for)}"),
                ]),
                deviations: vec![
                    Deviation {
                        line: Some(2),
                        what: "$B_k = \\nabla^2 f(x_k)$, the exact Hessian (by central differences of the analytic gradient): a trust-region Newton method. The subproblem is solved by the choice you make: the Cauchy point (Algorithm 4.2), the dogleg (eqs. 4.15–4.16), or exactly (Theorem 4.1).",
                        why: "Algorithm 4.1 leaves $B_k$ and the subproblem solver open. The exact Hessian makes the model honest, so $\\rho_k$ shows how good the quadratic model is; the three solvers are the three the chapter develops.",
                    },
                    Deviation {
                        line: Some(2),
                        what: "Dogleg with an indefinite $B_k$ falls back to the Cauchy point. The exact solver uses the eigendecomposition of $B_k$ and bisection on $\\lVert p(\\lambda) \\rVert = \\Delta_k$ (hard case by eq. 4.45), not the Cholesky-based iteration of Algorithm 4.3.",
                        why: "The dogleg is only defined for positive definite $B$ (§4.1); the Cauchy point is the step the chapter's convergence theory rests on. In two dimensions the eigendecomposition is exact and simpler, and gives the same solution as Algorithm 4.3.",
                    },
                    Deviation {
                        line: Some(5),
                        what: "$\\lVert p_k \\rVert = \\Delta_k$ is tested as $\\lVert p_k \\rVert \\ge (1 - 10^{-9})\\,\\Delta_k$.",
                        why: "In floating point a step on the boundary is rarely exactly $\\Delta_k$ long.",
                    },
                    Deviation {
                        line: Some(3),
                        what: "If the predicted decrease $m_k(0) - m_k(p_k)$ is zero (at a stationary point), $\\rho_k$ is taken as 0: the step is rejected and the radius shrinks.",
                        why: "(4.4) is undefined there; rejecting keeps $x_k$, which is already stationary.",
                    },
                    no_stopping_test(1),
                ],
            },
        }
    }
    fn init(&mut self, x0: &[f64]) {
        self.x = x0.to_vec();
        self.delta = self.delta0.min(self.delta_max);
        self.last_center = x0.to_vec();
        self.last_path.clear();
        self.last_accepted = true;
    }
    fn step(&mut self, p: &dyn Problem, _: &mut Rng, mut trc: Option<&mut Trace>) -> StepInfo {
        let n = self.x.len();
        let mut g = vec![0.0; n];
        let f = p.grad(&self.x, &mut g);
        let mut b = vec![0.0; n * n];
        p.hess(&self.x, &mut b);
        let delta = self.delta;
        // the subproblem; path = the points the step goes through, for drawing
        let pc = cauchy(&g, &b, delta);
        let (step, path, extra) = match self.solver.round() as u32 {
            0 => (pc.clone(), vec![pc.clone()], 0.0),
            2 => {
                let (s, lam) = exact(&g, &b, delta);
                (s.clone(), vec![s], lam)
            }
            _ => {
                // dogleg: needs B positive definite
                match super::newton::cholesky(&b, n) {
                    Some(l) => {
                        let neg_g: Vec<f64> = g.iter().map(|v| -v).collect();
                        let pb = super::newton::chol_solve(&l, n, &neg_g);
                        let gbg = dot(&g, &mat_vec(&b, &g));
                        let pu: Vec<f64> = g.iter().map(|v| -dot(&g, &g) / gbg * v).collect();
                        if norm(&pb) <= delta {
                            (pb.clone(), vec![pu, pb], 0.0)
                        } else if norm(&pu) >= delta {
                            let s: Vec<f64> = pu.iter().map(|v| v * delta / norm(&pu)).collect();
                            (s.clone(), vec![s], 0.0)
                        } else {
                            // τ − 1 = s with ‖pu + s (pb − pu)‖ = Δ
                            let d: Vec<f64> = pb.iter().zip(&pu).map(|(a, c)| a - c).collect();
                            let (a2, b2, c2) = (dot(&d, &d), 2.0 * dot(&pu, &d), dot(&pu, &pu) - delta * delta);
                            let s = (-b2 + (b2 * b2 - 4.0 * a2 * c2).max(0.0).sqrt()) / (2.0 * a2);
                            let st: Vec<f64> = pu.iter().zip(&d).map(|(u, di)| u + s * di).collect();
                            (st.clone(), vec![pu, st], 0.0)
                        }
                    }
                    None => (pc.clone(), vec![pc.clone()], 0.0),
                }
            }
        };
        tr!(trc, 2, r"g_k" => g, r"B_k" => b, r"\Delta_k" => delta, r"p^C" => pc, r"p_k" => step, r"\lambda" => extra);
        let trial: Vec<f64> = self.x.iter().zip(&step).map(|(a, c)| a + c).collect();
        let f_trial = p.eval(&trial);
        let predicted = -model(&g, &b, &step);
        let actual = f - f_trial;
        let rho = if predicted > 0.0 && f_trial.is_finite() { actual / predicted } else { 0.0 };
        tr!(trc, 3, r"f(x_k) - f(x_k + p_k)" => actual, r"m_k(0) - m_k(p_k)" => predicted, r"\rho_k" => rho);
        let on_boundary = norm(&step) >= (1.0 - 1e-9) * delta;
        if rho < 0.25 {
            self.delta = 0.25 * delta;
            tr!(trc, 4, r"\Delta_{k+1}" => self.delta);
        } else if rho > 0.75 && on_boundary {
            self.delta = (2.0 * delta).min(self.delta_max);
            tr!(trc, 5, r"\Delta_{k+1}" => self.delta);
        } else {
            tr!(trc, 6, r"\Delta_{k+1}" => self.delta);
        }
        // remember the attempt for drawing
        self.last_center = self.x.clone();
        self.last_delta = delta;
        self.last_path =
            path.into_iter().flat_map(|q| self.x.iter().zip(q).map(|(a, c)| a + c).collect::<Vec<_>>()).collect();
        self.last_accepted = rho > self.eta;
        if self.last_accepted {
            self.x = trial;
            tr!(trc, 7, r"x_{k+1}" => self.x);
        } else {
            tr!(trc, 8, r"x_{k+1}" => self.x);
        }
        StepInfo { f, grad_norm: norm(&g), evals: 2 }
    }
    fn x(&self) -> &[f64] {
        &self.x
    }
    fn set_param(&mut self, key: &str, v: f64) -> bool {
        match key {
            "solver" => self.solver = v,
            "delta0" => self.delta0 = v,
            "delta_max" => self.delta_max = v,
            "eta" => self.eta = v,
            _ => return false,
        }
        true
    }
    /// The trust region around the current iterate (circle, radius Δ, in the first two coordinates), the region
    /// the last step searched (dashed), that step's path from its centre, and a cross on a rejected trial point.
    fn shape(&self) -> Option<Vec<f64>> {
        if self.x.len() < 2 {
            return None;
        }
        let nan = [f64::NAN, f64::NAN];
        let mut out = vec![];
        let segs = 72;
        for k in 0..=segs {
            let a = std::f64::consts::TAU * k as f64 / segs as f64;
            out.extend([self.x[0] + self.delta * a.cos(), self.x[1] + self.delta * a.sin()]);
        }
        if !self.last_path.is_empty() {
            // the region the last step was chosen in, dashed
            let (c, r) = (&self.last_center, self.last_delta);
            for k in 0..segs / 2 {
                out.extend(nan);
                for e in 0..2 {
                    let a = std::f64::consts::TAU * (2 * k + e) as f64 / segs as f64;
                    out.extend([c[0] + r * a.cos(), c[1] + r * a.sin()]);
                }
            }
            out.extend(nan);
            // the path, subdivided so it drapes over the surface
            let mut prev = [self.last_center[0], self.last_center[1]];
            out.extend(prev);
            for q in self.last_path.chunks(self.x.len()) {
                for s in 1..=12 {
                    let t = s as f64 / 12.0;
                    out.extend([prev[0] + t * (q[0] - prev[0]), prev[1] + t * (q[1] - prev[1])]);
                }
                prev = [q[0], q[1]];
            }
            if !self.last_accepted {
                let r = 0.08 * self.delta.max(1e-3) + 0.02;
                out.extend(nan);
                out.extend([prev[0] - r, prev[1] - r, prev[0] + r, prev[1] + r]);
                out.extend(nan);
                out.extend([prev[0] - r, prev[1] + r, prev[0] + r, prev[1] - r]);
            }
        }
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::run;
    use super::*;
    use crate::problems;

    #[test]
    fn every_subproblem_solver_solves_rosenbrock() {
        let p = problems::by_id("rosenbrock", 2).unwrap();
        for (solver, iters) in [(1.0, 60), (2.0, 60), (0.0, 20_000)] {
            let mut t = TrustRegion { solver, ..Default::default() };
            let f = run(&mut t, p.as_ref(), &[-1.2, 1.0], iters);
            assert!(f < 1e-8, "solver {solver}: f = {f}");
        }
    }

    #[test]
    fn steps_never_leave_the_region_and_f_never_increases() {
        let p = problems::by_id("himmelblau", 2).unwrap();
        for solver in [0.0, 1.0, 2.0] {
            let mut t = TrustRegion { solver, delta0: 0.3, ..Default::default() };
            t.init(&[-0.3, -0.9]);
            let mut rng = Rng::new(0);
            let mut f_prev = p.eval(t.x());
            for _ in 0..80 {
                let (x0, d) = (t.x().to_vec(), t.delta);
                t.step(p.as_ref(), &mut rng, None);
                let moved = ((t.x()[0] - x0[0]).powi(2) + (t.x()[1] - x0[1]).powi(2)).sqrt();
                assert!(moved <= d * (1.0 + 1e-9), "solver {solver}: step {moved} > Δ {d}");
                let f = p.eval(t.x());
                assert!(f <= f_prev + 1e-12, "solver {solver}: f went up");
                f_prev = f;
            }
        }
    }

    #[test]
    fn exact_solver_satisfies_the_optimality_conditions_and_handles_indefinite_b() {
        // B indefinite (a saddle): the exact step must lie on the boundary with B + λI ⪰ 0
        let b = [2.0, 0.0, 0.0, -2.0];
        let g = [0.4, 0.3];
        let (p, lam) = exact(&g, &b, 0.5);
        assert!((norm(&p) - 0.5).abs() < 1e-9);
        assert!(lam >= 2.0 - 1e-9);
        // (B + λI)p = −g
        let r = [(2.0 + lam) * p[0] + g[0], (-2.0 + lam) * p[1] + g[1]];
        assert!(r[0].abs() < 1e-8 && r[1].abs() < 1e-8, "{r:?}");
        // hard case: g ⟂ lowest eigenvector
        let (p, _) = exact(&[0.4, 0.0], &b, 0.5);
        assert!((norm(&p) - 0.5).abs() < 1e-9 && p[1].abs() > 0.1, "{p:?}");
    }

    #[test]
    fn a_rejected_step_shrinks_the_region_and_is_drawn_with_a_cross() {
        // a huge initial radius on Rastrigin: the Newton model is poor far away, so early steps get rejected
        let p = problems::by_id("rastrigin", 2).unwrap();
        let mut t = TrustRegion { delta0: 4.0, solver: 2.0, ..Default::default() };
        t.init(&[3.3, -2.7]);
        let mut rng = Rng::new(0);
        let mut saw = false;
        for _ in 0..20 {
            let d = t.delta;
            t.step(p.as_ref(), &mut rng, None);
            if !t.last_accepted {
                assert!(t.delta < d);
                let sh = t.shape().unwrap();
                assert!(sh.iter().filter(|v| v.is_nan()).count() >= 6, "circle, path and two cross strokes");
                saw = true;
                break;
            }
        }
        assert!(saw, "expected a rejected step");
    }
}
