//! Newton's method with Hessian modification and backtracking (Nocedal & Wright 2006, Algorithms 3.2, 3.3,
//! 3.1).

use super::linesearch::backtracking;
use super::{Optimizer, dot, no_stopping_test, norm};
use crate::manifest::{Deviation, OptimizerManifest, ParamSpec, Pseudocode, lines};
use crate::trace::Trace;
use crate::{Problem, Rng, StepInfo, tr};

pub struct Newton {
    x: Vec<f64>,
    pub beta: f64,
    pub c: f64,
    pub rho: f64,
}

impl Default for Newton {
    fn default() -> Self {
        Self { x: vec![], beta: 1e-3, c: 1e-4, rho: 0.5 }
    }
}

/// Cholesky factor of a symmetric n×n matrix (row-major), or None if it isn't positive definite.
pub(crate) fn cholesky(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if s <= 0.0 || !s.is_finite() {
                    return None;
                }
                l[i * n + i] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    Some(l)
}

/// Solves L Lᵀ x = b.
pub(crate) fn chol_solve(l: &[f64], n: usize, b: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; n];
    for i in 0..n {
        let s: f64 = (0..i).map(|k| l[i * n + k] * y[k]).sum();
        y[i] = (b[i] - s) / l[i * n + i];
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let s: f64 = (i + 1..n).map(|k| l[k * n + i] * x[k]).sum();
        x[i] = (y[i] - s) / l[i * n + i];
    }
    x
}

impl Optimizer for Newton {
    fn manifest(&self) -> OptimizerManifest {
        OptimizerManifest {
            id: "newton",
            name: "Newton's method",
            family: "second-order",
            params: vec![
                ParamSpec::real(
                    "beta",
                    r"\beta",
                    "smallest shift added when the Hessian isn't positive definite",
                    1e-3,
                    1e-8,
                    1.0,
                    true,
                ),
                ParamSpec::real(
                    "c",
                    "c",
                    "sufficient-decrease constant of the backtracking search",
                    1e-4,
                    1e-6,
                    0.5,
                    true,
                ),
                ParamSpec::real("rho", r"\rho", "factor the step shrinks by when backtracking", 0.5, 0.1, 0.9, false),
            ],
            tex: r"x_{k+1} = x_k - \alpha_k \left(\nabla^2 f(x_k) + \tau_k I\right)^{-1} \nabla f(x_k)",
            pseudocode: Pseudocode {
                source: "Nocedal & Wright (2006), Numerical Optimization, 2nd ed., Algorithm 3.2 (line search Newton with modification), with Algorithm 3.3 (Cholesky with added multiple of the identity) and Algorithm 3.1 (backtracking)",
                lines: lines(&[
                    (0, false, r"\text{Given initial point } x_0"),
                    (0, false, r"\textbf{for } k = 0, 1, 2, \dots"),
                    (
                        1,
                        true,
                        r"\text{Factorize } B_k = \nabla^2 f(x_k) + E_k \text{, where } E_k = 0 \text{ if } \nabla^2 f(x_k) \text{ is sufficiently positive definite; otherwise } E_k \text{ ensures } B_k \text{ is}",
                    ),
                    (
                        2,
                        true,
                        r"\text{(Alg. 3.3) } A = \nabla^2 f(x_k);\ \textbf{if } \min_i a_{ii} > 0 \text{ set } \tau_0 \leftarrow 0 \textbf{ else } \tau_0 = -\min(a_{ii}) + \beta",
                    ),
                    (
                        2,
                        true,
                        r"\text{Attempt Cholesky on } A + \tau_j I;\ \textbf{if} \text{ successful, stop; } \textbf{else } \tau_{j+1} \leftarrow \max(2\tau_j, \beta)",
                    ),
                    (1, true, r"\text{Solve } B_k p_k = -\nabla f(x_k)"),
                    (
                        1,
                        true,
                        r"\text{Set } x_{k+1} \leftarrow x_k + \alpha_k p_k \text{, where } \alpha_k \text{ satisfies the Armijo backtracking conditions}",
                    ),
                    (
                        2,
                        true,
                        r"\text{(Alg. 3.1) } \alpha \leftarrow \bar\alpha = 1;\ \textbf{repeat until } f(x_k + \alpha p_k) \le f(x_k) + c\,\alpha\, \nabla f_k^\top p_k:\ \alpha \leftarrow \rho\alpha",
                    ),
                    (0, false, r"\textbf{end}"),
                ]),
                deviations: vec![
                    Deviation {
                        line: Some(2),
                        what: "$\\nabla^2 f$ comes from central differences of the analytic gradient (step $10^{-5}\\max(1, |x_i|)$), symmetrized.",
                        why: "Not every test function has a hand-written Hessian; the differences are accurate to about $10^{-8}$ relative, far below what changes a step.",
                    },
                    Deviation {
                        line: Some(4),
                        what: "At most 60 Cholesky attempts; if all fail, $p_k = -\\nabla f(x_k)$.",
                        why: "A guard against non-finite Hessians (far outside the landscape); the source assumes the loop terminates.",
                    },
                    Deviation {
                        line: Some(7),
                        what: "At most 60 backtracking halvings.",
                        why: "A guard for non-finite values; with a descent direction the source's loop always terminates.",
                    },
                    no_stopping_test(1),
                ],
            },
        }
    }
    fn init(&mut self, x0: &[f64]) {
        self.x = x0.to_vec();
    }
    fn step(&mut self, p: &dyn Problem, _: &mut Rng, mut trc: Option<&mut Trace>) -> StepInfo {
        let n = self.x.len();
        let mut g = vec![0.0; n];
        let f = p.grad(&self.x, &mut g);
        let mut a = vec![0.0; n * n];
        p.hess(&self.x, &mut a);
        tr!(trc, 2, r"\nabla f(x_k)" => g, r"\nabla^2 f(x_k)" => a);
        let min_diag = (0..n).map(|i| a[i * n + i]).fold(f64::INFINITY, f64::min);
        let mut tau = if min_diag > 0.0 { 0.0 } else { -min_diag + self.beta };
        tr!(trc, 3, r"\min_i a_{ii}" => min_diag, r"\tau_0" => tau);
        let mut factor = None;
        for _ in 0..60 {
            let mut b = a.clone();
            for i in 0..n {
                b[i * n + i] += tau;
            }
            let l = cholesky(&b, n);
            tr!(trc, 4, r"\tau_j" => tau, r"\text{Cholesky ok}" => if l.is_some() { 1.0 } else { 0.0 });
            if let Some(l) = l {
                factor = Some(l);
                break;
            }
            tau = (2.0 * tau).max(self.beta);
        }
        let neg_g: Vec<f64> = g.iter().map(|v| -v).collect();
        let d = match &factor {
            Some(l) => chol_solve(l, n, &neg_g),
            None => neg_g,
        };
        tr!(trc, 5, r"p_k" => d);
        let slope = dot(&g, &d);
        let (alpha, evals) = backtracking(p, &self.x, &d, f, slope, 1.0, self.c, self.rho, trc.as_deref_mut(), 7);
        for (x, di) in self.x.iter_mut().zip(&d) {
            *x += alpha * di;
        }
        tr!(trc, 6, r"\alpha_k" => alpha, r"x_{k+1}" => self.x);
        StepInfo { f, grad_norm: norm(&g), evals: 1 + evals }
    }
    fn x(&self) -> &[f64] {
        &self.x
    }
    fn set_param(&mut self, key: &str, v: f64) -> bool {
        match key {
            "beta" => self.beta = v,
            "c" => self.c = v,
            "rho" => self.rho = v,
            _ => return false,
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::run;
    use super::*;
    use crate::problems;

    #[test]
    fn newton_solves_the_bowl_in_one_step_and_rosenbrock_quickly() {
        let p = problems::by_id("bowl", 2).unwrap();
        let f = run(&mut Newton::default(), p.as_ref(), &[-1.8, 1.6], 1);
        assert!(f < 1e-12, "{f}");
        let p = problems::by_id("rosenbrock", 2).unwrap();
        let f = run(&mut Newton::default(), p.as_ref(), &[-1.2, 1.0], 40);
        assert!(f < 1e-12, "{f}");
    }

    #[test]
    fn newton_escapes_a_saddle_direction_instead_of_climbing() {
        // at the saddle's x-axis the Hessian is indefinite: the modified step must still descend
        let p = problems::by_id("saddle", 2).unwrap();
        let mut o = Newton::default();
        o.init(&[0.5, 0.1]);
        let f0 = p.eval(&[0.5, 0.1]);
        o.step(p.as_ref(), &mut Rng::new(0), None);
        assert!(p.eval(o.x()) < f0);
    }
}
