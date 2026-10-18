//! Limited-memory BFGS (Nocedal & Wright 2006, Algorithm 7.5 with the two-loop recursion of Algorithm 7.4).

use super::linesearch::strong_wolfe;
use super::{Optimizer, dot, no_stopping_test, norm};
use crate::manifest::{Deviation, OptimizerManifest, ParamSpec, Pseudocode, lines};
use crate::trace::Trace;
use crate::{Problem, Rng, StepInfo, tr};
use std::collections::VecDeque;

pub struct Lbfgs {
    x: Vec<f64>,
    g: Vec<f64>,
    f: f64,
    fresh: bool,
    /// (s, y, ρ) pairs, oldest first
    mem: VecDeque<(Vec<f64>, Vec<f64>, f64)>,
    pub m: f64,
    pub c1: f64,
    pub c2: f64,
}

impl Default for Lbfgs {
    fn default() -> Self {
        Self { x: vec![], g: vec![], f: 0.0, fresh: true, mem: VecDeque::new(), m: 5.0, c1: 1e-4, c2: 0.9 }
    }
}

impl Optimizer for Lbfgs {
    fn manifest(&self) -> OptimizerManifest {
        OptimizerManifest {
            id: "lbfgs",
            name: "L-BFGS",
            family: "quasi-Newton",
            params: vec![
                ParamSpec::real(
                    "m",
                    "m",
                    "memory: how many recent (s, y) pairs shape the curvature estimate",
                    5.0,
                    1.0,
                    20.0,
                    false,
                ),
                ParamSpec::real("c1", "c_1", "Wolfe sufficient-decrease constant", 1e-4, 1e-6, 0.4, true),
                ParamSpec::real("c2", "c_2", "Wolfe curvature constant", 0.9, 0.5, 0.99, false),
            ],
            tex: r"x_{k+1} = x_k - \alpha_k H_k \nabla f_k,\quad H_k \text{ from the last } m \text{ pairs } (s_i, y_i)",
            pseudocode: Pseudocode {
                source: "Nocedal & Wright (2006), Numerical Optimization, 2nd ed., Algorithm 7.5 (L-BFGS) with Algorithm 7.4 (two-loop recursion) and eq. (7.20)",
                lines: lines(&[
                    (0, false, r"\text{Choose starting point } x_0, \text{ integer } m > 0;\ k \leftarrow 0"),
                    (0, false, r"\textbf{repeat}"),
                    (
                        1,
                        true,
                        r"\text{Choose } H_k^0 = \gamma_k I,\ \gamma_k = \frac{s_{k-1}^\top y_{k-1}}{y_{k-1}^\top y_{k-1}} \text{ (7.20)}",
                    ),
                    (1, true, r"\text{Compute } p_k \leftarrow -H_k \nabla f_k \text{ from Algorithm 7.4:}"),
                    (
                        2,
                        true,
                        r"q \leftarrow \nabla f_k;\ \textbf{for } i = k-1, \dots, k-m:\ \alpha_i \leftarrow \rho_i s_i^\top q;\ q \leftarrow q - \alpha_i y_i",
                    ),
                    (
                        2,
                        true,
                        r"r \leftarrow H_k^0 q;\ \textbf{for } i = k-m, \dots, k-1:\ \beta \leftarrow \rho_i y_i^\top r;\ r \leftarrow r + s_i(\alpha_i - \beta)",
                    ),
                    (
                        1,
                        true,
                        r"\text{Compute } x_{k+1} \leftarrow x_k + \alpha_k p_k \text{, where } \alpha_k \text{ is chosen to satisfy the Wolfe conditions}",
                    ),
                    (
                        1,
                        true,
                        r"\textbf{if } k > m \text{: discard the vector pair } \{s_{k-m}, y_{k-m}\} \text{ from storage}",
                    ),
                    (
                        1,
                        true,
                        r"\text{Compute and save } s_k \leftarrow x_{k+1} - x_k,\ y_k = \nabla f_{k+1} - \nabla f_k",
                    ),
                    (1, true, r"k \leftarrow k + 1"),
                    (0, false, r"\textbf{until } \text{convergence}"),
                ]),
                deviations: vec![
                    Deviation {
                        line: Some(2),
                        what: "At $k = 0$, with no pairs yet, $H_0^0 = I / \\lVert \\nabla f_0 \\rVert$, so the first trial step has length 1.",
                        why: "(7.20) needs a previous pair; the source leaves the first choice open. Scaling by the gradient norm keeps the first step a sensible size on every landscape.",
                    },
                    Deviation {
                        line: Some(6),
                        what: "The step satisfies the strong Wolfe conditions (Algorithm 3.5, $\\alpha$ starting at 1); after 25 trials the best point found is taken.",
                        why: "Strong Wolfe is the usual choice for L-BFGS and guarantees $s_k^\\top y_k > 0$; the trial cap guards against non-finite values far outside the landscape.",
                    },
                    Deviation {
                        line: Some(8),
                        what: "A pair is only saved if $s_k^\\top y_k > 10^{-10}\\, \\lVert s_k \\rVert \\lVert y_k \\rVert$ and $\\rho_k = 1/s_k^\\top y_k$ is finite.",
                        why: "Keeps $H_k$ positive definite when the line search hit its trial cap; with a successful Wolfe step this never triggers.",
                    },
                    no_stopping_test(10),
                ],
            },
        }
    }
    fn init(&mut self, x0: &[f64]) {
        self.x = x0.to_vec();
        self.g = vec![0.0; x0.len()];
        self.fresh = true;
        self.mem.clear();
    }
    fn step(&mut self, p: &dyn Problem, _: &mut Rng, mut trc: Option<&mut Trace>) -> StepInfo {
        let mut evals = 0;
        if self.fresh {
            self.f = p.grad(&self.x, &mut self.g);
            evals += 1;
            self.fresh = false;
        }
        let info_f = self.f;
        let info_gn = norm(&self.g);
        if info_gn == 0.0 {
            // exactly at a stationary point: nothing to do (the direction would be 0)
            tr!(trc, 3, r"\nabla f_k" => self.g);
            return StepInfo { f: info_f, grad_norm: 0.0, evals };
        }
        let gamma = match self.mem.back() {
            Some((s, y, _)) if dot(y, y) > 0.0 => dot(s, y) / dot(y, y),
            _ => 1.0 / info_gn,
        };
        tr!(trc, 2, r"\gamma_k" => gamma);
        // two-loop recursion
        let mut q = self.g.clone();
        let mut alphas = vec![0.0; self.mem.len()];
        for (i, (s, y, rho)) in self.mem.iter().enumerate().rev() {
            alphas[i] = rho * dot(s, &q);
            for (qj, yj) in q.iter_mut().zip(y) {
                *qj -= alphas[i] * yj;
            }
        }
        tr!(trc, 3, r"\nabla f_k" => self.g);
        tr!(trc, 4, "q" => q, r"\alpha_i" => alphas);
        let mut r: Vec<f64> = q.iter().map(|v| gamma * v).collect();
        for (i, (s, y, rho)) in self.mem.iter().enumerate() {
            let beta = rho * dot(y, &r);
            for (rj, sj) in r.iter_mut().zip(s) {
                *rj += sj * (alphas[i] - beta);
            }
        }
        let d: Vec<f64> = r.iter().map(|v| -v).collect();
        tr!(trc, 5, "r" => r, r"p_k" => d);
        let ws = strong_wolfe(p, &self.x, &d, self.f, &self.g, 1.0, self.c1, self.c2, trc.as_deref_mut(), 6);
        evals += ws.evals;
        let s: Vec<f64> = d.iter().map(|v| ws.alpha * v).collect();
        let y: Vec<f64> = ws.g.iter().zip(&self.g).map(|(a, b)| a - b).collect();
        for (x, si) in self.x.iter_mut().zip(&s) {
            *x += si;
        }
        tr!(trc, 6, r"\alpha_k" => ws.alpha, r"x_{k+1}" => self.x);
        let m = self.m.round().max(1.0) as usize;
        while self.mem.len() >= m {
            self.mem.pop_front();
        }
        tr!(trc, 7, r"\text{pairs kept}" => self.mem.len());
        let sy = dot(&s, &y);
        if sy > 1e-10 * norm(&s) * norm(&y) && (1.0 / sy).is_finite() {
            self.mem.push_back((s.clone(), y.clone(), 1.0 / sy));
        }
        tr!(trc, 8, "s_k" => s, "y_k" => y);
        self.f = ws.f;
        self.g = ws.g;
        tr!(trc, 9, r"f(x_{k+1})" => self.f);
        StepInfo { f: info_f, grad_norm: info_gn, evals }
    }
    fn x(&self) -> &[f64] {
        &self.x
    }
    fn set_param(&mut self, key: &str, v: f64) -> bool {
        match key {
            "m" => self.m = v,
            "c1" => self.c1 = v,
            "c2" => self.c2 = v,
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
    fn lbfgs_solves_rosenbrock_fast_and_10d_rosenbrock() {
        let p = problems::by_id("rosenbrock", 2).unwrap();
        let f = run(&mut Lbfgs::default(), p.as_ref(), &[-1.2, 1.0], 60);
        assert!(f < 1e-10, "{f}");
        let p = problems::by_id("rosenbrock", 10).unwrap();
        let f = run(&mut Lbfgs::default(), p.as_ref(), &p.manifest().start, 300);
        assert!(f < 1e-8, "{f}");
    }
}
