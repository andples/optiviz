//! The Nelder–Mead simplex method as stated by Lagarias, Reeds, Wright & Wright (1998), §2.

use super::{Optimizer, no_stopping_test};
use crate::manifest::{Deviation, OptimizerManifest, ParamSpec, Pseudocode, lines};
use crate::trace::Trace;
use crate::{Problem, Rng, StepInfo, tr};

pub struct NelderMead {
    /// vertices (n+1) and their f values, kept ordered best first after each step
    v: Vec<Vec<f64>>,
    fv: Vec<f64>,
    fresh: bool,
    x0: Vec<f64>,
    pub size: f64,
    pub rho: f64,
    pub chi: f64,
    pub gamma: f64,
    pub sigma: f64,
}

impl Default for NelderMead {
    fn default() -> Self {
        Self { v: vec![], fv: vec![], fresh: true, x0: vec![], size: 0.25, rho: 1.0, chi: 2.0, gamma: 0.5, sigma: 0.5 }
    }
}

fn comb(a: &[f64], wa: f64, b: &[f64], wb: f64) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| wa * x + wb * y).collect()
}

impl NelderMead {
    fn order(&mut self) {
        let mut idx: Vec<usize> = (0..self.v.len()).collect();
        idx.sort_by(|&a, &b| self.fv[a].total_cmp(&self.fv[b]));
        self.v = idx.iter().map(|&i| self.v[i].clone()).collect();
        self.fv = idx.iter().map(|&i| self.fv[i]).collect();
    }
}

impl Optimizer for NelderMead {
    fn manifest(&self) -> OptimizerManifest {
        OptimizerManifest {
            id: "nelder-mead",
            name: "Nelder–Mead",
            family: "derivative-free",
            params: vec![
                ParamSpec::real("size", r"\delta", "initial simplex edge, as a length in x", 0.25, 0.001, 2.0, true),
                ParamSpec::real("rho", r"\rho", "reflection coefficient", 1.0, 0.1, 2.0, false),
                ParamSpec::real("chi", r"\chi", "expansion coefficient", 2.0, 1.1, 4.0, false),
                ParamSpec::real("gamma", r"\gamma", "contraction coefficient", 0.5, 0.05, 0.95, false),
                ParamSpec::real("sigma", r"\sigma", "shrink coefficient", 0.5, 0.05, 0.95, false),
            ],
            tex: r"x_r = \bar x + \rho(\bar x - x_{n+1}),\ \text{expand, contract or shrink by } \chi, \gamma, \sigma",
            pseudocode: Pseudocode {
                source: "Lagarias, Reeds, Wright & Wright (1998), Convergence properties of the Nelder–Mead simplex method in low dimensions, SIAM J. Optim. 9(1), §2 (one iteration)",
                lines: lines(&[
                    (
                        0,
                        false,
                        r"\text{Given a simplex of } n+1 \text{ vertices; coefficients } \rho = 1,\ \chi = 2,\ \gamma = \tfrac12,\ \sigma = \tfrac12",
                    ),
                    (0, false, r"\textbf{repeat}"),
                    (1, true, r"\textbf{1. Order.}\ f(x_1) \le f(x_2) \le \dots \le f(x_{n+1})"),
                    (
                        1,
                        true,
                        r"\textbf{2. Reflect.}\ x_r = \bar x + \rho(\bar x - x_{n+1}),\ \bar x = \textstyle\sum_{i=1}^n x_i / n.\ \textbf{If } f_1 \le f_r < f_n \text{, accept } x_r",
                    ),
                    (
                        1,
                        true,
                        r"\textbf{3. Expand.}\ \textbf{If } f_r < f_1:\ x_e = \bar x + \chi(x_r - \bar x);\ \text{accept } x_e \textbf{ if } f_e < f_r \textbf{ else } x_r",
                    ),
                    (
                        1,
                        true,
                        r"\textbf{4a. Contract outside.}\ \textbf{If } f_n \le f_r < f_{n+1}:\ x_c = \bar x + \gamma(x_r - \bar x);\ \textbf{if } f_c \le f_r \text{ accept } x_c \textbf{ else } \text{shrink}",
                    ),
                    (
                        1,
                        true,
                        r"\textbf{4b. Contract inside.}\ \textbf{If } f_r \ge f_{n+1}:\ x_{cc} = \bar x - \gamma(\bar x - x_{n+1});\ \textbf{if } f_{cc} < f_{n+1} \text{ accept } x_{cc} \textbf{ else } \text{shrink}",
                    ),
                    (1, true, r"\textbf{5. Shrink.}\ v_i = x_1 + \sigma(x_i - x_1),\ i = 2, \dots, n+1"),
                    (0, false, r"\textbf{until } \text{the simplex is small enough}"),
                ]),
                deviations: vec![
                    Deviation {
                        line: Some(0),
                        what: "The initial simplex is $x_0$ and $x_0 + \\delta e_i$ for each coordinate direction ($\\delta$ is a parameter).",
                        why: "The paper analyses one iteration and leaves the initial simplex to the user; axis-aligned is the common choice.",
                    },
                    Deviation {
                        line: Some(2),
                        what: "Ties in $f$ keep their previous order (a stable sort), rather than the paper's exact tie-breaking rules for new vertices.",
                        why: "On continuous test functions exact ties almost never happen, and a stable sort gives the same order in nearly all cases.",
                    },
                    no_stopping_test(8),
                ],
            },
        }
    }
    fn init(&mut self, x0: &[f64]) {
        self.x0 = x0.to_vec();
        self.fresh = true;
        let n = x0.len();
        self.v = std::iter::once(x0.to_vec())
            .chain((0..n).map(|i| {
                let mut v = x0.to_vec();
                v[i] += self.size;
                v
            }))
            .collect();
        self.fv = vec![f64::NAN; n + 1];
    }
    fn step(&mut self, p: &dyn Problem, _: &mut Rng, mut trc: Option<&mut Trace>) -> StepInfo {
        let n = self.v.len() - 1;
        let mut evals = 0;
        if self.fresh {
            self.fv = self.v.iter().map(|v| p.eval(v)).collect();
            evals += n as u32 + 1;
            self.fresh = false;
        }
        self.order();
        let f_start = self.fv[0];
        tr!(trc, 2, "f_1" => self.fv[0], "f_n" => self.fv[n - 1], "f_{n+1}" => self.fv[n]);
        let mut xbar = vec![0.0; self.v[0].len()];
        for v in &self.v[..n] {
            for (b, x) in xbar.iter_mut().zip(v) {
                *b += x / n as f64;
            }
        }
        let worst = self.v[n].clone();
        let xr = comb(&xbar, 1.0 + self.rho, &worst, -self.rho);
        let fr = p.eval(&xr);
        evals += 1;
        tr!(trc, 3, r"\bar x" => xbar, "x_r" => xr, "f_r" => fr);
        let mut shrink = false;
        if self.fv[0] <= fr && fr < self.fv[n - 1] {
            self.v[n] = xr;
            self.fv[n] = fr;
        } else if fr < self.fv[0] {
            let xe = comb(&xbar, 1.0 + self.rho * self.chi, &worst, -self.rho * self.chi);
            let fe = p.eval(&xe);
            evals += 1;
            tr!(trc, 4, "x_e" => xe, "f_e" => fe);
            if fe < fr {
                self.v[n] = xe;
                self.fv[n] = fe;
            } else {
                self.v[n] = xr;
                self.fv[n] = fr;
            }
        } else if fr < self.fv[n] {
            let xc = comb(&xbar, 1.0 + self.rho * self.gamma, &worst, -self.rho * self.gamma);
            let fc = p.eval(&xc);
            evals += 1;
            tr!(trc, 5, "x_c" => xc, "f_c" => fc);
            if fc <= fr {
                self.v[n] = xc;
                self.fv[n] = fc;
            } else {
                shrink = true;
            }
        } else {
            let xcc = comb(&xbar, 1.0 - self.gamma, &worst, self.gamma);
            let fcc = p.eval(&xcc);
            evals += 1;
            tr!(trc, 6, "x_{cc}" => xcc, "f_{cc}" => fcc);
            if fcc < self.fv[n] {
                self.v[n] = xcc;
                self.fv[n] = fcc;
            } else {
                shrink = true;
            }
        }
        if shrink {
            let best = self.v[0].clone();
            for i in 1..=n {
                self.v[i] = comb(&best, 1.0 - self.sigma, &self.v[i], self.sigma);
                self.fv[i] = p.eval(&self.v[i]);
                evals += 1;
            }
            tr!(trc, 7, "v_i" => self.v[1..].concat());
        }
        self.order();
        StepInfo { f: f_start, grad_norm: f64::NAN, evals }
    }
    fn x(&self) -> &[f64] {
        // best vertex (the simplex is kept ordered after every step)
        &self.v[0]
    }
    fn set_param(&mut self, key: &str, v: f64) -> bool {
        match key {
            "size" => self.size = v,
            "rho" => self.rho = v,
            "chi" => self.chi = v,
            "gamma" => self.gamma = v,
            "sigma" => self.sigma = v,
            _ => return false,
        }
        true
    }
    fn shape(&self) -> Option<Vec<f64>> {
        Some(self.v.concat())
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::run;
    use super::*;
    use crate::problems;

    #[test]
    fn nelder_mead_finds_the_himmelblau_minimum_without_gradients() {
        let p = problems::by_id("himmelblau", 2).unwrap();
        let f = run(&mut NelderMead::default(), p.as_ref(), &[-0.3, -0.9], 200);
        assert!(f < 1e-10, "{f}");
    }
}
