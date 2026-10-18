//! First-order optimizers as state machines stepped one iteration at a time. M0 has the three the first view
//! needs; M1 adds the rest of DESIGN.md §3.1 against the same trait.

use crate::manifest::{Deviation, OptimizerManifest, ParamSpec, Pseudocode, lines};
use crate::{Problem, Rng, StepInfo};

pub trait Optimizer: Send {
    fn manifest(&self) -> OptimizerManifest;
    /// Resets all state and starts from `x0`.
    fn init(&mut self, x0: &[f64]);
    fn step(&mut self, p: &dyn Problem, rng: &mut Rng) -> StepInfo;
    /// The current iterate
    fn x(&self) -> &[f64];
    /// Live tweak; returns false for an unknown key. State (momentum, moments) is kept.
    fn set_param(&mut self, key: &str, v: f64) -> bool;
}

pub const IDS: &[&str] = &["gd", "momentum", "adam"];

pub fn by_id(id: &str) -> Option<Box<dyn Optimizer>> {
    Some(match id {
        "gd" => Box::new(GradientDescent::default()),
        "momentum" => Box::new(Momentum::default()),
        "adam" => Box::new(Adam::default()),
        _ => return None,
    })
}

/// Every source states a convergence test; here the run continues until you stop it (or the iterate stops being
/// finite), because watching what happens after "convergence" is part of the point.
fn no_stopping_test(line: usize) -> Deviation {
    Deviation {
        line: Some(line),
        what: "No convergence test: the loop runs until you pause it, or until $x_k$ is no longer finite (shown as diverged).",
        why: "Stopping rules (tolerances on $\\lVert\\nabla f\\rVert$ or on progress) will be selectable stopping points in the step debugger; by default you watch the whole run.",
    }
}

fn lr_spec(default: f64) -> ParamSpec {
    ParamSpec {
        key: "lr",
        symbol: r"\eta",
        description: "step size (learning rate)",
        default,
        min: 1e-5,
        max: 1.0,
        log: true,
    }
}

fn norm(g: &[f64]) -> f64 {
    g.iter().map(|v| v * v).sum::<f64>().sqrt()
}

#[derive(Default)]
struct Base {
    x: Vec<f64>,
    g: Vec<f64>,
}

impl Base {
    fn init(&mut self, x0: &[f64]) {
        self.x = x0.to_vec();
        self.g = vec![0.0; x0.len()];
    }
    /// Evaluates f and ∇f at the current x (into self.g).
    fn eval(&mut self, p: &dyn Problem) -> StepInfo {
        let f = p.grad(&self.x, &mut self.g);
        StepInfo { f, grad_norm: norm(&self.g), evals: 1 }
    }
}

/// x ← x − η ∇f(x)
pub struct GradientDescent {
    b: Base,
    pub lr: f64,
}

impl Default for GradientDescent {
    fn default() -> Self {
        Self { b: Base::default(), lr: 1e-3 }
    }
}

impl Optimizer for GradientDescent {
    fn manifest(&self) -> OptimizerManifest {
        OptimizerManifest {
            id: "gd",
            name: "Gradient descent",
            family: "first-order",
            params: vec![lr_spec(1e-3)],
            tex: r"x_{k+1} = x_k - \eta\,\nabla f(x_k)",
            pseudocode: Pseudocode {
                source: "Boyd & Vandenberghe (2004), Convex Optimization, §9.3, Algorithm 9.3 (gradient descent method)",
                lines: lines(&[
                    (0, false, r"\textbf{given } \text{a starting point } x \in \operatorname{dom} f"),
                    (0, false, r"\textbf{repeat}"),
                    (1, true, r"\Delta x := -\nabla f(x)"),
                    (
                        1,
                        true,
                        r"\textit{Line search.}\ \text{Choose step size } t \text{ via exact or backtracking line search}",
                    ),
                    (1, true, r"\textit{Update.}\ x := x + t\,\Delta x"),
                    (0, false, r"\textbf{until } \text{stopping criterion is satisfied}"),
                ]),
                deviations: vec![
                    Deviation {
                        line: Some(3),
                        what: "No line search: the step size is the fixed $t = \\eta$ you set.",
                        why: "Fixed-step gradient descent is the version most people meet first and shows step-size effects (zig-zag, divergence) directly. Armijo and Wolfe line searches arrive as separate, pluggable choices.",
                    },
                    no_stopping_test(5),
                ],
            },
        }
    }
    fn init(&mut self, x0: &[f64]) {
        self.b.init(x0);
    }
    fn step(&mut self, p: &dyn Problem, _: &mut Rng) -> StepInfo {
        let info = self.b.eval(p);
        for (x, g) in self.b.x.iter_mut().zip(&self.b.g) {
            *x -= self.lr * g;
        }
        info
    }
    fn x(&self) -> &[f64] {
        &self.b.x
    }
    fn set_param(&mut self, key: &str, v: f64) -> bool {
        match key {
            "lr" => self.lr = v,
            _ => return false,
        }
        true
    }
}

/// Heavy ball: v ← βv + ∇f(x), x ← x − ηv
pub struct Momentum {
    b: Base,
    v: Vec<f64>,
    pub lr: f64,
    pub beta: f64,
}

impl Default for Momentum {
    fn default() -> Self {
        Self { b: Base::default(), v: vec![], lr: 1e-3, beta: 0.9 }
    }
}

impl Optimizer for Momentum {
    fn manifest(&self) -> OptimizerManifest {
        OptimizerManifest {
            id: "momentum",
            name: "Momentum",
            family: "first-order",
            params: vec![
                lr_spec(1e-3),
                ParamSpec {
                    key: "beta",
                    symbol: r"\beta",
                    description: "momentum: how much of the last step carries over",
                    default: 0.9,
                    min: 0.0,
                    max: 0.999,
                    log: false,
                },
            ],
            tex: r"v_{k+1} = \beta v_k + \nabla f(x_k),\quad x_{k+1} = x_k - \eta\, v_{k+1}",
            pseudocode: Pseudocode {
                source: "Polyak (1964), Some methods of speeding up the convergence of iteration methods, eq. (9) (heavy-ball method)",
                lines: lines(&[
                    (0, false, r"\textbf{given } x^0,\ \alpha > 0,\ 0 \le \beta < 1;\quad x^{-1} := x^0"),
                    (0, false, r"\textbf{for } k = 0, 1, 2, \dots"),
                    (1, true, r"x^{k+1} = x^k - \alpha\,\nabla f(x^k) + \beta\,(x^k - x^{k-1})"),
                ]),
                deviations: vec![
                    Deviation {
                        line: Some(2),
                        what: "Computed in velocity form: $v_{k+1} = \\beta v_k + \\nabla f(x_k)$, $x_{k+1} = x_k - \\eta\\, v_{k+1}$ with $v_0 = 0$ and $\\eta = \\alpha$.",
                        why: "Identical in exact arithmetic ($x_{k+1} - x_k = -\\eta v_{k+1}$ unrolls to Polyak's update) and the usual form in practice. Two visible consequences: rounding differs slightly, and changing $\\eta$ mid-run rescales the remembered velocity's effect, where Polyak's form would keep the last step's length.",
                    },
                    no_stopping_test(1),
                ],
            },
        }
    }
    fn init(&mut self, x0: &[f64]) {
        self.b.init(x0);
        self.v = vec![0.0; x0.len()];
    }
    fn step(&mut self, p: &dyn Problem, _: &mut Rng) -> StepInfo {
        let info = self.b.eval(p);
        for ((x, g), v) in self.b.x.iter_mut().zip(&self.b.g).zip(&mut self.v) {
            *v = self.beta * *v + g;
            *x -= self.lr * *v;
        }
        info
    }
    fn x(&self) -> &[f64] {
        &self.b.x
    }
    fn set_param(&mut self, key: &str, v: f64) -> bool {
        match key {
            "lr" => self.lr = v,
            "beta" => self.beta = v,
            _ => return false,
        }
        true
    }
}

/// Adam (Kingma & Ba 2015) with bias correction.
pub struct Adam {
    b: Base,
    m: Vec<f64>,
    v: Vec<f64>,
    t: i32,
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
}

impl Default for Adam {
    fn default() -> Self {
        Self { b: Base::default(), m: vec![], v: vec![], t: 0, lr: 0.05, beta1: 0.9, beta2: 0.999, eps: 1e-8 }
    }
}

impl Optimizer for Adam {
    fn manifest(&self) -> OptimizerManifest {
        OptimizerManifest {
            id: "adam",
            name: "Adam",
            family: "first-order",
            params: vec![
                lr_spec(0.05),
                ParamSpec {
                    key: "beta1",
                    symbol: r"\beta_1",
                    description: "decay of the gradient average (momentum)",
                    default: 0.9,
                    min: 0.0,
                    max: 0.999,
                    log: false,
                },
                ParamSpec {
                    key: "beta2",
                    symbol: r"\beta_2",
                    description: "decay of the squared-gradient average",
                    default: 0.999,
                    min: 0.9,
                    max: 0.99999,
                    log: false,
                },
                ParamSpec {
                    key: "eps",
                    symbol: r"\epsilon",
                    description: "keeps the denominator away from zero",
                    default: 1e-8,
                    min: 1e-12,
                    max: 1e-2,
                    log: true,
                },
            ],
            tex: r"m \leftarrow \beta_1 m + (1-\beta_1) g,\; v \leftarrow \beta_2 v + (1-\beta_2) g^2,\; x \leftarrow x - \eta\,\frac{\hat m}{\sqrt{\hat v}+\epsilon}",
            pseudocode: Pseudocode {
                source: "Kingma & Ba (2015), Adam: A Method for Stochastic Optimization, Algorithm 1",
                lines: lines(&[
                    (0, false, r"\textbf{Require: } \alpha \text{: stepsize}"),
                    (
                        0,
                        false,
                        r"\textbf{Require: } \beta_1, \beta_2 \in [0, 1) \text{: exponential decay rates for the moment estimates}",
                    ),
                    (
                        0,
                        false,
                        r"\textbf{Require: } f(\theta) \text{: stochastic objective function with parameters } \theta",
                    ),
                    (0, false, r"\textbf{Require: } \theta_0 \text{: initial parameter vector}"),
                    (0, false, r"m_0 \leftarrow 0;\quad v_0 \leftarrow 0;\quad t \leftarrow 0"),
                    (0, false, r"\textbf{while } \theta_t \text{ not converged } \textbf{do}"),
                    (1, true, r"t \leftarrow t + 1"),
                    (1, true, r"g_t \leftarrow \nabla_\theta f_t(\theta_{t-1})"),
                    (1, true, r"m_t \leftarrow \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t"),
                    (1, true, r"v_t \leftarrow \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2"),
                    (1, true, r"\hat m_t \leftarrow m_t / (1 - \beta_1^t)"),
                    (1, true, r"\hat v_t \leftarrow v_t / (1 - \beta_2^t)"),
                    (
                        1,
                        true,
                        r"\theta_t \leftarrow \theta_{t-1} - \alpha \cdot \hat m_t / (\sqrt{\hat v_t} + \epsilon)",
                    ),
                    (0, false, r"\textbf{end while}"),
                    (0, false, r"\textbf{return } \theta_t"),
                ]),
                deviations: vec![
                    Deviation {
                        line: Some(0),
                        what: "The stepsize $\\alpha$ is labelled $\\eta$ in the controls.",
                        why: "Every optimizer here uses $\\eta$ for its step size so they can be compared side by side; the update is the same.",
                    },
                    Deviation {
                        line: Some(7),
                        what: "$f_t = f$: the objective is deterministic, so $g_t$ is the exact gradient.",
                        why: "The test surfaces are exact functions. Noisy (minibatch-like) gradients will be an option on the problem, at which point $f_t$ varies with $t$ as in the paper.",
                    },
                    no_stopping_test(5),
                ],
            },
        }
    }
    fn init(&mut self, x0: &[f64]) {
        self.b.init(x0);
        self.m = vec![0.0; x0.len()];
        self.v = vec![0.0; x0.len()];
        self.t = 0;
    }
    fn step(&mut self, p: &dyn Problem, _: &mut Rng) -> StepInfo {
        let info = self.b.eval(p);
        self.t += 1;
        let c1 = 1.0 - self.beta1.powi(self.t);
        let c2 = 1.0 - self.beta2.powi(self.t);
        for (((x, g), m), v) in self.b.x.iter_mut().zip(&self.b.g).zip(&mut self.m).zip(&mut self.v) {
            *m = self.beta1 * *m + (1.0 - self.beta1) * g;
            *v = self.beta2 * *v + (1.0 - self.beta2) * g * g;
            *x -= self.lr * (*m / c1) / ((*v / c2).sqrt() + self.eps);
        }
        info
    }
    fn x(&self) -> &[f64] {
        &self.b.x
    }
    fn set_param(&mut self, key: &str, v: f64) -> bool {
        match key {
            "lr" => self.lr = v,
            "beta1" => self.beta1 = v,
            "beta2" => self.beta2 = v,
            "eps" => self.eps = v,
            _ => return false,
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problems;

    fn run(opt: &mut dyn Optimizer, p: &dyn Problem, x0: &[f64], iters: usize) -> f64 {
        let mut rng = Rng::new(0);
        opt.init(x0);
        for _ in 0..iters {
            opt.step(p, &mut rng);
        }
        p.eval(opt.x())
    }

    #[test]
    fn gd_converges_linearly_on_the_bowl() {
        // κ = 20, η = 1/L = 1/20: the error shrinks by (1 − 1/κ) per step in the slow direction
        let p = problems::by_id("bowl", 2).unwrap();
        let mut gd = GradientDescent { lr: 1.0 / 20.0, ..Default::default() };
        let f0 = p.eval(&[1.0, 1.0]);
        let f = run(&mut gd, p.as_ref(), &[1.0, 1.0], 200);
        let bound = f0 * (1.0f64 - 1.0 / 20.0).powi(2 * 200);
        assert!(f <= bound * 1.0001, "f {f} > bound {bound}");
    }

    #[test]
    fn each_optimizer_reaches_the_bowl_minimum() {
        let p = problems::by_id("bowl", 2).unwrap();
        for &id in IDS {
            let mut o = by_id(id).unwrap();
            let lr = if id == "adam" { 0.05 } else { 0.04 };
            o.set_param("lr", lr);
            let f = run(o.as_mut(), p.as_ref(), &[-1.8, 1.6], 3000);
            assert!(f < 1e-6, "{id}: f = {f}");
        }
    }

    #[test]
    fn adam_solves_rosenbrock() {
        let p = problems::by_id("rosenbrock", 2).unwrap();
        let mut a = Adam { lr: 0.01, ..Default::default() };
        run(&mut a, p.as_ref(), &[-1.2, 1.0], 20_000);
        let x = a.x();
        assert!((x[0] - 1.0).abs() < 1e-2 && (x[1] - 1.0).abs() < 2e-2, "{x:?}");
    }

    #[test]
    fn set_param_rejects_unknown_keys_and_manifests_list_params() {
        for &id in IDS {
            let mut o = by_id(id).unwrap();
            assert!(!o.set_param("nope", 1.0));
            for spec in o.manifest().params {
                assert!(o.set_param(spec.key, spec.default), "{id}.{}", spec.key);
                assert!(spec.min <= spec.default && spec.default <= spec.max);
            }
        }
    }

    #[test]
    fn every_optimizer_has_sourced_pseudocode() {
        for &id in IDS {
            let pc = by_id(id).unwrap().manifest().pseudocode;
            assert!(pc.source.contains('(') && pc.source.len() > 20, "{id}: cite the source");
            assert!(pc.lines.iter().any(|l| l.step), "{id}: mark the lines a step executes");
            for d in &pc.deviations {
                if let Some(l) = d.line {
                    assert!(l < pc.lines.len(), "{id}: deviation points past the last line");
                }
                assert!(!d.what.is_empty() && !d.why.is_empty());
            }
        }
    }

    #[test]
    fn runs_are_deterministic() {
        let p = problems::by_id("ackley", 2).unwrap();
        let a = run(by_id("adam").unwrap().as_mut(), p.as_ref(), &[3.1, -2.4], 500);
        let b = run(by_id("adam").unwrap().as_mut(), p.as_ref(), &[3.1, -2.4], 500);
        assert_eq!(a.to_bits(), b.to_bits());
    }
}
