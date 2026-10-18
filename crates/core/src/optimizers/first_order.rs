//! First-order methods. Line indices in `tr!` refer to each manifest's pseudocode.

use super::linesearch::{backtracking, strong_wolfe};
use super::{Optimizer, dot, eta_named, exact_gradient, lr_spec, no_stopping_test, norm};
use crate::manifest::{Deviation, OptimizerManifest, ParamSpec, Pseudocode, lines};
use crate::trace::Trace;
use crate::{Problem, Rng, StepInfo, tr};

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
    /// f and ∇f at the current x (into self.g)
    fn eval(&mut self, p: &dyn Problem) -> StepInfo {
        let f = p.grad(&self.x, &mut self.g);
        StepInfo { f, grad_norm: norm(&self.g), evals: 1 }
    }
}

macro_rules! params {
    ($self:ident, $key:ident, $v:ident; $($name:literal => $field:ident),*) => {
        match $key {
            $($name => $self.$field = $v,)*
            _ => return false,
        }
    };
}

/* ---------------- gradient descent ---------------- */

pub struct GradientDescent {
    b: Base,
    pub lr: f64,
    /// 0 fixed step, 1 backtracking, 2 strong Wolfe
    pub ls: f64,
    pub alpha: f64,
    pub beta: f64,
}

impl Default for GradientDescent {
    fn default() -> Self {
        Self { b: Base::default(), lr: 1e-3, ls: 0.0, alpha: 0.3, beta: 0.5 }
    }
}

impl Optimizer for GradientDescent {
    fn manifest(&self) -> OptimizerManifest {
        OptimizerManifest {
            id: "gd",
            name: "Gradient descent",
            family: "first-order",
            params: vec![
                lr_spec(1e-3),
                ParamSpec::choice(
                    "ls",
                    r"\text{line search}",
                    "how the step size t is chosen each iteration",
                    0,
                    &["fixed t = η", "backtracking", "strong Wolfe"],
                ),
                ParamSpec::real(
                    "alpha",
                    r"\alpha",
                    "backtracking: fraction of the predicted decrease to accept",
                    0.3,
                    0.01,
                    0.5,
                    false,
                ),
                ParamSpec::real("beta", r"\beta", "backtracking: factor the step shrinks by", 0.5, 0.1, 0.9, false),
            ],
            tex: r"x_{k+1} = x_k - t\,\nabla f(x_k)",
            pseudocode: Pseudocode {
                source: "Boyd & Vandenberghe (2004), Convex Optimization, §9.3, Algorithm 9.3 (gradient descent method); backtracking from Algorithm 9.2",
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
                        what: "Three choices of step size: a fixed $t = \\eta$ (the default), backtracking (Algorithm 9.2: start at $t = 1$, multiply by $\\beta$ until $f(x + t\\Delta x) \\le f(x) + \\alpha t \\nabla f(x)^\\top \\Delta x$), or a strong-Wolfe search (Nocedal & Wright Algorithm 3.5) instead of an exact one.",
                        why: "Fixed-step descent is the version most people meet first and shows step-size effects (zig-zag, divergence) directly. An exact line search has no closed form on these surfaces; strong Wolfe is the standard practical substitute.",
                    },
                    no_stopping_test(5),
                ],
            },
        }
    }
    fn init(&mut self, x0: &[f64]) {
        self.b.init(x0);
    }
    fn step(&mut self, p: &dyn Problem, _: &mut Rng, mut trc: Option<&mut Trace>) -> StepInfo {
        let mut info = self.b.eval(p);
        let d: Vec<f64> = self.b.g.iter().map(|g| -g).collect();
        tr!(trc, 2, r"\nabla f(x)" => self.b.g, r"\Delta x" => d);
        let t = match self.ls as u32 {
            1 => {
                let slope = dot(&self.b.g, &d);
                let (t, e) =
                    backtracking(p, &self.b.x, &d, info.f, slope, 1.0, self.alpha, self.beta, trc.as_deref_mut(), 3);
                info.evals += e;
                t
            }
            2 => {
                let s = strong_wolfe(p, &self.b.x, &d, info.f, &self.b.g, 1.0, 1e-4, 0.9, trc.as_deref_mut(), 3);
                info.evals += s.evals;
                s.alpha
            }
            _ => {
                tr!(trc, 3, "t" => self.lr);
                self.lr
            }
        };
        for (x, d) in self.b.x.iter_mut().zip(&d) {
            *x += t * d;
        }
        tr!(trc, 4, "t" => t, "x" => self.b.x);
        info
    }
    fn x(&self) -> &[f64] {
        &self.b.x
    }
    fn set_param(&mut self, key: &str, v: f64) -> bool {
        params!(self, key, v; "lr" => lr, "ls" => ls, "alpha" => alpha, "beta" => beta);
        true
    }
}

/* ---------------- heavy ball ---------------- */

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
                ParamSpec::real(
                    "beta",
                    r"\beta",
                    "momentum: how much of the last step carries over",
                    0.9,
                    0.0,
                    0.999,
                    false,
                ),
            ],
            tex: r"x_{k+1} = x_k - \eta\,\nabla f(x_k) + \beta\,(x_k - x_{k-1})",
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
    fn step(&mut self, p: &dyn Problem, _: &mut Rng, mut trc: Option<&mut Trace>) -> StepInfo {
        let info = self.b.eval(p);
        for ((x, g), v) in self.b.x.iter_mut().zip(&self.b.g).zip(&mut self.v) {
            *v = self.beta * *v + g;
            *x -= self.lr * *v;
        }
        tr!(trc, 2, r"\nabla f(x^k)" => self.b.g, "v" => self.v, r"x^{k+1}" => self.b.x);
        info
    }
    fn x(&self) -> &[f64] {
        &self.b.x
    }
    fn set_param(&mut self, key: &str, v: f64) -> bool {
        params!(self, key, v; "lr" => lr, "beta" => beta);
        true
    }
}

/* ---------------- Nesterov ---------------- */

pub struct Nesterov {
    b: Base,
    v: Vec<f64>,
    look: Vec<f64>,
    pub lr: f64,
    pub mu: f64,
}

impl Default for Nesterov {
    fn default() -> Self {
        Self { b: Base::default(), v: vec![], look: vec![], lr: 1e-3, mu: 0.9 }
    }
}

impl Optimizer for Nesterov {
    fn manifest(&self) -> OptimizerManifest {
        OptimizerManifest {
            id: "nesterov",
            name: "Nesterov momentum",
            family: "first-order",
            params: vec![lr_spec(1e-3), ParamSpec::real("mu", r"\mu", "momentum coefficient", 0.9, 0.0, 0.999, false)],
            tex: r"v_{t+1} = \mu v_t - \eta\,\nabla f(\theta_t + \mu v_t),\quad \theta_{t+1} = \theta_t + v_{t+1}",
            pseudocode: Pseudocode {
                source: "Sutskever, Martens, Dahl & Hinton (2013), On the importance of initialization and momentum in deep learning, eqs. (3)–(4)",
                lines: lines(&[
                    (0, false, r"\textbf{given } \theta_0,\ \varepsilon > 0,\ \mu \in [0, 1);\quad v_0 = 0"),
                    (0, false, r"\textbf{for } t = 0, 1, 2, \dots"),
                    (1, true, r"v_{t+1} = \mu v_t - \varepsilon\, \nabla f(\theta_t + \mu v_t)"),
                    (1, true, r"\theta_{t+1} = \theta_t + v_{t+1}"),
                ]),
                deviations: vec![
                    eta_named(2, "The learning rate $\\varepsilon$ is labelled $\\eta$ in the controls."),
                    no_stopping_test(1),
                ],
            },
        }
    }
    fn init(&mut self, x0: &[f64]) {
        self.b.init(x0);
        self.v = vec![0.0; x0.len()];
        self.look = vec![0.0; x0.len()];
    }
    fn step(&mut self, p: &dyn Problem, _: &mut Rng, mut trc: Option<&mut Trace>) -> StepInfo {
        // StepInfo reports f and ∇f at θ_t itself (for the readout); the update uses the look-ahead point
        let mut info = self.b.eval(p);
        for i in 0..self.v.len() {
            self.look[i] = self.b.x[i] + self.mu * self.v[i];
        }
        let mut gl = vec![0.0; self.v.len()];
        p.grad(&self.look, &mut gl);
        info.evals += 1;
        for i in 0..self.v.len() {
            self.v[i] = self.mu * self.v[i] - self.lr * gl[i];
        }
        tr!(trc, 2, r"\theta_t + \mu v_t" => self.look, r"\nabla f(\theta_t + \mu v_t)" => gl, r"v_{t+1}" => self.v);
        for (x, v) in self.b.x.iter_mut().zip(&self.v) {
            *x += v;
        }
        tr!(trc, 3, r"\theta_{t+1}" => self.b.x);
        info
    }
    fn x(&self) -> &[f64] {
        &self.b.x
    }
    fn set_param(&mut self, key: &str, v: f64) -> bool {
        params!(self, key, v; "lr" => lr, "mu" => mu);
        true
    }
}

/* ---------------- AdaGrad ---------------- */

pub struct AdaGrad {
    b: Base,
    r: Vec<f64>,
    pub lr: f64,
    pub delta: f64,
}

impl Default for AdaGrad {
    fn default() -> Self {
        Self { b: Base::default(), r: vec![], lr: 0.5, delta: 1e-7 }
    }
}

impl Optimizer for AdaGrad {
    fn manifest(&self) -> OptimizerManifest {
        OptimizerManifest {
            id: "adagrad",
            name: "AdaGrad",
            family: "adaptive",
            params: vec![
                lr_spec(0.5),
                ParamSpec::real("delta", r"\delta", "small constant for numerical stability", 1e-7, 1e-12, 1e-2, true),
            ],
            tex: r"r \leftarrow r + g \odot g,\quad \theta \leftarrow \theta - \frac{\eta}{\delta + \sqrt r} \odot g",
            pseudocode: Pseudocode {
                source: "Goodfellow, Bengio & Courville (2016), Deep Learning, Algorithm 8.4 (AdaGrad; Duchi, Hazan & Singer 2011)",
                lines: lines(&[
                    (0, false, r"\textbf{Require: } \text{Global learning rate } \epsilon"),
                    (0, false, r"\textbf{Require: } \text{Initial parameter } \theta"),
                    (
                        0,
                        false,
                        r"\textbf{Require: } \text{Small constant } \delta, \text{ perhaps } 10^{-7}, \text{ for numerical stability}",
                    ),
                    (0, false, r"\text{Initialize gradient accumulation variable } r = 0"),
                    (0, false, r"\textbf{while } \text{stopping criterion not met } \textbf{do}"),
                    (
                        1,
                        true,
                        r"\text{Compute gradient: } g \leftarrow \frac1m \nabla_\theta \textstyle\sum_i L(f(x^{(i)}; \theta), y^{(i)})",
                    ),
                    (1, true, r"\text{Accumulate squared gradient: } r \leftarrow r + g \odot g"),
                    (
                        1,
                        true,
                        r"\text{Compute update: } \Delta\theta \leftarrow -\frac{\epsilon}{\delta + \sqrt r} \odot g",
                    ),
                    (1, true, r"\text{Apply update: } \theta \leftarrow \theta + \Delta\theta"),
                    (0, false, r"\textbf{end while}"),
                ]),
                deviations: vec![
                    eta_named(0, "The global learning rate $\\epsilon$ is labelled $\\eta$ in the controls."),
                    exact_gradient(5),
                    no_stopping_test(4),
                ],
            },
        }
    }
    fn init(&mut self, x0: &[f64]) {
        self.b.init(x0);
        self.r = vec![0.0; x0.len()];
    }
    fn step(&mut self, p: &dyn Problem, _: &mut Rng, mut trc: Option<&mut Trace>) -> StepInfo {
        let info = self.b.eval(p);
        tr!(trc, 5, "g" => self.b.g);
        for (r, g) in self.r.iter_mut().zip(&self.b.g) {
            *r += g * g;
        }
        tr!(trc, 6, "r" => self.r);
        let dt: Vec<f64> = self.r.iter().zip(&self.b.g).map(|(r, g)| -self.lr / (self.delta + r.sqrt()) * g).collect();
        tr!(trc, 7, r"\Delta\theta" => dt);
        for (x, d) in self.b.x.iter_mut().zip(&dt) {
            *x += d;
        }
        tr!(trc, 8, r"\theta" => self.b.x);
        info
    }
    fn x(&self) -> &[f64] {
        &self.b.x
    }
    fn set_param(&mut self, key: &str, v: f64) -> bool {
        params!(self, key, v; "lr" => lr, "delta" => delta);
        true
    }
}

/* ---------------- RMSProp ---------------- */

pub struct RmsProp {
    b: Base,
    r: Vec<f64>,
    pub lr: f64,
    pub rho: f64,
    pub delta: f64,
}

impl Default for RmsProp {
    fn default() -> Self {
        Self { b: Base::default(), r: vec![], lr: 0.01, rho: 0.9, delta: 1e-6 }
    }
}

impl Optimizer for RmsProp {
    fn manifest(&self) -> OptimizerManifest {
        OptimizerManifest {
            id: "rmsprop",
            name: "RMSProp",
            family: "adaptive",
            params: vec![
                lr_spec(0.01),
                ParamSpec::real("rho", r"\rho", "decay rate of the squared-gradient average", 0.9, 0.0, 0.9999, false),
                ParamSpec::real("delta", r"\delta", "small constant for numerical stability", 1e-6, 1e-12, 1e-2, true),
            ],
            tex: r"r \leftarrow \rho r + (1-\rho)\, g \odot g,\quad \theta \leftarrow \theta - \frac{\eta}{\sqrt{\delta + r}} \odot g",
            pseudocode: Pseudocode {
                source: "Goodfellow, Bengio & Courville (2016), Deep Learning, Algorithm 8.5 (RMSProp; Hinton 2012)",
                lines: lines(&[
                    (0, false, r"\textbf{Require: } \text{Global learning rate } \epsilon, \text{ decay rate } \rho"),
                    (0, false, r"\textbf{Require: } \text{Initial parameter } \theta"),
                    (
                        0,
                        false,
                        r"\textbf{Require: } \text{Small constant } \delta, \text{ usually } 10^{-6}, \text{ used to stabilize division by small numbers}",
                    ),
                    (0, false, r"\text{Initialize accumulation variables } r = 0"),
                    (0, false, r"\textbf{while } \text{stopping criterion not met } \textbf{do}"),
                    (
                        1,
                        true,
                        r"\text{Compute gradient: } g \leftarrow \frac1m \nabla_\theta \textstyle\sum_i L(f(x^{(i)}; \theta), y^{(i)})",
                    ),
                    (1, true, r"\text{Accumulate squared gradient: } r \leftarrow \rho r + (1 - \rho)\, g \odot g"),
                    (
                        1,
                        true,
                        r"\text{Compute parameter update: } \Delta\theta = -\frac{\epsilon}{\sqrt{\delta + r}} \odot g",
                    ),
                    (1, true, r"\text{Apply update: } \theta \leftarrow \theta + \Delta\theta"),
                    (0, false, r"\textbf{end while}"),
                ]),
                deviations: vec![
                    eta_named(0, "The global learning rate $\\epsilon$ is labelled $\\eta$ in the controls."),
                    exact_gradient(5),
                    no_stopping_test(4),
                ],
            },
        }
    }
    fn init(&mut self, x0: &[f64]) {
        self.b.init(x0);
        self.r = vec![0.0; x0.len()];
    }
    fn step(&mut self, p: &dyn Problem, _: &mut Rng, mut trc: Option<&mut Trace>) -> StepInfo {
        let info = self.b.eval(p);
        tr!(trc, 5, "g" => self.b.g);
        for (r, g) in self.r.iter_mut().zip(&self.b.g) {
            *r = self.rho * *r + (1.0 - self.rho) * g * g;
        }
        tr!(trc, 6, "r" => self.r);
        let dt: Vec<f64> = self.r.iter().zip(&self.b.g).map(|(r, g)| -self.lr / (self.delta + r).sqrt() * g).collect();
        tr!(trc, 7, r"\Delta\theta" => dt);
        for (x, d) in self.b.x.iter_mut().zip(&dt) {
            *x += d;
        }
        tr!(trc, 8, r"\theta" => self.b.x);
        info
    }
    fn x(&self) -> &[f64] {
        &self.b.x
    }
    fn set_param(&mut self, key: &str, v: f64) -> bool {
        params!(self, key, v; "lr" => lr, "rho" => rho, "delta" => delta);
        true
    }
}

/* ---------------- Adam and AdamW ---------------- */

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

fn adam_params(lr: f64) -> Vec<ParamSpec> {
    vec![
        lr_spec(lr),
        ParamSpec::real("beta1", r"\beta_1", "decay of the gradient average (momentum)", 0.9, 0.0, 0.999, false),
        ParamSpec::real("beta2", r"\beta_2", "decay of the squared-gradient average", 0.999, 0.9, 0.99999, false),
        ParamSpec::real("eps", r"\epsilon", "keeps the denominator away from zero", 1e-8, 1e-12, 1e-2, true),
    ]
}

impl Adam {
    /// Lines 7–12 of Kingma & Ba's Algorithm 1 (also AdamW's 4–10, offset by `o`); returns m̂/(√v̂+ε).
    fn moments(&mut self, trc: &mut Option<&mut Trace>, o: usize) -> Vec<f64> {
        self.t += 1;
        tr!(trc, 6 + o, "t" => self.t);
        tr!(trc, 7 + o, "g_t" => self.b.g);
        let (b1, b2) = (self.beta1, self.beta2);
        for ((m, v), g) in self.m.iter_mut().zip(&mut self.v).zip(&self.b.g) {
            *m = b1 * *m + (1.0 - b1) * g;
            *v = b2 * *v + (1.0 - b2) * g * g;
        }
        tr!(trc, 8 + o, "m_t" => self.m);
        tr!(trc, 9 + o, "v_t" => self.v);
        let (c1, c2) = (1.0 - b1.powi(self.t), 1.0 - b2.powi(self.t));
        let mh: Vec<f64> = self.m.iter().map(|m| m / c1).collect();
        let vh: Vec<f64> = self.v.iter().map(|v| v / c2).collect();
        tr!(trc, 10 + o, r"\hat m_t" => mh);
        tr!(trc, 11 + o, r"\hat v_t" => vh);
        mh.iter().zip(&vh).map(|(m, v)| m / (v.sqrt() + self.eps)).collect()
    }
}

impl Optimizer for Adam {
    fn manifest(&self) -> OptimizerManifest {
        OptimizerManifest {
            id: "adam",
            name: "Adam",
            family: "adaptive",
            params: adam_params(0.05),
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
                    eta_named(0, "The stepsize $\\alpha$ is labelled $\\eta$ in the controls."),
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
    fn step(&mut self, p: &dyn Problem, _: &mut Rng, mut trc: Option<&mut Trace>) -> StepInfo {
        let info = self.b.eval(p);
        let dir = self.moments(&mut trc, 0);
        for (x, d) in self.b.x.iter_mut().zip(&dir) {
            *x -= self.lr * d;
        }
        tr!(trc, 12, r"\theta_t" => self.b.x);
        info
    }
    fn x(&self) -> &[f64] {
        &self.b.x
    }
    fn set_param(&mut self, key: &str, v: f64) -> bool {
        params!(self, key, v; "lr" => lr, "beta1" => beta1, "beta2" => beta2, "eps" => eps);
        true
    }
}

pub struct AdamW {
    a: Adam,
    pub wd: f64,
}

impl Default for AdamW {
    fn default() -> Self {
        Self { a: Adam::default(), wd: 0.01 }
    }
}

impl Optimizer for AdamW {
    fn manifest(&self) -> OptimizerManifest {
        let mut params = adam_params(0.05);
        params.push(ParamSpec::real(
            "wd",
            r"\lambda",
            "decoupled weight decay: pulls θ towards 0 each step",
            0.01,
            0.0,
            1.0,
            false,
        ));
        OptimizerManifest {
            id: "adamw",
            name: "AdamW",
            family: "adaptive",
            params,
            tex: r"\theta_t \leftarrow \theta_{t-1} - \eta_t \left(\alpha\, \hat m_t / (\sqrt{\hat v_t} + \epsilon) + \lambda \theta_{t-1}\right)",
            pseudocode: Pseudocode {
                source: "Loshchilov & Hutter (2019), Decoupled Weight Decay Regularization, Algorithm 2 (AdamW)",
                lines: lines(&[
                    (
                        0,
                        false,
                        r"\textbf{given } \alpha = 0.001,\ \beta_1 = 0.9,\ \beta_2 = 0.999,\ \epsilon = 10^{-8},\ \lambda \in \mathbb{R}",
                    ),
                    (
                        0,
                        false,
                        r"\textbf{initialize } t \leftarrow 0,\ \theta_{t=0} \in \mathbb{R}^n,\ m_{t=0} \leftarrow 0,\ v_{t=0} \leftarrow 0,\ \eta_{t=0} \in \mathbb{R}",
                    ),
                    (0, false, r"\textbf{repeat}"),
                    (1, true, r"t \leftarrow t + 1"),
                    (1, true, r"\nabla f_t(\theta_{t-1}) \leftarrow \text{SelectBatch}(\theta_{t-1})"),
                    (1, true, r"g_t \leftarrow \nabla f_t(\theta_{t-1})"),
                    (1, true, r"m_t \leftarrow \beta_1 m_{t-1} + (1 - \beta_1) g_t"),
                    (1, true, r"v_t \leftarrow \beta_2 v_{t-1} + (1 - \beta_2) g_t^2"),
                    (1, true, r"\hat m_t \leftarrow m_t / (1 - \beta_1^t)"),
                    (1, true, r"\hat v_t \leftarrow v_t / (1 - \beta_2^t)"),
                    (1, true, r"\eta_t \leftarrow \text{SetScheduleMultiplier}(t)"),
                    (
                        1,
                        true,
                        r"\theta_t \leftarrow \theta_{t-1} - \eta_t \left(\alpha\, \hat m_t / (\sqrt{\hat v_t} + \epsilon) + \lambda \theta_{t-1}\right)",
                    ),
                    (0, false, r"\textbf{until } \text{stopping criterion is met}"),
                    (0, false, r"\textbf{return } \text{optimized parameters } \theta_t"),
                ]),
                deviations: vec![
                    eta_named(
                        0,
                        "The stepsize $\\alpha$ is labelled $\\eta$ in the controls (the paper's $\\eta_t$ is the schedule multiplier).",
                    ),
                    exact_gradient(4),
                    Deviation {
                        line: Some(10),
                        what: "$\\eta_t = 1$ for every $t$: no learning-rate schedule.",
                        why: "Schedules are a separate, later option shared by all optimizers; with none, $\\eta_t$ drops out.",
                    },
                    no_stopping_test(12),
                ],
            },
        }
    }
    fn init(&mut self, x0: &[f64]) {
        self.a.init(x0);
    }
    fn step(&mut self, p: &dyn Problem, _: &mut Rng, mut trc: Option<&mut Trace>) -> StepInfo {
        let info = self.a.b.eval(p);
        // AdamW's lines are Adam's shifted by -3 (t ← t+1 is line 3 here, 6 in Adam), with SelectBatch between
        self.a.t += 1;
        tr!(trc, 3, "t" => self.a.t);
        tr!(trc, 4, r"\nabla f_t(\theta_{t-1})" => self.a.b.g);
        self.a.t -= 1;
        let dir = {
            // reuse Adam's moment update, but record it on AdamW's lines 5–9
            let mut sub = Trace::default();
            let traced = trc.is_some();
            let d = self.a.moments(&mut if traced { Some(&mut sub) } else { None }, 0);
            if let Some(t) = trc.as_deref_mut() {
                for e in sub.entries.into_iter().filter(|e| e.line >= 7) {
                    t.line(e.line - 2, e.vars);
                }
            }
            d
        };
        tr!(trc, 10, r"\eta_t" => 1.0);
        let (lr, wd) = (self.a.lr, self.wd);
        for (x, d) in self.a.b.x.iter_mut().zip(&dir) {
            *x -= lr * d + wd * *x;
        }
        tr!(trc, 11, r"\theta_t" => self.a.b.x);
        info
    }
    fn x(&self) -> &[f64] {
        &self.a.b.x
    }
    fn set_param(&mut self, key: &str, v: f64) -> bool {
        if key == "wd" {
            self.wd = v;
            return true;
        }
        self.a.set_param(key, v)
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::run;
    use super::*;
    use crate::problems;

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
    fn gd_line_searches_make_progress_on_rosenbrock() {
        let p = problems::by_id("rosenbrock", 2).unwrap();
        for ls in [1.0, 2.0] {
            let mut gd = GradientDescent { ls, ..Default::default() };
            let f = run(&mut gd, p.as_ref(), &[-1.2, 1.0], 3000);
            assert!(f < 1e-2, "ls {ls}: f = {f}");
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
    fn adamw_with_zero_decay_is_adam() {
        let p = problems::by_id("ackley", 2).unwrap();
        let mut a = Adam::default();
        let mut w = AdamW { wd: 0.0, ..Default::default() };
        run(&mut a, p.as_ref(), &[3.1, -2.4], 200);
        run(&mut w, p.as_ref(), &[3.1, -2.4], 200);
        assert_eq!(a.x(), w.x());
    }

    #[test]
    fn nesterov_beats_momentum_on_the_bowl() {
        let p = problems::by_id("bowl", 2).unwrap();
        let mut m = Momentum { lr: 0.04, beta: 0.9, ..Default::default() };
        let mut n = Nesterov { lr: 0.04, mu: 0.9, ..Default::default() };
        let fm = run(&mut m, p.as_ref(), &[-1.8, 1.6], 60);
        let fn_ = run(&mut n, p.as_ref(), &[-1.8, 1.6], 60);
        assert!(fn_ < fm, "nesterov {fn_} vs momentum {fm}");
    }
}
