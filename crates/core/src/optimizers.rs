//! First-order optimizers as state machines stepped one iteration at a time. M0 has the three the first view
//! needs; M1 adds the rest of DESIGN.md §3.1 against the same trait.

use crate::manifest::{OptimizerManifest, ParamSpec};
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
    fn runs_are_deterministic() {
        let p = problems::by_id("ackley", 2).unwrap();
        let a = run(by_id("adam").unwrap().as_mut(), p.as_ref(), &[3.1, -2.4], 500);
        let b = run(by_id("adam").unwrap().as_mut(), p.as_ref(), &[3.1, -2.4], 500);
        assert_eq!(a.to_bits(), b.to_bits());
    }
}
