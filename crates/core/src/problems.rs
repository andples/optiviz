//! Continuous test problems. Each has an analytic gradient (checked against finite differences in the tests)
//! and, in 2D, a shader expression used to draw its surface (checked against `eval` by crates/gpu-test).

use crate::manifest::ProblemManifest;
use std::f64::consts::{E, TAU};

pub trait Problem: Send + Sync {
    fn manifest(&self) -> ProblemManifest;
    fn dim(&self) -> usize;
    fn eval(&self, x: &[f64]) -> f64;
    /// Writes ∇f(x) into `g` and returns f(x).
    fn grad(&self, x: &[f64], g: &mut [f64]) -> f64;
}

/// Every problem id the registry knows, in gallery order.
pub const IDS: &[&str] = &["rosenbrock", "himmelblau", "rastrigin", "ackley", "beale", "bowl"];

/// Builds a problem by id. `dim` is honoured by the N-dimensional problems; 2D-only problems ignore it.
pub fn by_id(id: &str, dim: usize) -> Option<Box<dyn Problem>> {
    let dim = dim.max(2);
    Some(match id {
        "rosenbrock" => Box::new(Rosenbrock { dim }),
        "himmelblau" => Box::new(Himmelblau),
        "rastrigin" => Box::new(Rastrigin { dim }),
        "ackley" => Box::new(Ackley { dim }),
        "beale" => Box::new(Beale),
        "bowl" => Box::new(Bowl { dim, kappa: 20.0 }),
        _ => return None,
    })
}

fn boxed(dim: usize, lo: f64, hi: f64) -> Vec<(f64, f64)> {
    vec![(lo, hi); dim]
}

/// Rosenbrock's banana valley: easy to find, slow to follow.
pub struct Rosenbrock {
    pub dim: usize,
}

impl Problem for Rosenbrock {
    fn manifest(&self) -> ProblemManifest {
        ProblemManifest {
            id: "rosenbrock",
            name: "Rosenbrock",
            tex: r"f(x,y) = (1-x)^2 + 100\,(y-x^2)^2",
            dim: self.dim,
            bounds: if self.dim == 2 { vec![(-2.0, 2.0), (-1.0, 3.0)] } else { boxed(self.dim, -2.0, 2.0) },
            minima: vec![vec![1.0; self.dim]],
            shader_expr: (self.dim == 2).then_some("(1.0 - x)*(1.0 - x) + 100.0*(y - x*x)*(y - x*x)"),
            log_height: true,
            start: (0..self.dim).map(|i| if i % 2 == 0 { -1.2 } else { 1.0 }).collect(),
        }
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn eval(&self, x: &[f64]) -> f64 {
        x.windows(2).map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2)).sum()
    }
    fn grad(&self, x: &[f64], g: &mut [f64]) -> f64 {
        g.fill(0.0);
        let mut f = 0.0;
        for i in 0..x.len() - 1 {
            let (a, b) = (x[i], x[i + 1]);
            let t = b - a * a;
            f += 100.0 * t * t + (1.0 - a) * (1.0 - a);
            g[i] += -400.0 * a * t - 2.0 * (1.0 - a);
            g[i + 1] += 200.0 * t;
        }
        f
    }
}

/// Himmelblau: four equal minima, so where you end up depends on where you start.
pub struct Himmelblau;

impl Problem for Himmelblau {
    fn manifest(&self) -> ProblemManifest {
        ProblemManifest {
            id: "himmelblau",
            name: "Himmelblau",
            tex: r"f(x,y) = (x^2+y-11)^2 + (x+y^2-7)^2",
            dim: 2,
            bounds: boxed(2, -5.0, 5.0),
            minima: vec![
                vec![3.0, 2.0],
                vec![-2.805_118, 3.131_312],
                vec![-3.779_310, -3.283_186],
                vec![3.584_428, -1.848_126],
            ],
            shader_expr: Some("(x*x + y - 11.0)*(x*x + y - 11.0) + (x + y*y - 7.0)*(x + y*y - 7.0)"),
            log_height: true,
            start: vec![-0.3, -0.9],
        }
    }
    fn dim(&self) -> usize {
        2
    }
    fn eval(&self, x: &[f64]) -> f64 {
        let (a, b) = (x[0] * x[0] + x[1] - 11.0, x[0] + x[1] * x[1] - 7.0);
        a * a + b * b
    }
    fn grad(&self, x: &[f64], g: &mut [f64]) -> f64 {
        let (a, b) = (x[0] * x[0] + x[1] - 11.0, x[0] + x[1] * x[1] - 7.0);
        g[0] = 4.0 * x[0] * a + 2.0 * b;
        g[1] = 2.0 * a + 4.0 * x[1] * b;
        a * a + b * b
    }
}

/// Rastrigin: a bowl covered in a regular grid of local minima.
pub struct Rastrigin {
    pub dim: usize,
}

impl Problem for Rastrigin {
    fn manifest(&self) -> ProblemManifest {
        ProblemManifest {
            id: "rastrigin",
            name: "Rastrigin",
            tex: r"f(\mathbf{x}) = 10n + \sum_i \left(x_i^2 - 10\cos 2\pi x_i\right)",
            dim: self.dim,
            bounds: boxed(self.dim, -5.12, 5.12),
            minima: vec![vec![0.0; self.dim]],
            shader_expr: (self.dim == 2).then_some("20.0 + x*x - 10.0*cos(2.0*PI*x) + y*y - 10.0*cos(2.0*PI*y)"),
            log_height: false,
            start: (0..self.dim).map(|i| if i % 2 == 0 { 3.3 } else { -2.7 }).collect(),
        }
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn eval(&self, x: &[f64]) -> f64 {
        10.0 * x.len() as f64 + x.iter().map(|&v| v * v - 10.0 * (TAU * v).cos()).sum::<f64>()
    }
    fn grad(&self, x: &[f64], g: &mut [f64]) -> f64 {
        let mut f = 10.0 * x.len() as f64;
        for (gi, &v) in g.iter_mut().zip(x) {
            let (s, c) = (TAU * v).sin_cos();
            f += v * v - 10.0 * c;
            *gi = 2.0 * v + 10.0 * TAU * s;
        }
        f
    }
}

/// Ackley: nearly flat outside, a deep funnel at the origin, ripples everywhere.
pub struct Ackley {
    pub dim: usize,
}

impl Problem for Ackley {
    fn manifest(&self) -> ProblemManifest {
        ProblemManifest {
            id: "ackley",
            name: "Ackley",
            tex: r"f(\mathbf{x}) = -20\,e^{-0.2\sqrt{\frac1n\sum x_i^2}} - e^{\frac1n\sum\cos 2\pi x_i} + 20 + e",
            dim: self.dim,
            bounds: boxed(self.dim, -5.0, 5.0),
            minima: vec![vec![0.0; self.dim]],
            shader_expr: (self.dim == 2).then_some(
                "-20.0*exp(-0.2*sqrt(0.5*(x*x + y*y))) - exp(0.5*(cos(2.0*PI*x) + cos(2.0*PI*y))) + 20.0 + 2.718281828459045",
            ),
            log_height: false,
            start: (0..self.dim).map(|i| if i % 2 == 0 { 3.1 } else { -2.4 }).collect(),
        }
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn eval(&self, x: &[f64]) -> f64 {
        let n = x.len() as f64;
        let s = x.iter().map(|v| v * v).sum::<f64>() / n;
        let c = x.iter().map(|v| (TAU * v).cos()).sum::<f64>() / n;
        -20.0 * (-0.2 * s.sqrt()).exp() - c.exp() + 20.0 + E
    }
    fn grad(&self, x: &[f64], g: &mut [f64]) -> f64 {
        let n = x.len() as f64;
        let s = x.iter().map(|v| v * v).sum::<f64>() / n;
        let c = x.iter().map(|v| (TAU * v).cos()).sum::<f64>() / n;
        let r = s.sqrt();
        let (e1, e2) = ((-0.2 * r).exp(), c.exp());
        for (gi, &v) in g.iter_mut().zip(x) {
            let radial = if r > 0.0 { 4.0 * e1 * v / (n * r) } else { 0.0 };
            *gi = radial + TAU / n * e2 * (TAU * v).sin();
        }
        -20.0 * e1 - e2 + 20.0 + E
    }
}

/// Beale: a flat plateau with steep walls and narrow valleys meeting near the minimum.
pub struct Beale;

impl Problem for Beale {
    fn manifest(&self) -> ProblemManifest {
        ProblemManifest {
            id: "beale",
            name: "Beale",
            tex: r"f(x,y) = (1.5-x+xy)^2 + (2.25-x+xy^2)^2 + (2.625-x+xy^3)^2",
            dim: 2,
            bounds: boxed(2, -4.5, 4.5),
            minima: vec![vec![3.0, 0.5]],
            shader_expr: Some(
                "(1.5 - x + x*y)*(1.5 - x + x*y) + (2.25 - x + x*y*y)*(2.25 - x + x*y*y) + (2.625 - x + x*y*y*y)*(2.625 - x + x*y*y*y)",
            ),
            log_height: true,
            start: vec![-1.0, 1.4],
        }
    }
    fn dim(&self) -> usize {
        2
    }
    fn eval(&self, x: &[f64]) -> f64 {
        let (a, b) = (x[0], x[1]);
        let t = [1.5 - a + a * b, 2.25 - a + a * b * b, 2.625 - a + a * b * b * b];
        t.iter().map(|v| v * v).sum()
    }
    fn grad(&self, x: &[f64], g: &mut [f64]) -> f64 {
        let (a, b) = (x[0], x[1]);
        let t1 = 1.5 - a + a * b;
        let t2 = 2.25 - a + a * b * b;
        let t3 = 2.625 - a + a * b * b * b;
        g[0] = 2.0 * t1 * (b - 1.0) + 2.0 * t2 * (b * b - 1.0) + 2.0 * t3 * (b * b * b - 1.0);
        g[1] = 2.0 * t1 * a + 4.0 * t2 * a * b + 6.0 * t3 * a * b * b;
        t1 * t1 + t2 * t2 + t3 * t3
    }
}

/// An ill-conditioned quadratic bowl, condition number κ: the clearest view of zig-zagging and momentum.
pub struct Bowl {
    pub dim: usize,
    pub kappa: f64,
}

impl Bowl {
    fn weight(&self, i: usize) -> f64 {
        self.kappa.powf(i as f64 / (self.dim - 1) as f64)
    }
}

impl Problem for Bowl {
    fn manifest(&self) -> ProblemManifest {
        ProblemManifest {
            id: "bowl",
            name: "Elliptic bowl",
            tex: r"f(x,y) = \tfrac12\left(x^2 + 20\,y^2\right)",
            dim: self.dim,
            bounds: boxed(self.dim, -2.0, 2.0),
            minima: vec![vec![0.0; self.dim]],
            shader_expr: (self.dim == 2 && self.kappa == 20.0).then_some("0.5*(x*x + 20.0*y*y)"),
            log_height: false,
            start: (0..self.dim).map(|i| if i % 2 == 0 { -1.8 } else { 1.6 }).collect(),
        }
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn eval(&self, x: &[f64]) -> f64 {
        0.5 * x.iter().enumerate().map(|(i, v)| self.weight(i) * v * v).sum::<f64>()
    }
    fn grad(&self, x: &[f64], g: &mut [f64]) -> f64 {
        let mut f = 0.0;
        for (i, (gi, &v)) in g.iter_mut().zip(x).enumerate() {
            let w = self.weight(i);
            *gi = w * v;
            f += 0.5 * w * v * v;
        }
        f
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Rng;

    #[test]
    fn gradients_match_finite_differences() {
        let mut rng = Rng::new(11);
        for &id in IDS {
            for dim in [2, 5] {
                let p = by_id(id, dim).unwrap();
                let m = p.manifest();
                for _ in 0..50 {
                    let x: Vec<f64> = m.bounds.iter().map(|&(lo, hi)| rng.range(lo, hi)).collect();
                    let mut g = vec![0.0; p.dim()];
                    let f = p.grad(&x, &mut g);
                    assert!((f - p.eval(&x)).abs() <= 1e-9 * f.abs().max(1.0), "{id}: grad's f differs");
                    for i in 0..p.dim() {
                        let h = 1e-6 * x[i].abs().max(1.0);
                        let (mut a, mut b) = (x.clone(), x.clone());
                        a[i] += h;
                        b[i] -= h;
                        let fd = (p.eval(&a) - p.eval(&b)) / (2.0 * h);
                        let tol = 1e-4 * fd.abs().max(g[i].abs()).max(1.0);
                        assert!((fd - g[i]).abs() < tol, "{id} dim {dim} ∂{i}: fd {fd} vs {}", g[i]);
                    }
                }
            }
        }
    }

    #[test]
    fn known_minima_are_minima() {
        for &id in IDS {
            let p = by_id(id, 2).unwrap();
            for xm in p.manifest().minima {
                let mut g = vec![0.0; 2];
                let f = p.grad(&xm, &mut g);
                assert!(f.abs() < 1e-8, "{id}: f(x*) = {f}");
                assert!(g.iter().all(|v| v.abs() < 1e-4), "{id}: ∇f(x*) = {g:?}");
            }
        }
    }

    #[test]
    fn every_2d_problem_has_a_shader_expression_and_valid_start() {
        for &id in IDS {
            let m = by_id(id, 2).unwrap().manifest();
            assert!(m.shader_expr.is_some(), "{id}");
            assert_eq!(m.bounds.len(), 2);
            for (s, (lo, hi)) in m.start.iter().zip(&m.bounds) {
                assert!(lo <= s && s <= hi, "{id} start outside bounds");
            }
        }
    }
}
