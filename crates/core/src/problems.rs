//! Continuous test problems. Each has an analytic gradient (checked against finite differences in the tests)
//! and, in 2D, a shader expression used to draw its surface (checked against `eval` by crates/gpu-test).

use crate::Rng;
use crate::manifest::ProblemManifest;
use std::f64::consts::{E, TAU};

pub trait Problem: Send + Sync {
    fn manifest(&self) -> ProblemManifest;
    fn dim(&self) -> usize;
    fn eval(&self, x: &[f64]) -> f64;
    /// Writes ∇f(x) into `g` and returns f(x).
    fn grad(&self, x: &[f64], g: &mut [f64]) -> f64;
    /// Writes the Hessian (row-major, n×n) into `h`. Default: central differences of the analytic gradient,
    /// accurate to about 1e-8 relative, which is what Newton's method here documents.
    fn hess(&self, x: &[f64], h: &mut [f64]) {
        let n = x.len();
        let (mut a, mut b) = (vec![0.0; n], vec![0.0; n]);
        let mut xp = x.to_vec();
        for j in 0..n {
            let eps = 1e-5 * x[j].abs().max(1.0);
            xp[j] = x[j] + eps;
            self.grad(&xp, &mut a);
            xp[j] = x[j] - eps;
            self.grad(&xp, &mut b);
            xp[j] = x[j];
            for i in 0..n {
                h[i * n + j] = (a[i] - b[i]) / (2.0 * eps);
            }
        }
        // symmetrize
        for i in 0..n {
            for j in 0..i {
                let m = 0.5 * (h[i * n + j] + h[j * n + i]);
                h[i * n + j] = m;
                h[j * n + i] = m;
            }
        }
    }
}

/// Every problem id the registry knows, in gallery order.
pub const IDS: &[&str] = &[
    "rosenbrock",
    "himmelblau",
    "rastrigin",
    "ackley",
    "beale",
    "bowl",
    "styblinski-tang",
    "camel",
    "saddle",
    "hills",
];

/// Builds a problem by id. `dim` is honoured by the N-dimensional problems; 2D-only problems ignore it.
pub fn by_id(id: &str, dim: usize) -> Option<Box<dyn Problem>> {
    by_id_seeded(id, dim, 1)
}

/// As `by_id`; `seed` shapes the randomly generated problems ("hills").
pub fn by_id_seeded(id: &str, dim: usize, seed: u64) -> Option<Box<dyn Problem>> {
    let dim = dim.max(2);
    Some(match id {
        "rosenbrock" => Box::new(Rosenbrock { dim }),
        "himmelblau" => Box::new(Himmelblau),
        "rastrigin" => Box::new(Rastrigin { dim }),
        "ackley" => Box::new(Ackley { dim }),
        "beale" => Box::new(Beale),
        "bowl" => Box::new(Bowl { dim, kappa: 20.0 }),
        "styblinski-tang" => Box::new(StyblinskiTang { dim }),
        "camel" => Box::new(Camel),
        "saddle" => Box::new(Saddle),
        "hills" => Box::new(Hills::new(seed)),
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
            shader_expr: (self.dim == 2).then(|| "(1.0 - x)*(1.0 - x) + 100.0*(y - x*x)*(y - x*x)".into()),
            log_height: true,
            start: (0..self.dim).map(|i| if i % 2 == 0 { -1.2 } else { 1.0 }).collect(),
            f_min: Some(0.0),
            blurb: "A curved, nearly flat valley: finding it is easy, following it to the end is not.",
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
            shader_expr: Some("(x*x + y - 11.0)*(x*x + y - 11.0) + (x + y*y - 7.0)*(x + y*y - 7.0)".into()),
            log_height: true,
            start: vec![-0.3, -0.9],
            f_min: Some(0.0),
            blurb: "Four equal minima: where you end up depends on where you start.",
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
            shader_expr: (self.dim == 2).then(|| "20.0 + x*x - 10.0*cos(2.0*PI*x) + y*y - 10.0*cos(2.0*PI*y)".into()),
            log_height: false,
            start: (0..self.dim).map(|i| if i % 2 == 0 { 3.3 } else { -2.7 }).collect(),
            f_min: Some(0.0),
            blurb: "A bowl covered in a regular grid of local minima: local methods get stuck in the nearest one.",
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
            shader_expr: (self.dim == 2).then(|| {
                "-20.0*exp(-0.2*sqrt(0.5*(x*x + y*y))) - exp(0.5*(cos(2.0*PI*x) + cos(2.0*PI*y))) + 20.0 + 2.718281828459045".into()
            }),
            log_height: false,
            start: (0..self.dim).map(|i| if i % 2 == 0 { 3.1 } else { -2.4 }).collect(),
            f_min: Some(0.0),
            blurb: "Nearly flat far out, a deep funnel at the centre, ripples everywhere.",
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
                "(1.5 - x + x*y)*(1.5 - x + x*y) + (2.25 - x + x*y*y)*(2.25 - x + x*y*y) + (2.625 - x + x*y*y*y)*(2.625 - x + x*y*y*y)".into(),
            ),
            log_height: true,
            start: vec![-1.0, 1.4],
            f_min: Some(0.0),
            blurb: "A flat plateau with steep walls; narrow valleys meet near the minimum.",
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
            shader_expr: (self.dim == 2 && self.kappa == 20.0).then(|| "0.5*(x*x + 20.0*y*y)".into()),
            log_height: false,
            start: (0..self.dim).map(|i| if i % 2 == 0 { -1.8 } else { 1.6 }).collect(),
            f_min: Some(0.0),
            blurb: "A stretched quadratic (curvatures differ ×20): the clearest view of zig-zag and momentum.",
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

/// Styblinski–Tang: one global and several local minima, separable, negative values.
pub struct StyblinskiTang {
    pub dim: usize,
}

const ST_X: f64 = -2.903_534_027_771_177_6;

impl Problem for StyblinskiTang {
    fn manifest(&self) -> ProblemManifest {
        ProblemManifest {
            id: "styblinski-tang",
            name: "Styblinski–Tang",
            tex: r"f(\mathbf{x}) = \tfrac12 \sum_i \left(x_i^4 - 16x_i^2 + 5x_i\right)",
            dim: self.dim,
            bounds: boxed(self.dim, -5.0, 5.0),
            minima: vec![vec![ST_X; self.dim]],
            shader_expr: (self.dim == 2)
                .then(|| "0.5*(x*x*x*x - 16.0*x*x + 5.0*x + y*y*y*y - 16.0*y*y + 5.0*y)".into()),
            log_height: false,
            start: (0..self.dim).map(|i| if i % 2 == 0 { 0.6 } else { 0.3 }).collect(),
            f_min: Some(self.dim as f64 * self.eval1(ST_X)),
            blurb: "Four basins of different depths; the deepest is in one corner.",
        }
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn eval(&self, x: &[f64]) -> f64 {
        x.iter().map(|&v| self.eval1(v)).sum()
    }
    fn grad(&self, x: &[f64], g: &mut [f64]) -> f64 {
        for (gi, &v) in g.iter_mut().zip(x) {
            *gi = 0.5 * (4.0 * v * v * v - 32.0 * v + 5.0);
        }
        self.eval(x)
    }
}

impl StyblinskiTang {
    fn eval1(&self, v: f64) -> f64 {
        0.5 * (v.powi(4) - 16.0 * v * v + 5.0 * v)
    }
}

/// Six-hump camel: two global minima, four local ones, a saddle between.
pub struct Camel;

impl Problem for Camel {
    fn manifest(&self) -> ProblemManifest {
        ProblemManifest {
            id: "camel",
            name: "Six-hump camel",
            tex: r"f(x,y) = \left(4 - 2.1x^2 + \tfrac{x^4}{3}\right)x^2 + xy + (-4 + 4y^2)\,y^2",
            dim: 2,
            bounds: vec![(-2.0, 2.0), (-1.1, 1.1)],
            minima: vec![vec![0.089_842_008_6, -0.712_656_403_0], vec![-0.089_842_008_6, 0.712_656_403_0]],
            shader_expr: Some("(4.0 - 2.1*x*x + x*x*x*x/3.0)*x*x + x*y + (-4.0 + 4.0*y*y)*y*y".into()),
            log_height: false,
            start: vec![-1.5, 0.9],
            f_min: Some(-1.031_628_453_489_877),
            blurb: "Two equal global minima, four local ones and saddles between them.",
        }
    }
    fn dim(&self) -> usize {
        2
    }
    fn eval(&self, x: &[f64]) -> f64 {
        let (a, b) = (x[0], x[1]);
        (4.0 - 2.1 * a * a + a.powi(4) / 3.0) * a * a + a * b + (-4.0 + 4.0 * b * b) * b * b
    }
    fn grad(&self, x: &[f64], g: &mut [f64]) -> f64 {
        let (a, b) = (x[0], x[1]);
        g[0] = 8.0 * a - 8.4 * a.powi(3) + 2.0 * a.powi(5) + b;
        g[1] = a - 8.0 * b + 16.0 * b.powi(3);
        self.eval(x)
    }
}

/// A saddle: no minimum at all; descent methods slide off along y. The divergence handling's test case.
pub struct Saddle;

impl Problem for Saddle {
    fn manifest(&self) -> ProblemManifest {
        ProblemManifest {
            id: "saddle",
            name: "Saddle",
            tex: r"f(x,y) = x^2 - y^2",
            dim: 2,
            bounds: boxed(2, -2.0, 2.0),
            minima: vec![],
            shader_expr: Some("x*x - y*y".into()),
            log_height: false,
            start: vec![-1.8, 0.02],
            f_min: None,
            blurb: "No minimum at all: f is unbounded below, and descent slides off the saddle.",
        }
    }
    fn dim(&self) -> usize {
        2
    }
    fn eval(&self, x: &[f64]) -> f64 {
        x[0] * x[0] - x[1] * x[1]
    }
    fn grad(&self, x: &[f64], g: &mut [f64]) -> f64 {
        g[0] = 2.0 * x[0];
        g[1] = -2.0 * x[1];
        self.eval(x)
    }
}

/// Random Gaussian hills and pits on a gentle bowl, generated from a seed.
pub struct Hills {
    /// (cx, cy, amplitude, width)
    bumps: Vec<(f64, f64, f64, f64)>,
}

impl Hills {
    pub fn new(seed: u64) -> Self {
        let mut r = Rng::new(seed ^ 0x6869_6c6c);
        let bumps = (0..9)
            .map(|_| {
                let sign = if r.f64() < 0.4 { 1.0 } else { -1.0 };
                (r.range(-2.6, 2.6), r.range(-2.6, 2.6), sign * r.range(0.5, 1.5), r.range(0.45, 1.0))
            })
            .collect();
        Hills { bumps }
    }
}

impl Problem for Hills {
    fn manifest(&self) -> ProblemManifest {
        let mut expr = String::from("0.06*(x*x + y*y)");
        for &(cx, cy, a, s) in &self.bumps {
            let k = 1.0 / (2.0 * s * s);
            expr += &format!(
                " + ({a:.6})*exp(-({k:.6})*((x - ({cx:.6}))*(x - ({cx:.6})) + (y - ({cy:.6}))*(y - ({cy:.6}))))"
            );
        }
        ProblemManifest {
            id: "hills",
            name: "Random hills",
            tex: r"f(x,y) = 0.06\,(x^2+y^2) + \sum_i a_i\, e^{-\lVert (x,y) - c_i \rVert^2 / 2s_i^2}",
            dim: 2,
            bounds: boxed(2, -3.0, 3.0),
            minima: vec![],
            shader_expr: Some(expr),
            log_height: false,
            start: vec![-2.2, 2.0],
            f_min: None,
            blurb: "Gaussian hills and pits on a gentle bowl, generated from the run's seed.",
        }
    }
    fn dim(&self) -> usize {
        2
    }
    fn eval(&self, x: &[f64]) -> f64 {
        let mut f = 0.06 * (x[0] * x[0] + x[1] * x[1]);
        for &(cx, cy, a, s) in &self.bumps {
            let k = 1.0 / (2.0 * s * s);
            f += a * (-k * ((x[0] - cx).powi(2) + (x[1] - cy).powi(2))).exp();
        }
        f
    }
    fn grad(&self, x: &[f64], g: &mut [f64]) -> f64 {
        g[0] = 0.12 * x[0];
        g[1] = 0.12 * x[1];
        let mut f = 0.06 * (x[0] * x[0] + x[1] * x[1]);
        for &(cx, cy, a, s) in &self.bumps {
            let k = 1.0 / (2.0 * s * s);
            let e = a * (-k * ((x[0] - cx).powi(2) + (x[1] - cy).powi(2))).exp();
            f += e;
            g[0] += -2.0 * k * (x[0] - cx) * e;
            g[1] += -2.0 * k * (x[1] - cy) * e;
        }
        f
    }
}

/// [min, max] of the drawn height (f, or log(1 + f) for `log_height` problems) over an n×n grid of a box.
pub fn height_range(p: &dyn Problem, bounds: [(f64, f64); 2], n: usize) -> (f64, f64) {
    let log = p.manifest().log_height;
    let (bx, by) = (bounds[0], bounds[1]);
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for i in 0..n {
        for j in 0..n {
            let x = bx.0 + (bx.1 - bx.0) * i as f64 / (n - 1) as f64;
            let y = by.0 + (by.1 - by.0) * j as f64 / (n - 1) as f64;
            let f = p.eval(&[x, y]);
            let h = if log { f.max(0.0).ln_1p() } else { f };
            if h.is_finite() {
                lo = lo.min(h);
                hi = hi.max(h);
            }
        }
    }
    (lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;

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
            let m = p.manifest();
            for xm in &m.minima {
                let mut g = vec![0.0; 2];
                let f = p.grad(xm, &mut g);
                assert!((f - m.f_min.unwrap()).abs() < 1e-7, "{id}: f(x*) = {f}");
                assert!(g.iter().all(|v| v.abs() < 1e-4), "{id}: ∇f(x*) = {g:?}");
            }
        }
    }

    #[test]
    fn hessian_is_symmetric_and_matches_the_bowl() {
        let p = by_id("bowl", 2).unwrap();
        let mut h = [0.0; 4];
        p.hess(&[0.3, -0.7], &mut h);
        assert!((h[0] - 1.0).abs() < 1e-6 && (h[3] - 20.0).abs() < 1e-5 && h[1].abs() < 1e-6, "{h:?}");
    }

    #[test]
    fn hills_depend_on_the_seed_only() {
        let (a, b, c) = (Hills::new(3), Hills::new(3), Hills::new(4));
        assert_eq!(a.eval(&[0.4, 0.1]), b.eval(&[0.4, 0.1]));
        assert_ne!(a.eval(&[0.4, 0.1]), c.eval(&[0.4, 0.1]));
    }

    #[test]
    fn every_2d_problem_has_a_shader_expression_and_valid_start() {
        for &id in IDS {
            let m = by_id(id, 2).unwrap().manifest();
            assert!(m.shader_expr.is_some(), "{id}");
            assert!(!m.blurb.is_empty());
            assert_eq!(m.bounds.len(), 2);
            for (s, (lo, hi)) in m.start.iter().zip(&m.bounds) {
                assert!(lo <= s && s <= hi, "{id} start outside bounds");
            }
        }
    }
}
