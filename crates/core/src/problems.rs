//! Continuous test problems. Each has an analytic gradient (checked against finite differences in the tests),
//! editable coefficients (its `params`, owner's request 2026-10-18) and, in 2D, a shader expression built from
//! the current coefficients (checked against `eval` by crates/gpu-test).
//!
//! Known minima are analytic for the textbook coefficients; when coefficients change and the minima move, they
//! are found numerically (`find_minima`: Newton from a grid of starts, keeping the lowest stationary points with
//! a positive-definite Hessian).

use crate::manifest::{ParamSpec, ProblemManifest};
use crate::{Rng, optimizers::Optimizer};
use std::f64::consts::{E, PI, TAU};

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

/// Coefficient values by key; anything missing takes its default.
pub type Coeffs<'a> = &'a [(&'a str, f64)];

/// Builds a problem by id with its textbook coefficients. `dim` is honoured by the N-dimensional problems.
pub fn by_id(id: &str, dim: usize) -> Option<Box<dyn Problem>> {
    build(id, dim, &[])
}

/// Builds a problem with some coefficients changed (unknown keys are ignored, values are clamped to range).
pub fn build(id: &str, dim: usize, c: Coeffs) -> Option<Box<dyn Problem>> {
    let dim = dim.max(2);
    let specs = specs(id)?;
    let v: Vec<f64> = specs
        .iter()
        .map(|s| {
            let x = c.iter().find(|(k, _)| *k == s.key).map_or(s.default, |&(_, v)| v);
            let x = if x.is_finite() { x.clamp(s.min, s.max) } else { s.default };
            if s.key == "seed" || s.key == "count" { x.round() } else { x }
        })
        .collect();
    let custom = specs.iter().zip(&v).any(|(s, x)| *x != s.default);
    let p: Box<dyn Problem> = match id {
        "rosenbrock" => Box::new(Rosenbrock { dim, a: v[0], b: v[1] }),
        "himmelblau" => Box::new(Himmelblau { a: v[0], b: v[1], custom }),
        "rastrigin" => Box::new(Rastrigin { dim, a: v[0] }),
        "ackley" => Box::new(Ackley { dim, a: v[0], b: v[1], c: v[2] }),
        "beale" => Box::new(Beale { c: [v[0], v[1], v[2]], custom }),
        "bowl" => Box::new(Bowl { dim, kappa: v[0], theta: v[1] }),
        "styblinski-tang" => Box::new(StyblinskiTang { dim, a: v[0], b: v[1], custom }),
        "camel" => Box::new(Camel),
        "saddle" => Box::new(Saddle { c: v[0] }),
        "hills" => Box::new(Hills::new(v[0] as u64, v[1] as usize, v[2])),
        _ => return None,
    };
    Some(p)
}

/// The editable coefficients of each problem.
pub fn specs(id: &str) -> Option<Vec<ParamSpec>> {
    Some(match id {
        "rosenbrock" => vec![
            ParamSpec::real("a", "a", "where the valley ends: the minimum is at (a, a²)", 1.0, -1.5, 1.5, false),
            ParamSpec::real("b", "b", "steepness of the valley walls", 100.0, 1.0, 1000.0, true),
        ],
        "himmelblau" => vec![
            ParamSpec::real("a", "a", "first constant (11 in the textbook form)", 11.0, 0.0, 20.0, false),
            ParamSpec::real("b", "b", "second constant (7 in the textbook form)", 7.0, 0.0, 20.0, false),
        ],
        "rastrigin" => vec![ParamSpec::real("A", "A", "height of the ripples", 10.0, 0.0, 30.0, false)],
        "ackley" => vec![
            ParamSpec::real("a", "a", "depth of the funnel", 20.0, 0.0, 40.0, false),
            ParamSpec::real("b", "b", "how quickly the funnel flattens out", 0.2, 0.01, 1.0, false),
            ParamSpec::real("c", "c", "frequency of the ripples", TAU, 0.0, 4.0 * PI, false),
        ],
        "beale" => vec![
            ParamSpec::real("c1", "c_1", "first constant", 1.5, 0.0, 4.0, false),
            ParamSpec::real("c2", "c_2", "second constant", 2.25, 0.0, 4.0, false),
            ParamSpec::real("c3", "c_3", "third constant", 2.625, 0.0, 4.0, false),
        ],
        "bowl" => vec![
            ParamSpec::real(
                "kappa",
                r"\kappa",
                "condition number: how much more curved one axis is than the other",
                20.0,
                1.0,
                1000.0,
                true,
            ),
            ParamSpec::real(
                "theta",
                r"\theta",
                "rotation of the ellipse, in degrees (coordinate-wise methods feel it)",
                0.0,
                -90.0,
                90.0,
                false,
            ),
        ],
        "styblinski-tang" => vec![
            ParamSpec::real("a", "a", "depth of the four basins (16 in the textbook form)", 16.0, 0.0, 30.0, false),
            ParamSpec::real("b", "b", "tilt between the basins (5 in the textbook form)", 5.0, -15.0, 15.0, false),
        ],
        "camel" => vec![],
        "saddle" => vec![ParamSpec::real(
            "c",
            "c",
            "curvature along y (negative turns the saddle into a bowl)",
            1.0,
            -2.0,
            2.0,
            false,
        )],
        "hills" => vec![
            ParamSpec::real("seed", r"\text{seed}", "which random landscape", 1.0, 0.0, 1e9, false),
            ParamSpec::real("count", "n", "number of hills and pits", 9.0, 1.0, 30.0, false),
            ParamSpec::real("width", "s", "typical width of a hill", 0.7, 0.2, 2.0, false),
        ],
        _ => return None,
    })
}

/// A float literal valid in both WGSL and GLSL.
fn lit(v: f64) -> String {
    let s = format!("{v:?}");
    if v < 0.0 { format!("({s})") } else { s }
}

fn boxed(dim: usize, lo: f64, hi: f64) -> Vec<(f64, f64)> {
    vec![(lo, hi); dim]
}

/// Global minima of a 2D problem inside (a slightly enlarged) `bounds`, found numerically. Empty if none is
/// found there (e.g. unbounded below).
pub fn find_minima(p: &dyn Problem, bounds: &[(f64, f64)]) -> (Vec<Vec<f64>>, Option<f64>) {
    let n = 14;
    let scale = bounds.iter().map(|(a, b)| b - a).fold(0.0, f64::max);
    let mut found: Vec<(Vec<f64>, f64)> = vec![];
    for i in 0..n {
        for j in 0..n {
            let x0 = [
                bounds[0].0 + (bounds[0].1 - bounds[0].0) * (i as f64 + 0.5) / n as f64,
                bounds[1].0 + (bounds[1].1 - bounds[1].0) * (j as f64 + 0.5) / n as f64,
            ];
            let mut o = crate::optimizers::Newton::default();
            o.init(&x0);
            let mut rng = Rng::new(0);
            for _ in 0..60 {
                o.step(p, &mut rng, None);
            }
            let x = o.x().to_vec();
            let inside = x.iter().zip(bounds).all(|(v, (a, b))| *v >= a - 0.05 * (b - a) && *v <= b + 0.05 * (b - a));
            let mut g = [0.0; 2];
            let f = p.grad(&x, &mut g);
            let mut h = [0.0; 4];
            p.hess(&x, &mut h);
            let pd = h[0] > 0.0 && h[0] * h[3] - h[1] * h[2] > 0.0;
            let stationary = g[0].hypot(g[1]) < 1e-6 * (1.0 + f.abs());
            let new = !found.iter().any(|(y, _)| (y[0] - x[0]).hypot(y[1] - x[1]) < 1e-4 * scale);
            if inside && f.is_finite() && stationary && pd && new {
                found.push((x, f));
            }
        }
    }
    let Some(best) = found.iter().map(|(_, f)| *f).reduce(f64::min) else { return (vec![], None) };
    let mins = found.into_iter().filter(|(_, f)| *f <= best + 1e-8 * (1.0 + best.abs())).map(|(x, _)| x).collect();
    (mins, Some(best))
}

/* ---------------- the problems ---------------- */

/// Rosenbrock's banana valley: easy to find, slow to follow.
pub struct Rosenbrock {
    pub dim: usize,
    pub a: f64,
    pub b: f64,
}

impl Problem for Rosenbrock {
    fn manifest(&self) -> ProblemManifest {
        let (a, b) = (self.a, self.b);
        // the minimum: x₁ = a, and each following coordinate is the square of the one before
        let mut xm = vec![a; self.dim];
        for i in 1..self.dim {
            xm[i] = xm[i - 1] * xm[i - 1];
        }
        let f_min = if self.dim == 2 { 0.0 } else { self.eval(&xm) };
        ProblemManifest {
            id: "rosenbrock",
            name: "Rosenbrock",
            tex: r"f(x,y) = (a-x)^2 + b\,(y-x^2)^2",
            dim: self.dim,
            bounds: if self.dim == 2 { vec![(-2.0, 2.0), (-1.0, 3.0)] } else { boxed(self.dim, -2.0, 2.0) },
            minima: if self.dim == 2 || a == 1.0 { vec![xm] } else { vec![] },
            shader_expr: (self.dim == 2)
                .then(|| format!("({} - x)*({} - x) + {}*(y - x*x)*(y - x*x)", lit(a), lit(a), lit(b))),
            log_height: true,
            start: (0..self.dim).map(|i| if i % 2 == 0 { -1.2 } else { 1.0 }).collect(),
            f_min: Some(f_min),
            blurb: "A curved, nearly flat valley: finding it is easy, following it to the end is not.",
            params: specs("rosenbrock").unwrap(),
        }
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn eval(&self, x: &[f64]) -> f64 {
        x.windows(2).map(|w| self.b * (w[1] - w[0] * w[0]).powi(2) + (self.a - w[0]).powi(2)).sum()
    }
    fn grad(&self, x: &[f64], g: &mut [f64]) -> f64 {
        g.fill(0.0);
        let mut f = 0.0;
        for i in 0..x.len() - 1 {
            let (u, v) = (x[i], x[i + 1]);
            let t = v - u * u;
            f += self.b * t * t + (self.a - u) * (self.a - u);
            g[i] += -4.0 * self.b * u * t - 2.0 * (self.a - u);
            g[i + 1] += 2.0 * self.b * t;
        }
        f
    }
}

/// Himmelblau: four equal minima, so where you end up depends on where you start.
pub struct Himmelblau {
    pub a: f64,
    pub b: f64,
    custom: bool,
}

impl Problem for Himmelblau {
    fn manifest(&self) -> ProblemManifest {
        let bounds = boxed(2, -5.0, 5.0);
        let (minima, f_min) = if self.custom {
            find_minima(self, &bounds)
        } else {
            (
                vec![
                    vec![3.0, 2.0],
                    vec![-2.805_118_086_952_745, 3.131_312_518_250_573],
                    vec![-3.779_310_253_377_747, -3.283_185_991_286_170],
                    vec![3.584_428_340_330_492, -1.848_126_526_964_404],
                ],
                Some(0.0),
            )
        };
        let (a, b) = (lit(self.a), lit(self.b));
        ProblemManifest {
            id: "himmelblau",
            name: "Himmelblau",
            tex: r"f(x,y) = (x^2+y-a)^2 + (x+y^2-b)^2",
            dim: 2,
            bounds,
            minima,
            shader_expr: Some(format!("(x*x + y - {a})*(x*x + y - {a}) + (x + y*y - {b})*(x + y*y - {b})")),
            log_height: true,
            start: vec![-0.3, -0.9],
            f_min,
            blurb: "Four equal minima: where you end up depends on where you start.",
            params: specs("himmelblau").unwrap(),
        }
    }
    fn dim(&self) -> usize {
        2
    }
    fn eval(&self, x: &[f64]) -> f64 {
        let (u, v) = (x[0] * x[0] + x[1] - self.a, x[0] + x[1] * x[1] - self.b);
        u * u + v * v
    }
    fn grad(&self, x: &[f64], g: &mut [f64]) -> f64 {
        let (u, v) = (x[0] * x[0] + x[1] - self.a, x[0] + x[1] * x[1] - self.b);
        g[0] = 4.0 * x[0] * u + 2.0 * v;
        g[1] = 2.0 * u + 4.0 * x[1] * v;
        u * u + v * v
    }
}

/// Rastrigin: a bowl covered in a regular grid of local minima.
pub struct Rastrigin {
    pub dim: usize,
    pub a: f64,
}

impl Problem for Rastrigin {
    fn manifest(&self) -> ProblemManifest {
        let a = lit(self.a);
        ProblemManifest {
            id: "rastrigin",
            name: "Rastrigin",
            tex: r"f(\mathbf{x}) = An + \sum_i \left(x_i^2 - A\cos 2\pi x_i\right)",
            dim: self.dim,
            bounds: boxed(self.dim, -5.12, 5.12),
            minima: vec![vec![0.0; self.dim]],
            shader_expr: (self.dim == 2)
                .then(|| format!("2.0*{a} + x*x - {a}*cos(2.0*PI*x) + y*y - {a}*cos(2.0*PI*y)")),
            log_height: false,
            start: (0..self.dim).map(|i| if i % 2 == 0 { 3.3 } else { -2.7 }).collect(),
            f_min: Some(0.0),
            blurb: "A bowl covered in a regular grid of local minima: local methods get stuck in the nearest one.",
            params: specs("rastrigin").unwrap(),
        }
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn eval(&self, x: &[f64]) -> f64 {
        self.a * x.len() as f64 + x.iter().map(|&v| v * v - self.a * (TAU * v).cos()).sum::<f64>()
    }
    fn grad(&self, x: &[f64], g: &mut [f64]) -> f64 {
        let mut f = self.a * x.len() as f64;
        for (gi, &v) in g.iter_mut().zip(x) {
            let (s, c) = (TAU * v).sin_cos();
            f += v * v - self.a * c;
            *gi = 2.0 * v + self.a * TAU * s;
        }
        f
    }
}

/// Ackley: nearly flat outside, a deep funnel at the origin, ripples everywhere.
pub struct Ackley {
    pub dim: usize,
    pub a: f64,
    pub b: f64,
    pub c: f64,
}

impl Problem for Ackley {
    fn manifest(&self) -> ProblemManifest {
        let (a, b, c) = (lit(self.a), lit(self.b), lit(self.c));
        ProblemManifest {
            id: "ackley",
            name: "Ackley",
            tex: r"f(\mathbf{x}) = -a\,e^{-b\sqrt{\frac1n\sum x_i^2}} - e^{\frac1n\sum\cos c x_i} + a + e",
            dim: self.dim,
            bounds: boxed(self.dim, -5.0, 5.0),
            minima: vec![vec![0.0; self.dim]],
            shader_expr: (self.dim == 2).then(|| {
                format!(
                    "-{a}*exp(-{b}*sqrt(0.5*(x*x + y*y))) - exp(0.5*(cos({c}*x) + cos({c}*y))) + {a} + 2.718281828459045"
                )
            }),
            log_height: false,
            start: (0..self.dim).map(|i| if i % 2 == 0 { 3.1 } else { -2.4 }).collect(),
            f_min: Some(0.0),
            blurb: "Nearly flat far out, a deep funnel at the centre, ripples everywhere.",
            params: specs("ackley").unwrap(),
        }
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn eval(&self, x: &[f64]) -> f64 {
        let n = x.len() as f64;
        let s = x.iter().map(|v| v * v).sum::<f64>() / n;
        let c = x.iter().map(|v| (self.c * v).cos()).sum::<f64>() / n;
        -self.a * (-self.b * s.sqrt()).exp() - c.exp() + self.a + E
    }
    fn grad(&self, x: &[f64], g: &mut [f64]) -> f64 {
        let n = x.len() as f64;
        let s = x.iter().map(|v| v * v).sum::<f64>() / n;
        let c = x.iter().map(|v| (self.c * v).cos()).sum::<f64>() / n;
        let r = s.sqrt();
        let (e1, e2) = ((-self.b * r).exp(), c.exp());
        for (gi, &v) in g.iter_mut().zip(x) {
            let radial = if r > 0.0 { self.a * self.b * e1 * v / (n * r) } else { 0.0 };
            *gi = radial + self.c / n * e2 * (self.c * v).sin();
        }
        -self.a * e1 - e2 + self.a + E
    }
}

/// Beale: a flat plateau with steep walls and narrow valleys meeting near the minimum.
pub struct Beale {
    pub c: [f64; 3],
    custom: bool,
}

impl Problem for Beale {
    fn manifest(&self) -> ProblemManifest {
        let bounds = boxed(2, -4.5, 4.5);
        let (minima, f_min) = if self.custom { find_minima(self, &bounds) } else { (vec![vec![3.0, 0.5]], Some(0.0)) };
        let [c1, c2, c3] = self.c.map(lit);
        ProblemManifest {
            id: "beale",
            name: "Beale",
            tex: r"f(x,y) = (c_1-x+xy)^2 + (c_2-x+xy^2)^2 + (c_3-x+xy^3)^2",
            dim: 2,
            bounds,
            minima,
            shader_expr: Some(format!(
                "({c1} - x + x*y)*({c1} - x + x*y) + ({c2} - x + x*y*y)*({c2} - x + x*y*y) + ({c3} - x + x*y*y*y)*({c3} - x + x*y*y*y)"
            )),
            log_height: true,
            start: vec![-1.0, 1.4],
            f_min,
            blurb: "A flat plateau with steep walls; narrow valleys meet near the minimum.",
            params: specs("beale").unwrap(),
        }
    }
    fn dim(&self) -> usize {
        2
    }
    fn eval(&self, x: &[f64]) -> f64 {
        let (a, b) = (x[0], x[1]);
        let t = [self.c[0] - a + a * b, self.c[1] - a + a * b * b, self.c[2] - a + a * b * b * b];
        t.iter().map(|v| v * v).sum()
    }
    fn grad(&self, x: &[f64], g: &mut [f64]) -> f64 {
        let (a, b) = (x[0], x[1]);
        let t1 = self.c[0] - a + a * b;
        let t2 = self.c[1] - a + a * b * b;
        let t3 = self.c[2] - a + a * b * b * b;
        g[0] = 2.0 * t1 * (b - 1.0) + 2.0 * t2 * (b * b - 1.0) + 2.0 * t3 * (b * b * b - 1.0);
        g[1] = 2.0 * t1 * a + 4.0 * t2 * a * b + 6.0 * t3 * a * b * b;
        t1 * t1 + t2 * t2 + t3 * t3
    }
}

/// An ill-conditioned quadratic bowl, condition number κ, optionally rotated by θ.
pub struct Bowl {
    pub dim: usize,
    pub kappa: f64,
    /// degrees; rotates the first two coordinates
    pub theta: f64,
}

impl Bowl {
    fn weight(&self, i: usize) -> f64 {
        self.kappa.powf(i as f64 / (self.dim - 1) as f64)
    }
    /// coordinates in the ellipse's own axes
    fn rot(&self, x: &[f64]) -> Vec<f64> {
        let (s, c) = self.theta.to_radians().sin_cos();
        let mut u = x.to_vec();
        u[0] = c * x[0] + s * x[1];
        u[1] = -s * x[0] + c * x[1];
        u
    }
}

impl Problem for Bowl {
    fn manifest(&self) -> ProblemManifest {
        let (s, c) = self.theta.to_radians().sin_cos();
        let (k, s, c) = (lit(self.kappa), lit(s), lit(c));
        ProblemManifest {
            id: "bowl",
            name: "Elliptic bowl",
            tex: r"f = \tfrac12\left(u^2 + \kappa\, v^2\right),\ (u, v) = R_\theta (x, y)",
            dim: self.dim,
            bounds: boxed(self.dim, -2.0, 2.0),
            minima: vec![vec![0.0; self.dim]],
            shader_expr: (self.dim == 2)
                .then(|| format!("0.5*(({c}*x + {s}*y)*({c}*x + {s}*y) + {k}*({c}*y - {s}*x)*({c}*y - {s}*x))")),
            log_height: false,
            start: (0..self.dim).map(|i| if i % 2 == 0 { -1.8 } else { 1.6 }).collect(),
            f_min: Some(0.0),
            blurb: "A stretched quadratic (curvatures differ ×κ): the clearest view of zig-zag and momentum.",
            params: specs("bowl").unwrap(),
        }
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn eval(&self, x: &[f64]) -> f64 {
        let u = self.rot(x);
        0.5 * u.iter().enumerate().map(|(i, v)| self.weight(i) * v * v).sum::<f64>()
    }
    fn grad(&self, x: &[f64], g: &mut [f64]) -> f64 {
        let u = self.rot(x);
        let mut f = 0.0;
        for (i, (gi, &v)) in g.iter_mut().zip(&u).enumerate() {
            let w = self.weight(i);
            *gi = w * v;
            f += 0.5 * w * v * v;
        }
        // back to x coordinates: g_x = Rᵀ g_u
        let (s, c) = self.theta.to_radians().sin_cos();
        let (g0, g1) = (g[0], g[1]);
        g[0] = c * g0 - s * g1;
        g[1] = s * g0 + c * g1;
        f
    }
}

/// Styblinski–Tang: four basins of different depths, separable.
pub struct StyblinskiTang {
    pub dim: usize,
    pub a: f64,
    pub b: f64,
    custom: bool,
}

const ST_X: f64 = -2.903_534_027_771_177_6;

impl StyblinskiTang {
    fn eval1(&self, v: f64) -> f64 {
        0.5 * (v.powi(4) - self.a * v * v + self.b * v)
    }
}

impl Problem for StyblinskiTang {
    fn manifest(&self) -> ProblemManifest {
        let bounds = boxed(self.dim, -5.0, 5.0);
        let (minima, f_min) = if !self.custom {
            (vec![vec![ST_X; self.dim]], Some(self.dim as f64 * self.eval1(ST_X)))
        } else if self.dim == 2 {
            find_minima(self, &bounds)
        } else {
            (vec![], None)
        };
        let (a, b) = (lit(self.a), lit(self.b));
        ProblemManifest {
            id: "styblinski-tang",
            name: "Styblinski–Tang",
            tex: r"f(\mathbf{x}) = \tfrac12 \sum_i \left(x_i^4 - a\,x_i^2 + b\,x_i\right)",
            dim: self.dim,
            bounds,
            minima,
            shader_expr: (self.dim == 2)
                .then(|| format!("0.5*(x*x*x*x - {a}*x*x + {b}*x + y*y*y*y - {a}*y*y + {b}*y)")),
            log_height: false,
            start: (0..self.dim).map(|i| if i % 2 == 0 { 0.6 } else { 0.3 }).collect(),
            f_min,
            blurb: "Four basins of different depths; the deepest is in one corner.",
            params: specs("styblinski-tang").unwrap(),
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
            *gi = 0.5 * (4.0 * v * v * v - 2.0 * self.a * v + self.b);
        }
        self.eval(x)
    }
}

/// Six-hump camel: two global minima, four local ones, saddles between.
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
            params: vec![],
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

/// A saddle x² − c·y²: for c > 0 no minimum at all (descent slides off along y), for c < 0 a bowl.
pub struct Saddle {
    pub c: f64,
}

impl Problem for Saddle {
    fn manifest(&self) -> ProblemManifest {
        let bowl = self.c < 0.0;
        ProblemManifest {
            id: "saddle",
            name: "Saddle",
            tex: r"f(x,y) = x^2 - c\,y^2",
            dim: 2,
            bounds: boxed(2, -2.0, 2.0),
            minima: if bowl { vec![vec![0.0, 0.0]] } else { vec![] },
            shader_expr: Some(format!("x*x - {}*y*y", lit(self.c))),
            log_height: false,
            start: vec![-1.8, 0.02],
            f_min: bowl.then_some(0.0),
            blurb: "No minimum at all: f is unbounded below, and descent slides off the saddle.",
            params: specs("saddle").unwrap(),
        }
    }
    fn dim(&self) -> usize {
        2
    }
    fn eval(&self, x: &[f64]) -> f64 {
        x[0] * x[0] - self.c * x[1] * x[1]
    }
    fn grad(&self, x: &[f64], g: &mut [f64]) -> f64 {
        g[0] = 2.0 * x[0];
        g[1] = -2.0 * self.c * x[1];
        self.eval(x)
    }
}

/// Random Gaussian hills and pits on a gentle bowl, generated from a seed.
pub struct Hills {
    /// (cx, cy, amplitude, width)
    bumps: Vec<(f64, f64, f64, f64)>,
}

impl Hills {
    pub fn new(seed: u64, count: usize, width: f64) -> Self {
        let mut r = Rng::new(seed ^ 0x6869_6c6c);
        let bumps = (0..count.max(1))
            .map(|_| {
                let sign = if r.f64() < 0.4 { 1.0 } else { -1.0 };
                (r.range(-2.6, 2.6), r.range(-2.6, 2.6), sign * r.range(0.5, 1.5), width * r.range(0.65, 1.4))
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
            let (cx, cy) = (lit(cx), lit(cy));
            expr += &format!(" + {}*exp(-{}*((x - {cx})*(x - {cx}) + (y - {cy})*(y - {cy})))", lit(a), lit(k));
        }
        let bounds = boxed(2, -3.0, 3.0);
        let (minima, f_min) = find_minima(self, &bounds);
        ProblemManifest {
            id: "hills",
            name: "Random hills",
            tex: r"f(x,y) = 0.06\,(x^2+y^2) + \sum_{i=1}^n a_i\, e^{-\lVert (x,y) - c_i \rVert^2 / 2s_i^2}",
            dim: 2,
            bounds,
            minima,
            shader_expr: Some(expr),
            log_height: false,
            start: vec![-2.2, 2.0],
            f_min,
            blurb: "Gaussian hills and pits on a gentle bowl, generated from a seed.",
            params: specs("hills").unwrap(),
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
    if hi <= lo {
        hi = lo + 1.0;
    }
    (lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Some coefficient settings away from the defaults, per problem, to test the general formulas.
    pub(crate) fn variants(id: &str) -> Vec<Vec<(&'static str, f64)>> {
        let mut v = vec![vec![]];
        match id {
            "rosenbrock" => v.push(vec![("a", 0.5), ("b", 20.0)]),
            "himmelblau" => v.push(vec![("a", 9.0), ("b", 5.0)]),
            "rastrigin" => v.push(vec![("A", 3.0)]),
            "ackley" => v.push(vec![("a", 10.0), ("b", 0.5), ("c", 3.0)]),
            "beale" => v.push(vec![("c1", 1.0), ("c2", 2.0), ("c3", 3.0)]),
            "bowl" => v.push(vec![("kappa", 50.0), ("theta", 30.0)]),
            "styblinski-tang" => v.push(vec![("a", 10.0), ("b", -3.0)]),
            "saddle" => v.push(vec![("c", -0.5)]),
            "hills" => v.push(vec![("seed", 7.0), ("count", 4.0), ("width", 1.2)]),
            _ => {}
        }
        v
    }

    #[test]
    fn gradients_match_finite_differences() {
        let mut rng = Rng::new(11);
        for &id in IDS {
            for c in variants(id) {
                for dim in [2, 5] {
                    let p = build(id, dim, &c).unwrap();
                    let m = p.manifest();
                    for _ in 0..40 {
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
                            assert!((fd - g[i]).abs() < tol, "{id} {c:?} dim {dim} ∂{i}: fd {fd} vs {}", g[i]);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn known_minima_are_minima_for_default_and_custom_coefficients() {
        for &id in IDS {
            for c in variants(id) {
                let p = build(id, 2, &c).unwrap();
                let m = p.manifest();
                for xm in &m.minima {
                    let mut g = vec![0.0; 2];
                    let f = p.grad(xm, &mut g);
                    assert!((f - m.f_min.unwrap()).abs() < 1e-7 * (1.0 + f.abs()), "{id} {c:?}: f(x*) = {f}");
                    assert!(g.iter().all(|v| v.abs() < 1e-4), "{id} {c:?}: ∇f(x*) = {g:?}");
                }
            }
        }
    }

    #[test]
    fn numeric_minima_agree_with_the_analytic_ones() {
        for id in ["himmelblau", "beale", "camel"] {
            let p = by_id(id, 2).unwrap();
            let m = p.manifest();
            let (found, f) = find_minima(p.as_ref(), &m.bounds);
            assert_eq!(found.len(), m.minima.len(), "{id}: {found:?}");
            assert!((f.unwrap() - m.f_min.unwrap()).abs() < 1e-8, "{id}");
            for x in &m.minima {
                assert!(found.iter().any(|y| (y[0] - x[0]).hypot(y[1] - x[1]) < 1e-5), "{id}: {x:?}");
            }
        }
    }

    #[test]
    fn coefficients_change_the_function_and_are_clamped() {
        let a = build("rosenbrock", 2, &[]).unwrap();
        let b = build("rosenbrock", 2, &[("a", 0.5)]).unwrap();
        assert_ne!(a.eval(&[0.2, 0.3]), b.eval(&[0.2, 0.3]));
        assert_eq!(b.manifest().minima[0], vec![0.5, 0.25]);
        let c = build("rosenbrock", 2, &[("b", 1e9)]).unwrap();
        assert_eq!(c.eval(&[0.0, 1.0]), 1.0 + 1000.0); // b clamped to its maximum, 1000
        let s = build("saddle", 2, &[("c", -1.0)]).unwrap();
        assert_eq!(s.manifest().f_min, Some(0.0));
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
        let (a, b, c) = (Hills::new(3, 9, 0.7), Hills::new(3, 9, 0.7), Hills::new(4, 9, 0.7));
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
            for spec in &m.params {
                assert!(spec.min <= spec.default && spec.default <= spec.max, "{id}.{}", spec.key);
            }
        }
    }
}
