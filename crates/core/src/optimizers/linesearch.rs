//! Line searches shared by the optimizers. Each trial is traced on the caller's pseudocode line, so the
//! debugger shows the backtracking happen.

use super::dot;
use crate::Problem;
use crate::tr;
use crate::trace::Trace;

/// Backtracking (Boyd & Vandenberghe 2004, Algorithm 9.2; the same rule as Nocedal & Wright Algorithm 3.1):
/// t := t0; while f(x + tΔx) > f(x) + α t ∇f(x)ᵀΔx, t := βt. Returns (t, evaluations). At most 60 halvings.
#[allow(clippy::too_many_arguments)]
pub fn backtracking(
    p: &dyn Problem,
    x: &[f64],
    d: &[f64],
    f0: f64,
    slope: f64,
    t0: f64,
    alpha: f64,
    beta: f64,
    mut trc: Option<&mut Trace>,
    line: usize,
) -> (f64, u32) {
    let mut t = t0;
    let mut xt = vec![0.0; x.len()];
    let mut evals = 0;
    for _ in 0..60 {
        for i in 0..x.len() {
            xt[i] = x[i] + t * d[i];
        }
        let ft = p.eval(&xt);
        evals += 1;
        let ok = ft <= f0 + alpha * t * slope;
        tr!(trc, line, "t" => t, r"f(x + t\Delta x)" => ft, r"f(x) + \alpha t \nabla f^\top \Delta x" => f0 + alpha * t * slope);
        if ok && ft.is_finite() {
            break;
        }
        t *= beta;
    }
    (t, evals)
}

/// The result of a strong-Wolfe search: step, f and ∇f there, evaluations used.
pub struct WolfeStep {
    pub alpha: f64,
    pub f: f64,
    pub g: Vec<f64>,
    pub evals: u32,
}

/// Line search for the strong Wolfe conditions (Nocedal & Wright 2006, Algorithm 3.5, with `zoom` from
/// Algorithm 3.6). Trial steps in `zoom` come from the minimizer of the cubic interpolating φ and φ' at the
/// bracket ends (N&W eq. 3.59), safeguarded to the middle 80% of the bracket. At most 25 trials in total;
/// if none satisfies the conditions, the best point seen is returned.
#[allow(clippy::too_many_arguments)]
pub fn strong_wolfe(
    p: &dyn Problem,
    x: &[f64],
    d: &[f64],
    f0: f64,
    g0: &[f64],
    alpha1: f64,
    c1: f64,
    c2: f64,
    mut trc: Option<&mut Trace>,
    line: usize,
) -> WolfeStep {
    let n = x.len();
    let dphi0 = dot(g0, d);
    let mut g = vec![0.0; n];
    let mut xt = vec![0.0; n];
    let mut evals = 0u32;
    // φ(α) and φ'(α)
    let mut phi = |a: f64, g: &mut Vec<f64>, evals: &mut u32| -> (f64, f64) {
        for i in 0..n {
            xt[i] = x[i] + a * d[i];
        }
        let f = p.grad(&xt, g);
        *evals += 1;
        (f, dot(g, d))
    };
    let mut best = WolfeStep { alpha: 0.0, f: f0, g: g0.to_vec(), evals: 0 };
    if dphi0 >= 0.0 || !dphi0.is_finite() {
        // not a descent direction (or a zero gradient): no step
        return best;
    }
    let (mut a_prev, mut f_prev, mut d_prev) = (0.0, f0, dphi0);
    let mut a = alpha1;
    let mut trials = 0;
    // Algorithm 3.5: expand until the minimum is bracketed
    let (mut lo, mut hi);
    loop {
        let (fa, da) = phi(a, &mut g, &mut evals);
        trials += 1;
        tr!(trc, line, r"\alpha" => a, r"\phi(\alpha)" => fa, r"\phi'(\alpha)" => da);
        if fa.is_finite() && fa < best.f {
            best = WolfeStep { alpha: a, f: fa, g: g.clone(), evals };
        }
        if !fa.is_finite() || fa > f0 + c1 * a * dphi0 || (trials > 1 && fa >= f_prev) {
            lo = (a_prev, f_prev, d_prev);
            hi = (a, fa, da);
            break;
        }
        if da.abs() <= -c2 * dphi0 {
            return WolfeStep { alpha: a, f: fa, g, evals };
        }
        if da >= 0.0 {
            lo = (a, fa, da);
            hi = (a_prev, f_prev, d_prev);
            break;
        }
        if trials >= 25 {
            best.evals = evals;
            return best;
        }
        (a_prev, f_prev, d_prev) = (a, fa, da);
        a *= 2.0;
    }
    // Algorithm 3.6: zoom between lo (lower φ) and hi
    while trials < 25 {
        let (a0, f0z, d0) = lo;
        let (a1, f1, d1) = hi;
        let (l, h) = (a0.min(a1), a0.max(a1));
        let mut aj = if f1.is_finite() && d1.is_finite() {
            // N&W (3.59)
            let d1c = d0 + d1 - 3.0 * (f0z - f1) / (a0 - a1);
            let rad = d1c * d1c - d0 * d1;
            if rad >= 0.0 {
                let d2 = (a1 - a0).signum() * rad.sqrt();
                a1 - (a1 - a0) * (d1 + d2 - d1c) / (d1 - d0 + 2.0 * d2)
            } else {
                f64::NAN
            }
        } else {
            f64::NAN
        };
        if !aj.is_finite() || aj < l + 0.1 * (h - l) || aj > h - 0.1 * (h - l) {
            aj = 0.5 * (l + h);
        }
        let (fj, dj) = phi(aj, &mut g, &mut evals);
        trials += 1;
        tr!(trc, line, r"\alpha" => aj, r"\phi(\alpha)" => fj, r"\phi'(\alpha)" => dj);
        if fj.is_finite() && fj < best.f {
            best = WolfeStep { alpha: aj, f: fj, g: g.clone(), evals };
        }
        if !fj.is_finite() || fj > f0 + c1 * aj * dphi0 || fj >= f0z {
            hi = (aj, fj, dj);
        } else {
            if dj.abs() <= -c2 * dphi0 {
                return WolfeStep { alpha: aj, f: fj, g, evals };
            }
            if dj * (a1 - a0) >= 0.0 {
                hi = lo;
            }
            lo = (aj, fj, dj);
        }
        if (h - l).abs() < 1e-16 {
            break;
        }
    }
    best.evals = evals;
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problems;

    #[test]
    fn strong_wolfe_conditions_hold_on_rosenbrock() {
        let p = problems::by_id("rosenbrock", 2).unwrap();
        let x = [-1.2, 1.0];
        let mut g = [0.0; 2];
        let f0 = p.grad(&x, &mut g);
        let d = [-g[0], -g[1]];
        let s = strong_wolfe(p.as_ref(), &x, &d, f0, &g, 1.0, 1e-4, 0.9, None, 0);
        let dphi0 = dot(&g, &d);
        assert!(s.f <= f0 + 1e-4 * s.alpha * dphi0, "sufficient decrease");
        assert!(dot(&s.g, &d).abs() <= 0.9 * dphi0.abs() + 1e-12, "curvature");
    }

    #[test]
    fn backtracking_satisfies_armijo() {
        let p = problems::by_id("beale", 2).unwrap();
        let x = [-1.0, 1.4];
        let mut g = [0.0; 2];
        let f0 = p.grad(&x, &mut g);
        let d = [-g[0], -g[1]];
        let slope = dot(&g, &d);
        let (t, _) = backtracking(p.as_ref(), &x, &d, f0, slope, 1.0, 0.25, 0.5, None, 0);
        let xt = [x[0] + t * d[0], x[1] + t * d[1]];
        assert!(p.eval(&xt) <= f0 + 0.25 * t * slope);
    }
}
