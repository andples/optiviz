//! Optimizers as state machines stepped one iteration at a time.
//!
//! Each carries its pseudocode as the cited source states it, and the list of ways the code differs and why
//! (DESIGN.md §4.1). The implementations follow those lines in order, and when a tracer is passed they record
//! each line with the values it touched (DESIGN.md §4.2), using the line indices of the pseudocode.

mod first_order;
mod lbfgs;
pub mod linesearch;
mod nelder_mead;
mod newton;

pub use first_order::{AdaGrad, Adam, AdamW, GradientDescent, Momentum, Nesterov, RmsProp};
pub use lbfgs::Lbfgs;
pub use nelder_mead::NelderMead;
pub use newton::Newton;

use crate::manifest::{Deviation, OptimizerManifest, ParamSpec};
use crate::trace::Trace;
use crate::{Problem, Rng, StepInfo};

pub trait Optimizer: Send {
    fn manifest(&self) -> OptimizerManifest;
    /// Resets all state and starts from `x0`.
    fn init(&mut self, x0: &[f64]);
    /// One iteration (one pass through the pseudocode's loop). With a tracer, records every line executed.
    fn step(&mut self, p: &dyn Problem, rng: &mut Rng, tr: Option<&mut Trace>) -> StepInfo;
    /// The current iterate (for simplex methods, the best vertex)
    fn x(&self) -> &[f64];
    /// Live tweak; returns false for an unknown key. State (momentum, moments, memory) is kept.
    fn set_param(&mut self, key: &str, v: f64) -> bool;
    /// Extra geometry to draw, as a closed polygon of points (Nelder–Mead's simplex).
    fn shape(&self) -> Option<Vec<f64>> {
        None
    }
}

/// Gallery order.
pub const IDS: &[&str] =
    &["gd", "momentum", "nesterov", "adagrad", "rmsprop", "adam", "adamw", "newton", "lbfgs", "nelder-mead"];

pub fn by_id(id: &str) -> Option<Box<dyn Optimizer>> {
    Some(match id {
        "gd" => Box::new(GradientDescent::default()),
        "momentum" => Box::new(Momentum::default()),
        "nesterov" => Box::new(Nesterov::default()),
        "adagrad" => Box::new(AdaGrad::default()),
        "rmsprop" => Box::new(RmsProp::default()),
        "adam" => Box::new(Adam::default()),
        "adamw" => Box::new(AdamW::default()),
        "newton" => Box::new(Newton::default()),
        "lbfgs" => Box::new(Lbfgs::default()),
        "nelder-mead" => Box::new(NelderMead::default()),
        _ => return None,
    })
}

/// Every source states a convergence test; here the run continues until you stop it, because watching what
/// happens after "convergence" is part of the point. Stopping tests are stopping points in the debugger.
pub(crate) fn no_stopping_test(line: usize) -> Deviation {
    Deviation {
        line: Some(line),
        what: "No convergence test: the loop runs until you pause it, a stopping point fires, or $x_k$ stops being finite.",
        why: "Tolerances on $\\lVert\\nabla f\\rVert$, on $f$ or on progress are stopping points you can switch on (the debugger's Stops panel); by default you watch the whole run.",
    }
}

/// The exact gradient is used where the source samples a minibatch.
pub(crate) fn exact_gradient(line: usize) -> Deviation {
    Deviation {
        line: Some(line),
        what: "The gradient is exact: no minibatch is sampled.",
        why: "The test surfaces are exact functions. Noisy gradients will be an option on the problem, and then this line samples as in the source.",
    }
}

/// The step size is called η everywhere in the controls.
pub(crate) fn eta_named(line: usize, theirs: &'static str) -> Deviation {
    Deviation {
        line: Some(line),
        what: theirs,
        why: "Every optimizer here calls its step size $\\eta$ so runs can be compared side by side; the update is unchanged.",
    }
}

pub(crate) fn lr_spec(default: f64) -> ParamSpec {
    ParamSpec::real("lr", r"\eta", "step size (learning rate)", default, 1e-5, 2.0, true)
}

pub(crate) fn norm(g: &[f64]) -> f64 {
    g.iter().map(|v| v * v).sum::<f64>().sqrt()
}

pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problems;

    pub(crate) fn run(opt: &mut dyn Optimizer, p: &dyn Problem, x0: &[f64], iters: usize) -> f64 {
        let mut rng = Rng::new(0);
        opt.init(x0);
        for _ in 0..iters {
            opt.step(p, &mut rng, None);
        }
        p.eval(opt.x())
    }

    #[test]
    fn every_optimizer_reaches_the_bowl_minimum() {
        let p = problems::by_id("bowl", 2).unwrap();
        for &id in IDS {
            let mut o = by_id(id).unwrap();
            let lr = match id {
                "adam" | "adamw" => 0.02,
                "rmsprop" => 0.002,
                "adagrad" => 0.5,
                _ => 0.04,
            };
            o.set_param("lr", lr);
            if id == "adamw" {
                o.set_param("wd", 0.0);
            }
            let f = run(o.as_mut(), p.as_ref(), &[-1.8, 1.6], 4000);
            // RMSProp with a constant step hovers at distance ~η around the minimum (its normalized step
            // doesn't shrink with the gradient), so it gets a looser tolerance
            let tol = if id == "rmsprop" { 1e-4 } else { 1e-6 };
            assert!(f < tol, "{id}: f = {f}");
        }
    }

    #[test]
    fn every_optimizer_has_sourced_pseudocode_and_valid_params() {
        for &id in IDS {
            let mut o = by_id(id).unwrap();
            let m = o.manifest();
            assert_eq!(m.id, id);
            let pc = &m.pseudocode;
            assert!(pc.source.contains('(') && pc.source.len() > 20, "{id}: cite the source");
            assert!(pc.lines.iter().any(|l| l.step), "{id}: mark the lines a step executes");
            for d in &pc.deviations {
                if let Some(l) = d.line {
                    assert!(l < pc.lines.len(), "{id}: deviation points past the last line");
                }
                assert!(!d.what.is_empty() && !d.why.is_empty());
            }
            assert!(!o.set_param("nope", 1.0));
            for spec in &m.params {
                assert!(o.set_param(spec.key, spec.default), "{id}.{}", spec.key);
                assert!(spec.min <= spec.default && spec.default <= spec.max, "{id}.{}", spec.key);
            }
        }
    }

    #[test]
    fn traces_only_name_lines_that_exist_and_are_marked_as_step_lines() {
        let p = problems::by_id("rosenbrock", 2).unwrap();
        for &id in IDS {
            let mut o = by_id(id).unwrap();
            let m = o.manifest();
            o.init(&[-1.2, 1.0]);
            let mut rng = Rng::new(0);
            for _ in 0..5 {
                let mut t = Trace::default();
                o.step(p.as_ref(), &mut rng, Some(&mut t));
                assert!(!t.entries.is_empty(), "{id}: a traced step records lines");
                for e in &t.entries {
                    assert!(e.line < m.pseudocode.lines.len(), "{id}: line {}", e.line);
                    assert!(
                        m.pseudocode.lines[e.line].step,
                        "{id}: traced line {} isn't marked as a step line",
                        e.line
                    );
                }
            }
        }
    }

    #[test]
    fn tracing_does_not_change_the_run() {
        let p = problems::by_id("himmelblau", 2).unwrap();
        for &id in IDS {
            let (mut a, mut b) = (by_id(id).unwrap(), by_id(id).unwrap());
            a.init(&[-0.3, -0.9]);
            b.init(&[-0.3, -0.9]);
            let (mut ra, mut rb) = (Rng::new(0), Rng::new(0));
            for _ in 0..50 {
                a.step(p.as_ref(), &mut ra, None);
                b.step(p.as_ref(), &mut rb, Some(&mut Trace::default()));
            }
            assert_eq!(a.x(), b.x(), "{id}");
        }
    }

    #[test]
    fn runs_are_deterministic() {
        let p = problems::by_id("ackley", 2).unwrap();
        for &id in IDS {
            let a = run(by_id(id).unwrap().as_mut(), p.as_ref(), &[3.1, -2.4], 300);
            let b = run(by_id(id).unwrap().as_mut(), p.as_ref(), &[3.1, -2.4], 300);
            assert_eq!(a.to_bits(), b.to_bits(), "{id}");
        }
    }
}
