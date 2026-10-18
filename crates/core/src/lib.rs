//! optiviz core: problems, optimizers and the manifests that describe them to the UI.
//!
//! Everything here is plain f64 Rust with no I/O, so the same code runs natively (tests, benchmarks, reference
//! runs) and in the browser (WebAssembly). See DESIGN.md §6.2 for the plugin model.

pub mod manifest;
pub mod monitor;
pub mod multistart;
pub mod optimizers;
pub mod problems;
pub mod rng;
pub mod session;
pub mod trace;

pub use manifest::{Deviation, OptimizerManifest, ParamSpec, ProblemManifest, PseudoLine, Pseudocode};
pub use optimizers::Optimizer;
pub use problems::Problem;
pub use rng::Rng;
pub use trace::Trace;

/// What one optimizer step did, for the status line, charts and (later) pseudocode highlighting.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StepInfo {
    /// f at the point the step started from
    pub f: f64,
    /// ‖∇f‖ at that point
    pub grad_norm: f64,
    /// objective/gradient evaluations used by this step
    pub evals: u32,
}
