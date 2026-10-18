//! Manifests: how problems and optimizers describe themselves to the UI, which builds the inspector, the
//! problem gallery and the shaders from them. Serialized to JSON for the web side.

use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct ProblemManifest {
    pub id: &'static str,
    pub name: &'static str,
    /// The formula in TeX, for the formula strip
    pub tex: &'static str,
    pub dim: usize,
    /// Search/view box, one (lo, hi) per dimension
    pub bounds: Vec<(f64, f64)>,
    /// Known global minimizers (2D problems list all of them)
    pub minima: Vec<Vec<f64>>,
    /// f(x, y) as an expression valid in both WGSL and GLSL ES 3.0 (floats only, `PI` defined by the shader),
    /// for 2D problems. Kernel tests check it against `Problem::eval` (crates/gpu-test).
    pub shader_expr: Option<&'static str>,
    /// Draw log(1 + f) rather than f: for functions whose range would flatten the interesting part
    pub log_height: bool,
    /// A sensible default start
    pub start: Vec<f64>,
}

/// A tunable parameter. The inspector is generated from these.
#[derive(Clone, Debug, Serialize)]
pub struct ParamSpec {
    pub key: &'static str,
    /// The mathematical symbol, in TeX (η, β₁, ...)
    pub symbol: &'static str,
    pub description: &'static str,
    pub default: f64,
    pub min: f64,
    pub max: f64,
    /// Slide on a log scale
    pub log: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct OptimizerManifest {
    pub id: &'static str,
    pub name: &'static str,
    pub family: &'static str,
    pub params: Vec<ParamSpec>,
    /// Update rule in TeX, shown with the inspector
    pub tex: &'static str,
}
