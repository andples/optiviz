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
    pub shader_expr: Option<String>,
    /// Draw log(1 + f) rather than f: for functions whose range would flatten the interesting part
    pub log_height: bool,
    /// A sensible default start
    pub start: Vec<f64>,
    /// The global minimum value, when known (charts plot f − f* on a log scale)
    pub f_min: Option<f64>,
    /// One line on what makes this landscape interesting
    pub blurb: &'static str,
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
    /// For a discrete choice: the option labels; the value is the index. Empty for continuous parameters.
    pub choices: &'static [&'static str],
}

impl ParamSpec {
    pub const fn real(
        key: &'static str,
        symbol: &'static str,
        description: &'static str,
        default: f64,
        min: f64,
        max: f64,
        log: bool,
    ) -> Self {
        ParamSpec { key, symbol, description, default, min, max, log, choices: &[] }
    }
    pub const fn choice(
        key: &'static str,
        symbol: &'static str,
        description: &'static str,
        default: usize,
        choices: &'static [&'static str],
    ) -> Self {
        ParamSpec {
            key,
            symbol,
            description,
            default: default as f64,
            min: 0.0,
            max: (choices.len() - 1) as f64,
            log: false,
            choices,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct OptimizerManifest {
    pub id: &'static str,
    pub name: &'static str,
    pub family: &'static str,
    pub params: Vec<ParamSpec>,
    /// Update rule in TeX, shown with the inspector
    pub tex: &'static str,
    pub pseudocode: Pseudocode,
}

/// The algorithm as its source states it, line for line, in TeX, plus every way the implementation differs
/// from that statement and why. The rule (DESIGN.md §3): match the literature as closely as possible; where
/// the code can't or shouldn't, say so here.
#[derive(Clone, Debug, Serialize)]
pub struct Pseudocode {
    /// Where the pseudocode comes from: author, year, title, and the algorithm/equation number
    pub source: &'static str,
    pub lines: Vec<PseudoLine>,
    pub deviations: Vec<Deviation>,
}

#[derive(Clone, Debug, Serialize)]
pub struct PseudoLine {
    /// The line in TeX (keywords in \textbf{...})
    pub tex: &'static str,
    pub indent: u8,
    /// Executed by every `Optimizer::step` (one loop iteration); the UI highlights these when stepping
    pub step: bool,
}

/// A difference between the source's pseudocode and what runs. `line` is the index of the line it concerns.
/// `what` and `why` are prose with inline TeX between `$` signs.
#[derive(Clone, Debug, Serialize)]
pub struct Deviation {
    pub line: Option<usize>,
    pub what: &'static str,
    pub why: &'static str,
}

/// Shorthand for building pseudocode lines: (indent, executed per step, TeX).
pub fn lines(spec: &[(u8, bool, &'static str)]) -> Vec<PseudoLine> {
    spec.iter().map(|&(indent, step, tex)| PseudoLine { tex, indent, step }).collect()
}
