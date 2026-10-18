//! Step tracing for the debugger (DESIGN.md §4.2): while tracing, an optimizer records each pseudocode line it
//! executes with the values of the variables that line touched, named as in the pseudocode. With no tracer the
//! cost is one branch per line.

use serde::Serialize;

#[derive(Clone, Debug, Default, Serialize)]
pub struct Trace {
    pub entries: Vec<TraceEntry>,
}

#[derive(Clone, Debug, Serialize)]
pub struct TraceEntry {
    /// Index into the manifest's pseudocode lines
    pub line: usize,
    /// (TeX name, value(s)); scalars are one-element vectors
    pub vars: Vec<(&'static str, Vec<f64>)>,
}

impl Trace {
    pub fn line(&mut self, line: usize, vars: Vec<(&'static str, Vec<f64>)>) {
        self.entries.push(TraceEntry { line, vars });
    }
}

/// Anything that can be shown as a variable value.
pub trait Var {
    fn vals(&self) -> Vec<f64>;
}
impl Var for f64 {
    fn vals(&self) -> Vec<f64> {
        vec![*self]
    }
}
impl Var for [f64] {
    fn vals(&self) -> Vec<f64> {
        self.to_vec()
    }
}
impl Var for Vec<f64> {
    fn vals(&self) -> Vec<f64> {
        self.clone()
    }
}
impl<const N: usize> Var for [f64; N] {
    fn vals(&self) -> Vec<f64> {
        self.to_vec()
    }
}
impl Var for i32 {
    fn vals(&self) -> Vec<f64> {
        vec![*self as f64]
    }
}
impl Var for usize {
    fn vals(&self) -> Vec<f64> {
        vec![*self as f64]
    }
}

/// `tr!(tracer, line, "name" => value, ...)`: records a line when tracing.
#[macro_export]
macro_rules! tr {
    ($t:expr, $line:expr $(, $name:expr => $val:expr)* $(,)?) => {
        if let Some(t) = $t.as_deref_mut() {
            t.line($line, vec![$(($name, $crate::trace::Var::vals(&$val))),*]);
        }
    };
}
