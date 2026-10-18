//! Many independent runs at once (many-start mode, and the M0 threading benchmark). With the `parallel`
//! feature the starts are split across a rayon pool (native threads, or Web Workers via wasm-bindgen-rayon);
//! results are identical either way because every start is independent and the reduction order is fixed.

use crate::{Rng, optimizers, problems};

/// Runs `opt_id` from each start (row-major, `dim` values per start) for `iters` steps, overwriting `starts`
/// with the final points. Returns the best final f (min, reduced in index order).
pub fn run_many(problem_id: &str, opt_id: &str, dim: usize, starts: &mut [f64], iters: usize) -> f64 {
    let p = problems::by_id(problem_id, dim).expect("problem id");
    let dim = p.dim();
    let one = |x: &mut [f64]| -> f64 {
        let mut o = optimizers::by_id(opt_id).expect("optimizer id");
        o.init(x);
        let mut rng = Rng::new(0);
        for _ in 0..iters {
            o.step(p.as_ref(), &mut rng);
        }
        x.copy_from_slice(o.x());
        p.eval(x)
    };
    #[cfg(feature = "parallel")]
    let fs: Vec<f64> = {
        use rayon::prelude::*;
        starts.par_chunks_mut(dim).map(one).collect()
    };
    #[cfg(not(feature = "parallel"))]
    let fs: Vec<f64> = starts.chunks_mut(dim).map(one).collect();
    fs.into_iter().fold(f64::INFINITY, f64::min)
}

/// Uniform random starts inside the problem's bounds.
pub fn random_starts(problem_id: &str, dim: usize, n: usize, seed: u64) -> Vec<f64> {
    let p = problems::by_id(problem_id, dim).expect("problem id");
    let b = p.manifest().bounds;
    let mut rng = Rng::new(seed);
    (0..n).flat_map(|_| b.iter().map(|&(lo, hi)| rng.range(lo, hi)).collect::<Vec<_>>()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn himmelblau_starts_find_all_four_minima() {
        let mut xs = random_starts("himmelblau", 2, 400, 5);
        let best = run_many("himmelblau", "adam", 2, &mut xs, 3000);
        assert!(best < 1e-6, "{best}");
        let minima = problems::by_id("himmelblau", 2).unwrap().manifest().minima;
        for m in &minima {
            let hit = xs.chunks(2).any(|x| (x[0] - m[0]).hypot(x[1] - m[1]) < 1e-2);
            assert!(hit, "no start reached {m:?}");
        }
    }

    #[test]
    fn result_is_independent_of_chunking() {
        let mut a = random_starts("rastrigin", 4, 64, 9);
        let mut b = a.clone();
        let fa = run_many("rastrigin", "momentum", 4, &mut a, 200);
        // the same starts run one at a time
        let mut fb = f64::INFINITY;
        for x in b.chunks_mut(4) {
            fb = fb.min(run_many("rastrigin", "momentum", 4, x, 200));
        }
        assert_eq!(fa.to_bits(), fb.to_bits());
        assert_eq!(a, b);
    }
}
