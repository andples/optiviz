//! Native benchmarks, printed as JSON. The workloads are the very functions the browser bench page calls
//! (crates/wasm), so `docs/perf` can compare native, WASM single-threaded and WASM threaded directly.

use std::time::Instant;

/// Best of `reps` timings of `f`, in seconds.
fn best<T>(reps: usize, mut f: impl FnMut() -> T) -> f64 {
    (0..reps)
        .map(|_| {
            let t = Instant::now();
            std::hint::black_box(f());
            t.elapsed().as_secs_f64()
        })
        .fold(f64::INFINITY, f64::min)
}

fn main() {
    let gd_iters = 10_000_000u32;
    let gd = best(5, || optiviz_wasm::bench_gd_rosenbrock(gd_iters));
    let (n, dim, iters) = (4096usize, 10usize, 500usize);
    let threads = rayon::current_num_threads();
    let many_par = best(5, || optiviz_wasm::bench_many(n, dim, iters));
    let pool1 = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let many_1 = pool1.install(|| best(3, || optiviz_wasm::bench_many(n, dim, iters)));
    let out = serde_json::json!({
        "gd_rosenbrock_2d": { "iters": gd_iters, "seconds": gd, "iters_per_s": gd_iters as f64 / gd },
        "adam_many_rastrigin": {
            "starts": n, "dim": dim, "iters": iters, "threads": threads,
            "seconds_1_thread": many_1, "seconds_all_threads": many_par, "speedup": many_1 / many_par,
            "start_iters_per_s_all_threads": (n * iters) as f64 / many_par,
        },
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
