//! xoshiro256++ seeded through splitmix64: small, fast and good. One `Rng` per run; parallel work takes
//! independent streams with `jump`, so results never depend on thread scheduling.

#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
}

fn splitmix64(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut x = seed;
        Self { s: [splitmix64(&mut x), splitmix64(&mut x), splitmix64(&mut x), splitmix64(&mut x)] }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let r = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        r
    }

    /// Uniform in [0, 1)
    #[inline]
    pub fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in [lo, hi)
    #[inline]
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.f64()
    }

    /// Standard normal (Box-Muller; one of the pair is discarded for simplicity)
    pub fn normal(&mut self) -> f64 {
        let u = 1.0 - self.f64();
        let v = self.f64();
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }

    /// Advance 2^128 steps: a new, non-overlapping stream.
    pub fn jump(&mut self) {
        const J: [u64; 4] =
            [0x180e_c6d3_3cfd_0aba, 0xd5a6_1266_f0c9_392c, 0xa958_2618_e03f_c9aa, 0x39ab_dc45_29b1_661c];
        let mut t = [0u64; 4];
        for j in J {
            for b in 0..64 {
                if j & (1 << b) != 0 {
                    for (ti, si) in t.iter_mut().zip(self.s) {
                        *ti ^= si;
                    }
                }
                self.next_u64();
            }
        }
        self.s = t;
    }

    /// The `i`-th independent stream derived from this generator (i jumps ahead), without changing self.
    pub fn stream(&self, i: usize) -> Rng {
        let mut r = self.clone();
        for _ in 0..=i {
            r.jump();
        }
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_sequence() {
        let (mut a, mut b) = (Rng::new(7), Rng::new(7));
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn uniform_mean_and_normal_moments() {
        let mut r = Rng::new(1);
        let n = 200_000;
        let mean = (0..n).map(|_| r.f64()).sum::<f64>() / n as f64;
        assert!((mean - 0.5).abs() < 0.005, "{mean}");
        let xs: Vec<f64> = (0..n).map(|_| r.normal()).collect();
        let m = xs.iter().sum::<f64>() / n as f64;
        let v = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n as f64;
        assert!(m.abs() < 0.01 && (v - 1.0).abs() < 0.02, "{m} {v}");
    }

    #[test]
    fn streams_differ() {
        let r = Rng::new(3);
        assert_ne!(r.stream(0).next_u64(), r.stream(1).next_u64());
    }
}
