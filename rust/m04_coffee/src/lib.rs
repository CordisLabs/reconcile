//! Quantifying the Rise and Fall of Complexity in Closed Systems: The Coffee Automaton
//! (Aaronson, Carroll, Ouellette, 2014).
//!
//! The model is the paper's **interacting** coffee cup: a grid where each cell holds one
//! particle, one fluid at the top and the other at the bottom. Each step picks a
//! horizontally or vertically adjacent pair of *differing* particles and swaps them, so
//! the number of each fluid is conserved.
//!
//! Two quantities are measured exactly as the paper defines them (§4-§5):
//!
//!   * entropy estimate — the compressed size of the **fine-grained** array;
//!   * apparent complexity — the compressed size of the **coarse-grained** array, where
//!     each cell is replaced by the average of the `g x g` square centred on it, the
//!     averages are thresholded into three buckets, and the result is compressed.
//!
//! The coarse array has the same dimensions as the fine one; only the values are smoothed.
//! Compression is a small LZ77 written for this purpose, standing in for gzip.

#![forbid(unsafe_code)]

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    pub fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
}

#[derive(Clone)]
pub struct Grid {
    pub n: usize,
    pub cells: Vec<u8>,
}

impl Grid {
    /// Top half cream, bottom half coffee — the cup before stirring.
    pub fn layered(n: usize) -> Self {
        let mut cells = vec![0u8; n * n];
        for y in 0..n / 2 {
            for x in 0..n {
                cells[y * n + x] = 1;
            }
        }
        Grid { n, cells }
    }

    pub fn at(&self, x: usize, y: usize) -> u8 {
        self.cells[y * self.n + x]
    }

    fn set(&mut self, x: usize, y: usize, v: u8) {
        self.cells[y * self.n + x] = v;
    }

    /// One step of the interacting model: swap an adjacent pair of differing particles.
    pub fn step(&mut self, rng: &mut Rng) {
        let x = rng.below(self.n);
        let y = rng.below(self.n);
        let (x2, y2) = if rng.unit() < 0.5 {
            ((x + 1) % self.n, y)
        } else {
            (x, (y + 1) % self.n)
        };
        let a = self.at(x, y);
        let b = self.at(x2, y2);
        if a != b {
            self.set(x, y, b);
            self.set(x2, y2, a);
        }
    }

    pub fn dark_fraction(&self) -> f64 {
        self.cells.iter().map(|&c| c as f64).sum::<f64>() / self.cells.len() as f64
    }
}

/// The coarse-grained array: each cell is the mean of the `g x g` square centred on it,
/// thresholded into three buckets (mostly coffee / mixed / mostly cream).
pub fn coarse_grained(grid: &Grid, g: usize) -> Vec<u8> {
    let n = grid.n;
    let r = g / 2;
    let mut out = Vec::with_capacity(n * n);
    for y in 0..n {
        for x in 0..n {
            let mut sum = 0u32;
            let mut count = 0u32;
            for dy in -(r as isize)..=(r as isize) {
                let yy = y as isize + dy;
                if yy < 0 || yy >= n as isize {
                    continue;
                }
                for dx in -(r as isize)..=(r as isize) {
                    let xx = x as isize + dx;
                    if xx < 0 || xx >= n as isize {
                        continue;
                    }
                    sum += grid.at(xx as usize, yy as usize) as u32;
                    count += 1;
                }
            }
            let mean = sum as f64 / count as f64;
            out.push(if mean < 0.4 {
                0
            } else if mean > 0.6 {
                2
            } else {
                1
            });
        }
    }
    out
}

/// Greedy LZ77 length in bytes; a stand-in for gzip as the paper uses it.
pub fn compressed_len(data: &[u8]) -> usize {
    let n = data.len();
    let window = 256usize;
    let max_len = 255usize;
    let mut out = 0usize;
    let mut i = 0usize;
    while i < n {
        let start = i.saturating_sub(window);
        let mut best = 0usize;
        let max_match = max_len.min(n - i);
        for j in start..i {
            let mut l = 0usize;
            while l < max_match && j + l < i && data[j + l] == data[i + l] {
                l += 1;
            }
            if l > best {
                best = l;
            }
        }
        if best >= 3 {
            out += 3;
            i += best;
        } else {
            out += 1;
            i += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Curve {
        steps: usize,
        entropy: Vec<usize>,
        complexity: Vec<usize>,
        dark: Vec<f64>,
    }

    fn run(n: usize, g: usize, total: usize, samples: usize, seed: u64) -> Curve {
        let mut grid = Grid::layered(n);
        let mut rng = Rng::new(seed);
        let mut t = 0usize;
        let mut entropy = Vec::new();
        let mut complexity = Vec::new();
        let mut dark = Vec::new();
        for k in 0..=samples {
            let target = total * k / samples;
            while t < target {
                grid.step(&mut rng);
                t += 1;
            }
            entropy.push(compressed_len(&grid.cells));
            complexity.push(compressed_len(&coarse_grained(&grid, g)));
            dark.push(grid.dark_fraction());
        }
        Curve { steps: total, entropy, complexity, dark }
    }

    #[test]
    fn the_number_of_particles_is_conserved() {
        let c = run(32, 6, 200_000, 10, 1);
        for d in &c.dark {
            assert!((d - 0.5).abs() < 1e-12, "dark fraction drifted to {d}");
        }
    }

    /// Second law of the toy model: the fine-grained description does not get simpler.
    #[test]
    fn entropy_rises_and_stays_up() {
        let c = run(64, 12, 6_000_000, 25, 42);
        let first = c.entropy[0] as f64;
        let last = *c.entropy.last().unwrap() as f64;
        assert!(last > 10.0 * first, "entropy went {first} -> {last}");
        let max_drop: f64 = c
            .entropy
            .windows(2)
            .map(|w| (w[0] as f64 - w[1] as f64) / w[0] as f64)
            .fold(0.0, f64::max);
        assert!(
            max_drop < 0.05,
            "largest single drop was {:.1}% of the previous value (estimator noise expected)",
            max_drop * 100.0
        );
    }

    /// The paper's headline: apparent complexity rises and then falls.
    #[test]
    fn apparent_complexity_rises_then_falls() {
        let c = run(64, 12, 6_000_000, 25, 42);
        let peak_idx = c
            .complexity
            .iter()
            .enumerate()
            .max_by_key(|(_, &v)| v)
            .map(|(i, _)| i)
            .unwrap();
        let peak = c.complexity[peak_idx];
        let first = c.complexity[0];
        let last = *c.complexity.last().unwrap();
        assert!(
            peak_idx > 0 && peak_idx < c.complexity.len() - 1,
            "peak at an endpoint ({peak_idx} of {})",
            c.complexity.len() - 1
        );
        assert!(peak > 3 * first, "peak {peak} vs start {first}");
        assert!(peak > 3 * last, "peak {peak} vs end {last}");
        let _ = c.steps;
    }

    #[test]
    fn the_curve_is_deterministic() {
        let a = run(32, 6, 100_000, 8, 7);
        let b = run(32, 6, 100_000, 8, 7);
        assert_eq!(a.entropy, b.entropy);
        assert_eq!(a.complexity, b.complexity);
    }
}

/// Order-2 context model length in bits: sum over contexts of the conditional entropy.
///
/// A second, structurally different compressor. The paper used gzip; this slice ships a
/// hand-written LZ77, so the question "is the curve an artefact of the compressor" needs a
/// second opinion from an algorithm that looks at the data differently.
pub fn order2_bits(data: &[u8]) -> f64 {
    use std::collections::HashMap;
    let mut ctx: HashMap<(u8, u8), HashMap<u8, u32>> = HashMap::new();
    for i in 2..data.len() {
        let key = (data[i - 2], data[i - 1]);
        *ctx.entry(key).or_default().entry(data[i]).or_insert(0) += 1;
    }
    let mut bits = 0.0;
    for (_k, counts) in ctx.iter() {
        let total: u32 = counts.values().sum();
        let n = total as f64;
        for &c in counts.values() {
            let p = c as f64 / n;
            bits -= (c as f64) * p.log2();
        }
    }
    bits
}

#[cfg(test)]
mod compressor_tests {
    use super::*;

    #[test]
    fn order2_beats_order0_on_structured_data() {
        // a repeating pattern has no order-0 structure but strong order-2 structure
        let pattern: Vec<u8> = (0..2000).map(|i| (i % 4) as u8).collect();
        let o2 = order2_bits(&pattern);
        let o0 = {
            let mut counts = [0u32; 256];
            for &b in &pattern {
                counts[b as usize] += 1;
            }
            let n = pattern.len() as f64;
            counts
                .iter()
                .filter(|&&c| c > 0)
                .map(|&c| -((c as f64) * ((c as f64 / n).log2())))
                .sum::<f64>()
        };
        assert!(o2 < o0, "order-2 {o2} should beat order-0 {o0} on a periodic sequence");
    }

    #[test]
    fn order2_of_random_data_is_near_uniform() {
        let mut rng = Rng::new(7);
        let data: Vec<u8> = (0..4000).map(|_| rng.below(4) as u8).collect();
        let bits = order2_bits(&data);
        let per_symbol = bits / data.len() as f64;
        assert!(per_symbol > 1.5, "random data over 4 symbols should cost ~2 bits each, got {per_symbol}");
    }
}
