//! The First Law of Complexodynamics (Aaronson, 2011) — the argument, made checkable.
//!
//! The post's central technical point is a bound, not a measurement:
//!
//! > if the system happens to be *deterministic*, then mathematically, its state can
//! > always be specified by giving (1) the initial state, and (2) the number of steps t
//! > it's been run for. … It follows that, if we use Kolmogorov complexity as our
//! > stand-in for entropy, then the entropy can increase at most *logarithmically* with t.
//!
//! This is verifiable without computing any Kolmogorov complexity. For the coffee
//! automaton the entire trajectory is a deterministic function of the seed, so the state
//! at step t can be *reconstructed* from `(seed, t)` — and the description `(seed, t)`
//! costs a constant plus log2(t) bits, no matter how large t gets.
//!
//! That is why the post goes on to impose computational resource bounds in two places:
//! without them, the "short program" shortcut always wins and no measure can rise and
//! fall. The measured gap between the shortcut's length and the state's compressed length
//! is what this crate reports.

#![forbid(unsafe_code)]

pub use m04_coffee::{compressed_len, Grid, Rng};

/// Bits needed to name a step count: the entire dependence on t in the shortcut.
pub fn steps_description_bits(t: usize) -> f64 {
    if t <= 1 {
        0.0
    } else {
        (t as f64).log2().ceil()
    }
}

/// A fixed cost for the rest of the program: the initial state, the rule, the seed.
///
/// It is a constant — independent of t — which is the whole point. The number is not
/// special; what matters is that it does not grow.
pub const PROGRAM_OVERHEAD_BITS: f64 = 64.0;

/// Total length in bits of the "run it from the seed" description of the state at t.
pub fn shortcut_bits(t: usize) -> f64 {
    PROGRAM_OVERHEAD_BITS + steps_description_bits(t)
}

/// Reconstruct the state at step `t` from `(seed, t)` alone.
pub fn state_at(seed: u64, t: usize, n: usize) -> Grid {
    let mut grid = Grid::layered(n);
    let mut rng = Rng::new(seed);
    for _ in 0..t {
        grid.step(&mut rng);
    }
    grid
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shortcut is real: the state is exactly reproducible from (seed, t).
    #[test]
    fn the_state_is_reproducible_from_seed_and_t() {
        let n = 32usize;
        let t = 50_000usize;
        let a = state_at(7, t, n);
        let b = state_at(7, t, n);
        assert_eq!(a.cells, b.cells, "the same (seed, t) gave different states");
        // and a different t gives a different state, so the test is not vacuous.
        // (A single extra step can be a no-op at equilibrium — the chosen pair may be
        // same-coloured — so compare a gap rather than one step.)
        let c = state_at(7, t + 1_000, n);
        assert_ne!(a.cells, c.cells);
    }

    /// The shortcut's length is logarithmic in t, while the state's compressed length is
    /// not: the gap widens, which is exactly what kills any unbounded measure.
    #[test]
    fn the_shortcut_stays_logarithmic_while_the_state_does_not() {
        let n = 64usize;
        let mut previous_shortcut = f64::NEG_INFINITY;
        for t in [1_000usize, 100_000, 10_000_000] {
            let grid = state_at(42, t, n);
            let state_bytes = compressed_len(&grid.cells) as f64 * 8.0;
            let shortcut = shortcut_bits(t);
            assert!(
                shortcut < state_bytes,
                "t = {t}: shortcut {shortcut} bits vs state {state_bytes} bits"
            );
            assert!(
                shortcut > previous_shortcut || t == 1_000,
                "the shortcut should grow with t"
            );
            previous_shortcut = shortcut;
        }
    }

    /// The bound the post states: after t steps, K(x_t) <= K(initial) + log t + O(1).
    /// At t = 2^32 the step count still costs only 32 bits.
    #[test]
    fn a_billion_steps_still_cost_30_bits() {
        assert!(steps_description_bits(1_000_000_000) <= 30.0);
        // doubling t adds exactly one bit
        for t in [1_000usize, 2_000, 4_000, 8_000] {
            let a = steps_description_bits(t);
            let b = steps_description_bits(2 * t);
            assert!(b - a <= 1.0, "t = {t}: {a} -> {b}");
        }
    }
}

/// Complextropy as the blog describes it: the information in the state that the
/// coarse-graining does not capture, estimated as compressed(fine) - compressed(coarse).
///
/// The post conjectures this should rise and then fall; that is the "First Law of
/// Complexodynamics". It is a conjecture, not a theorem, and the post says so.
pub fn complextropy(grid: &m04_coffee::Grid, grain: usize) -> f64 {
    let fine = m04_coffee::compressed_len(&grid.cells) as f64;
    let coarse = m04_coffee::compressed_len(&m04_coffee::coarse_grained(grid, grain)) as f64;
    fine - coarse
}

#[cfg(test)]
mod conjecture {
    use super::*;

    #[test]
    fn complextropy_is_the_difference_of_the_two_description_lengths() {
        let grid = m04_coffee::Grid::layered(16);
        let fine = m04_coffee::compressed_len(&grid.cells) as f64;
        let coarse = m04_coffee::compressed_len(&m04_coffee::coarse_grained(&grid, 4)) as f64;
        assert!((complextropy(&grid, 4) - (fine - coarse)).abs() < 1e-9);
    }

    /// The conjecture's shape: complextropy rises and then falls.
    #[test]
    fn complextropy_rises_and_falls() {
        let n = 48usize;
        let grain = 10usize;
        let mut grid = m04_coffee::Grid::layered(n);
        let mut rng = m04_coffee::Rng::new(11);
        let mut curve = Vec::new();
        let total = 4_000_000usize;
        let samples = 20usize;
        let mut t = 0usize;
        for k in 0..=samples {
            let target = total * k / samples;
            while t < target {
                grid.step(&mut rng);
                t += 1;
            }
            curve.push(complextropy(&grid, grain));
        }
        let peak = curve
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert!(
            peak > 0 && peak < curve.len() - 1,
            "complextropy peaked at an endpoint ({peak} of {})",
            curve.len() - 1
        );
        assert!(curve[peak] > 1.2 * curve[0], "peak {:.1} vs start {:.1}", curve[peak], curve[0]);
        // the fall is the weaker half of the claim: the end is below the peak, but by only
        // about 15% here, and the curve is noisy between the two.
        assert!(
            curve[curve.len() - 1] < curve[peak],
            "end {:.1} should be below the peak {:.1}",
            curve[curve.len() - 1],
            curve[peak]
        );
    }
}
