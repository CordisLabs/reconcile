//! A Tutorial Introduction to the Minimum Description Length Principle (Grünwald, 2004).
//!
//! The tutorial's central worked example is the **normalized maximum likelihood (NML)**
//! code for a Bernoulli source. For a sequence of n bits with k ones the maximum
//! likelihood is (k/n)^k ((n-k)/n)^(n-k), and the NML probability divides by
//! C(n) = sum over all sequences of that likelihood. The regret is log C(n):
//!
//! ```text
//! R(n) = log2 sum_{k=0}^{n} C(n,k) (k/n)^k ((n-k)/n)^(n-k)
//! ```
//!
//! Three claims from the tutorial are checked here, and all three are arithmetic:
//!   * R(n) > 0 (no free lunch: the MLE code is not a probability distribution),
//!   * R(n) = 0.5 log2 n + O(1) — the parametric complexity of one free parameter,
//!   * the NML code is a probability distribution: sum over sequences of 2^-L(x) = 1.
//!
//! Everything is computed with a log-sum-exp over the k's, so n can be large without
//! overflowing.

#![forbid(unsafe_code)]

fn log_add_exp(a: f64, b: f64) -> f64 {
    let m = a.max(b);
    if m.is_infinite() {
        return m;
    }
    m + ((a - m).exp() + (b - m).exp()).ln()
}

/// log C(n,k), built incrementally so no factorial ever overflows.
pub fn log_binom(n: usize, k: usize) -> f64 {
    let k = k.min(n - k);
    (1..=k)
        .map(|i| ((n - k + i) as f64 / i as f64).ln())
        .sum()
}

/// log of the NML normaliser: log sum_k C(n,k) p_hat^k (1-p_hat)^(n-k).
pub fn log_nml_normaliser(n: usize) -> f64 {
    // log C(n,k) is built by the recurrence log C(n,k) = log C(n,k-1) + ln((n-k+1)/k),
    // so the whole sum is O(n) rather than O(n^2).
    let mut acc = f64::NEG_INFINITY;
    let mut log_binom_k = 0.0; // log C(n,0)
    for k in 0..=n {
        let p = k as f64 / n as f64;
        let term = if k == 0 || k == n {
            // 0 log 0 is taken as 0, so both endpoints contribute exactly 1
            0.0
        } else {
            log_binom_k + k as f64 * p.ln() + (n - k) as f64 * (1.0 - p).ln()
        };
        acc = log_add_exp(acc, term);
        if k < n {
            log_binom_k += ((n - k) as f64 / (k + 1) as f64).ln();
        }
    }
    acc
}

/// NML regret in bits.
pub fn nml_regret_bits(n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    log_nml_normaliser(n) / std::f64::consts::LN_2
}

/// NML code length in bits for a sequence with k ones out of n.
pub fn nml_code_length_bits(n: usize, k: usize) -> f64 {
    let p = k as f64 / n as f64;
    let neg_log_lik = if k == 0 || k == n {
        0.0
    } else {
        -(k as f64 * p.ln() + (n - k) as f64 * (1.0 - p).ln())
    };
    (neg_log_lik + log_nml_normaliser(n)) / std::f64::consts::LN_2
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binom(n: usize, k: usize) -> f64 {
        // exact enough for the small n used in the Kraft check
        (1..=k).map(|i| (n - k + i) as f64 / i as f64).product()
    }

    #[test]
    fn regret_is_positive() {
        // n = 1 has two possible sequences and the MLE gives each probability 1, so the
        // normaliser is 2 and the regret is exactly one bit
        assert!((nml_regret_bits(1) - 1.0).abs() < 1e-12);
        for n in [2usize, 5, 10, 100, 1000] {
            assert!(nml_regret_bits(n) > 0.0, "n = {n}");
        }
    }

    /// R(n) = 0.5 log2 n + O(1): the slope is the parametric complexity, one parameter.
    #[test]
    fn regret_grows_like_half_log_n() {
        let ns: Vec<usize> = (1..=40).map(|i| 100 * i).collect();
        let xs: Vec<f64> = ns.iter().map(|&n| (n as f64).log2()).collect();
        let ys: Vec<f64> = ns.iter().map(|&n| nml_regret_bits(n)).collect();
        let mean_x = xs.iter().sum::<f64>() / xs.len() as f64;
        let mean_y = ys.iter().sum::<f64>() / ys.len() as f64;
        let num: f64 = xs.iter().zip(&ys).map(|(x, y)| (x - mean_x) * (y - mean_y)).sum();
        let den: f64 = xs.iter().map(|x| (x - mean_x) * (x - mean_x)).sum();
        let slope = num / den;
        assert!(
            (slope - 0.5).abs() < 0.02,
            "fitted slope = {slope:.4}, expected 0.5"
        );
    }

    /// The NML code is a probability distribution: Kraft sums to exactly 1.
    #[test]
    fn nml_code_satisfies_kraft() {
        let n = 12usize;
        let mut total = 0.0f64;
        for k in 0..=n {
            let count = binom(n, k);
            let bits = nml_code_length_bits(n, k);
            total += count * 2f64.powf(-bits);
        }
        assert!((total - 1.0).abs() < 1e-9, "Kraft sum = {total}");
    }

    #[test]
    fn offset_from_half_log_n_is_bounded() {
        for n in [10usize, 100, 1000, 10_000] {
            let offset = nml_regret_bits(n) - 0.5 * (n as f64).log2();
            assert!(offset.abs() < 2.0, "n = {n}: offset = {offset:.3} bits");
        }
    }
}

#[cfg(test)]
mod asymptotics {
    use super::*;

    /// The O(1) in R(n) = 0.5 log2 n + O(1) is a known constant for this model:
    /// 0.5 log2(pi/2) ~= 0.3258. At n = 100_000 the measured offset is within 0.01.
    #[test]
    fn offset_converges_to_half_log2_pi_over_two() {
        let expected = 0.5 * (std::f64::consts::PI / 2.0).log2();
        let offset = nml_regret_bits(100_000) - 0.5 * (100_000f64).log2();
        assert!(
            (offset - expected).abs() < 0.01,
            "offset {offset:.4} vs 0.5 log2(pi/2) = {expected:.4}"
        );
    }
}
