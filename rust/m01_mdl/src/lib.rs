//! Keeping Neural Networks Simple by Minimizing the Description Length of the
//! Weights (Hinton & van Camp, 1993) — two-part MDL code for a linear model in a
//! polynomial feature space. Zero dependencies.
//!
//! The paper's claim, stripped to its core: a network is *simple* when its weights
//! can be encoded cheaply, and simplicity should be paid for explicitly in the cost.
//! Here that is made literal. The code has two parts:
//!
//!   data part   = -log2 P(y | w, sigma^2)   = SSE/(2 sigma^2 ln 2) + n/2 log2(2 pi sigma^2)
//!   weight part = -log2 P(w | sigma_w^2)    = |w|^2/(2 sigma_w^2 ln 2) + k/2 log2(2 pi sigma_w^2)
//!
//! with the prior scale tied to the noise scale by `beta = sigma^2 / sigma_w^2`.
//! Sweeping `beta` traces out the MDL trade-off; the total code length is what
//! model selection is supposed to minimise. Whether it actually tracks held-out
//! error is the empirical question this slice answers (see `examples/report.rs`).

#![forbid(unsafe_code)]

pub mod noise;

use std::f64::consts::{LN_2, PI};

/// Deterministic xorshift64* generator: no dependency, same numbers on every
/// platform, which is what makes the reported results reproducible.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        let u = ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
        lo + (hi - lo) * u
    }

    pub fn normal(&mut self) -> f64 {
        let u1 = ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
        let u2 = ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

/// Ground truth: a cubic. A degree-3 model can represent it exactly; a degree-12
/// model can only do better on the training points by fitting the noise.
pub fn target(x: f64) -> f64 {
    1.5 * x - 0.8 * x * x * x
}

pub struct Dataset {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
}

pub fn make_dataset(n: usize, noise_sd: f64, seed: u64) -> Dataset {
    let mut rng = Rng::new(seed);
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let xi = rng.uniform(-1.0, 1.0);
        x.push(xi);
        y.push(target(xi) + noise_sd * rng.normal());
    }
    Dataset { x, y }
}

/// Design matrix with features 1, x, x^2, ..., x^(degree-1). The constant term is
/// regularised too: the MDL prior is on the whole weight vector, bias included.
pub fn design(x: &[f64], degree: usize) -> Vec<Vec<f64>> {
    x.iter()
        .map(|&xi| {
            let mut row = Vec::with_capacity(degree);
            let mut p = 1.0;
            for _ in 0..degree {
                row.push(p);
                p *= xi;
            }
            row
        })
        .collect()
}

/// Solve (X^T X + beta I) w = X^T y by Gaussian elimination with partial pivoting.
pub fn ridge_fit(design: &[Vec<f64>], y: &[f64], beta: f64) -> Vec<f64> {
    let n = design.len();
    let k = design[0].len();
    let mut a = vec![vec![0.0; k]; k];
    let mut b = vec![0.0; k];
    for i in 0..n {
        for r in 0..k {
            b[r] += design[i][r] * y[i];
            for c in 0..k {
                a[r][c] += design[i][r] * design[i][c];
            }
        }
    }
    for r in 0..k {
        a[r][r] += beta;
    }
    solve(a, b)
}

fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for col in 0..n {
        let mut piv = col;
        for r in (col + 1)..n {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        a.swap(col, piv);
        b.swap(col, piv);
        let p = a[col][col];
        assert!(p.abs() > 1e-300, "normal equations are singular");
        for r in (col + 1)..n {
            let f = a[r][col] / p;
            if f != 0.0 {
                for c in col..n {
                    a[r][c] -= f * a[col][c];
                }
                b[r] -= f * b[col];
            }
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= a[i][j] * x[j];
        }
        x[i] = s / a[i][i];
    }
    x
}

pub fn sse(design: &[Vec<f64>], y: &[f64], w: &[f64]) -> f64 {
    design
        .iter()
        .zip(y)
        .map(|(row, &yi)| {
            let pred: f64 = row.iter().zip(w).map(|(f, wi)| f * wi).sum();
            (pred - yi) * (pred - yi)
        })
        .sum()
}

pub fn mse(design: &[Vec<f64>], y: &[f64], w: &[f64]) -> f64 {
    sse(design, y, w) / y.len() as f64
}

#[derive(Clone, Copy, Debug)]
pub struct CodeLen {
    pub data_bits: f64,
    pub weight_bits: f64,
    pub total_bits: f64,
}

/// Two-part code length in bits.
///
/// The paper fixes the weight prior's standard deviation `sigma_w` **in advance**
/// (§2: "if the standard deviation sigma_w of this distribution is fixed in advance,
/// the description length of the weights is simply proportional to the sum of their
/// squares"). Only the assumed noise level `sigma` is swept. Tying `sigma_w` to
/// `sigma` instead is what makes the prior degenerate and the code length unbounded
/// below — see the crate README for the measurement that led to fixing this.
pub fn code_length(sse_train: f64, w: &[f64], n: usize, sigma2: f64, sigma_w2: f64) -> CodeLen {
    let data_bits = (sse_train / (2.0 * sigma2) + 0.5 * n as f64 * (2.0 * PI * sigma2).ln()) / LN_2;
    let w2: f64 = w.iter().map(|v| v * v).sum();
    let weight_bits =
        (w2 / (2.0 * sigma_w2) + 0.5 * w.len() as f64 * (2.0 * PI * sigma_w2).ln()) / LN_2;
    CodeLen {
        data_bits,
        weight_bits,
        total_bits: data_bits + weight_bits,
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SweepRow {
    /// Assumed standard deviation of the target noise, i.e. the swept quantity.
    pub sigma: f64,
    /// Ridge strength implied by the guess: beta = sigma^2 / sigma_w^2.
    pub beta: f64,
    pub train_mse: f64,
    pub test_mse: f64,
    pub code: CodeLen,
    pub w_sq: f64,
}

/// Fit once per assumed noise level and record both the code length and the
/// held-out error. Model selection may only look at the code length; `test_mse` is
/// recorded afterwards, for scoring.
pub fn sweep(
    train: &Dataset,
    test: &Dataset,
    degree: usize,
    sigma_w2: f64,
    sigmas: &[f64],
) -> Vec<SweepRow> {
    let xt = design(&train.x, degree);
    let xs = design(&test.x, degree);
    sigmas
        .iter()
        .map(|&sigma| {
            let sigma2 = sigma * sigma;
            let beta = sigma2 / sigma_w2;
            let w = ridge_fit(&xt, &train.y, beta);
            let train_sse = sse(&xt, &train.y, &w);
            SweepRow {
                sigma,
                beta,
                train_mse: train_sse / train.y.len() as f64,
                test_mse: mse(&xs, &test.y, &w),
                code: code_length(train_sse, &w, train.y.len(), sigma2, sigma_w2),
                w_sq: w.iter().map(|v| v * v).sum(),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIGMA_W2: f64 = 1.0;

    fn sigmas() -> Vec<f64> {
        // Assumed noise level, 0.01 .. 1.0 in quarter decades.
        (0..=32).map(|i| 10f64.powf(-2.0 + i as f64 * 0.0625)).collect()
    }

    fn argmin_code(rows: &[SweepRow]) -> SweepRow {
        *rows
            .iter()
            .min_by(|a, b| a.code.total_bits.partial_cmp(&b.code.total_bits).unwrap())
            .unwrap()
    }

    fn argmin_test(rows: &[SweepRow]) -> SweepRow {
        *rows
            .iter()
            .min_by(|a, b| a.test_mse.partial_cmp(&b.test_mse).unwrap())
            .unwrap()
    }

    fn setup() -> (Dataset, Dataset) {
        (make_dataset(20, 0.1, 42), make_dataset(200, 0.1, 7))
    }

    #[test]
    fn beta_sweep_has_interior_minimum() {
        let (train, test) = setup();
        let rows = sweep(&train, &test, 12, SIGMA_W2, &sigmas());
        let mut argmin = 0;
        for (i, r) in rows.iter().enumerate() {
            if r.code.total_bits < rows[argmin].code.total_bits {
                argmin = i;
            }
        }
        assert!(argmin > 0, "minimum at the smallest sigma: no trade-off found");
        assert!(
            argmin < rows.len() - 1,
            "minimum at the largest sigma: the grid is too narrow"
        );
    }

    /// The measured behaviour, stated without flattering the method.
    ///
    /// The MDL choice systematically picks a *smaller* assumed noise level than the
    /// held-out optimum — it under-regularises — and the price of doing so grows with
    /// the prior scale `sigma_w` (1.02x at 0.3, 1.49x at 2.0). So the claim is not
    /// "MDL finds the best trade-off", it is "MDL stays within 1.5x of it, in the
    /// direction of too little regularisation, as long as the prior is not absurd".
    #[test]
    fn mdl_under_regularises_but_stays_within_range() {
        let (train, test) = setup();
        for sigma_w in [0.3f64, 0.5, 1.0, 2.0] {
            let rows = sweep(&train, &test, 12, sigma_w * sigma_w, &sigmas());
            let by_code = argmin_code(&rows);
            let by_test = argmin_test(&rows);
            let price = by_code.test_mse / by_test.test_mse;
            assert!(
                price < 1.5,
                "sigma_w={sigma_w}: MDL choice costs {price:.3}x of the best test error"
            );
            assert!(
                by_test.sigma > by_code.sigma,
                "sigma_w={sigma_w}: MDL no longer under-regularises \
                 ({} vs {}); re-measure the claim",
                by_code.sigma,
                by_test.sigma
            );
        }
    }

    #[test]
    fn weight_code_grows_with_the_feature_count() {
        let (train, test) = setup();
        let mut prev = f64::NEG_INFINITY;
        for degree in [3usize, 6, 9, 12, 15] {
            let bits = sweep(&train, &test, degree, SIGMA_W2, &[0.1])[0].code.weight_bits;
            assert!(
                bits > prev,
                "weight code did not grow at degree {degree}: {bits} <= {prev}"
            );
            prev = bits;
        }
    }

    #[test]
    fn every_run_reproduces_the_same_numbers() {
        let (train, test) = setup();
        let a = sweep(&train, &test, 12, SIGMA_W2, &sigmas());
        let b = sweep(&train, &test, 12, SIGMA_W2, &sigmas());
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x.code.total_bits.to_bits(), y.code.total_bits.to_bits());
            assert_eq!(x.test_mse.to_bits(), y.test_mse.to_bits());
        }
    }

    /// With the prior fixed in advance, model selection across degrees must prefer a
    /// small model. This is the property the coupled-prior version did not have.
    #[test]
    fn cross_degree_selection_prefers_a_parsimonious_model() {
        let (train, test) = setup();
        let mut per_degree = Vec::new();
        for degree in 1..=15usize {
            let rows = sweep(&train, &test, degree, SIGMA_W2, &sigmas());
            let best = rows
                .iter()
                .min_by(|a, b| a.code.total_bits.partial_cmp(&b.code.total_bits).unwrap())
                .copied()
                .unwrap();
            per_degree.push((degree, best));
        }
        let mdl = per_degree
            .iter()
            .min_by(|a, b| a.1.code.total_bits.partial_cmp(&b.1.code.total_bits).unwrap())
            .unwrap();
        let best = per_degree
            .iter()
            .min_by(|a, b| a.1.test_mse.partial_cmp(&b.1.test_mse).unwrap())
            .unwrap();
        assert!(
            mdl.0 <= 8,
            "selected degree {}: the code length is rewarding complexity again",
            mdl.0
        );
        let price = mdl.1.test_mse / best.1.test_mse;
        assert!(
            price < 1.5,
            "selected degree {} costs {price:.3}x of degree {}",
            mdl.0,
            best.0
        );
    }
}
