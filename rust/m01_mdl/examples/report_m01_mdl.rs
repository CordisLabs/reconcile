//! Prints the numbers cited in this slice's README.
//! Run: cargo run -p m01_mdl --example report_m01_mdl

use m01_mdl::{make_dataset, sweep, SweepRow};

/// The weight prior is fixed in advance, as the paper requires. sigma_w = 1.0 is a
/// statement about the scale the weights are expected to live on (the target here is
/// O(1)); it is *not* tuned per model.
const SIGMA_W2: f64 = 1.0;

fn fmt(x: f64) -> String {
    if x.abs() >= 1000.0 || (x != 0.0 && x.abs() < 1e-3) {
        format!("{x:.3e}")
    } else {
        format!("{x:.4}")
    }
}

/// Assumed noise level, 0.01 .. 1.0 in quarter decades.
fn sigma_grid() -> Vec<f64> {
    (0..=32).map(|i| 10f64.powf(-2.0 + i as f64 * 0.0625)).collect()
}

fn main() {
    // Ground truth is a cubic with noise sd 0.1. `test` is a larger held-out draw.
    let train = make_dataset(20, 0.1, 42);
    let test = make_dataset(200, 0.1, 7);
    let sigmas = sigma_grid();

    // ---- 1. the sweep at a deliberately over-parameterised degree ----
    let rows = sweep(&train, &test, 12, SIGMA_W2, &sigmas);
    let min_total = rows
        .iter()
        .map(|r| r.code.total_bits)
        .fold(f64::INFINITY, f64::min);

    println!("== degree 12 (12 weights, 20 train points, sigma_w = 1) ==");
    println!("  absolute code lengths are differential: their zero point is arbitrary,");
    println!("  so the useful columns are `regret` (bits over the best choice) and test MSE.");
    println!();
    println!("  sigma       beta        train MSE     test MSE      data bits   weight bits   regret");
    for row in rows.iter().step_by(3) {
        print_row(row, min_total);
    }
    println!();

    let by_code = argmin_code(&rows);
    let by_test = argmin_test(&rows);
    println!(
        "  MDL picks     sigma = {:>8} (beta {:>8})  test MSE = {}",
        fmt(by_code.sigma),
        fmt(by_code.beta),
        fmt(by_code.test_mse)
    );
    println!(
        "  best possible sigma = {:>8} (beta {:>8})  test MSE = {}",
        fmt(by_test.sigma),
        fmt(by_test.beta),
        fmt(by_test.test_mse)
    );
    println!(
        "  price of using the MDL choice = {:.3}x",
        by_code.test_mse / by_test.test_mse
    );
    println!();

    // ---- 2. model selection across degrees, using the code length only ----
    println!("== choosing the degree too (total code length, train data only) ==");
    println!("  degree   best sigma   total bits    test MSE   regret (bits)");
    let mut per_degree: Vec<(usize, SweepRow)> = Vec::new();
    for degree in 1..=15usize {
        let r = sweep(&train, &test, degree, SIGMA_W2, &sigmas);
        per_degree.push((degree, argmin_code(&r)));
    }
    let global_min = per_degree
        .iter()
        .map(|(_, r)| r.code.total_bits)
        .fold(f64::INFINITY, f64::min);
    for (degree, r) in per_degree.iter() {
        println!(
            "  {:>5}   {:>10}   {:>10}   {:>10}   {:>10}",
            degree,
            fmt(r.sigma),
            fmt(r.code.total_bits),
            fmt(r.test_mse),
            fmt(r.code.total_bits - global_min)
        );
    }
    let mdl_pick = per_degree
        .iter()
        .min_by(|a, b| a.1.code.total_bits.partial_cmp(&b.1.code.total_bits).unwrap())
        .unwrap();
    let best_test = per_degree
        .iter()
        .min_by(|a, b| a.1.test_mse.partial_cmp(&b.1.test_mse).unwrap())
        .unwrap();
    println!();
    println!(
        "  MDL-selected: degree {} (test MSE {})",
        mdl_pick.0,
        fmt(mdl_pick.1.test_mse)
    );
    println!(
        "  best on test: degree {} (test MSE {})",
        best_test.0,
        fmt(best_test.1.test_mse)
    );
    println!(
        "  price of the MDL selection = {:.3}x",
        mdl_pick.1.test_mse / best_test.1.test_mse
    );
    println!();

    // ---- 2b. how much does the answer depend on the prior scale? ----
    // sigma_w is fixed in advance, but the choice is ours. Reporting the sweep is
    // the only honest way to know whether the conclusion survives it.
    println!("== sensitivity to the prior scale (degree 12) ==");
    println!("  sigma_w   MDL sigma   best sigma   price of the MDL choice");
    for sigma_w in [0.3f64, 0.5, 1.0, 2.0] {
        let r = sweep(&train, &test, 12, sigma_w * sigma_w, &sigmas);
        let c = argmin_code(&r);
        let t = argmin_test(&r);
        println!(
            "  {:>7}   {:>9}   {:>10}   {:>9.3}x",
            fmt(sigma_w),
            fmt(c.sigma),
            fmt(t.sigma),
            c.test_mse / t.test_mse
        );
    }
    println!();

    // ---- 3. every extra weight has to be paid for ----
    println!("== weight code length grows with the feature count (sigma = 0.1) ==");
    for degree in [3usize, 6, 9, 12, 15] {
        let r = sweep(&train, &test, degree, SIGMA_W2, &[0.1])[0];
        println!(
            "  degree {:>2}: weight bits = {:>9}, test MSE = {}",
            degree,
            fmt(r.code.weight_bits),
            fmt(r.test_mse)
        );
    }
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

fn print_row(row: &SweepRow, min_total: f64) {
    println!(
        "  {:>9}  {:>9}  {:>11}  {:>11}  {:>11}  {:>12}  {:>9}",
        fmt(row.sigma),
        fmt(row.beta),
        fmt(row.train_mse),
        fmt(row.test_mse),
        fmt(row.code.data_bits),
        fmt(row.code.weight_bits),
        fmt(row.code.total_bits - min_total)
    );
}
