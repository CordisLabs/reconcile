//! Prints the numbers cited in the "noisy weights" half of this slice's README.
//! Run: cargo run -p m01_mdl --example report_noise

use m01_mdl::noise::{Config, NoisyModel, Quadrature};

fn main() {
    let q = Quadrature::new(256);
    let cfg = Config {
        sigma_p2: 1.0,
        sigma_d2: 0.01,
    };

    // A small noisy-data problem: two inputs, two tanh units, linear output.
    let mut x = Vec::new();
    let mut t = Vec::new();
    for i in 0..10 {
        let a = i as f64 * 0.2 - 1.0;
        let b = (i as f64 * 0.37).cos();
        x.push(vec![a, b]);
        t.push(0.8 * (1.2 * a - 0.8 * b).tanh() + 0.05 * (i as f64 * 1.7).sin());
    }

    let mut model = NoisyModel::new(2, 2, 5);
    let trace = model.fit(&x, &t, &q, &cfg, 4000, 2e-3);
    let first = trace.first().copied().unwrap();
    let last = trace.last().copied().unwrap();

    println!("== training on the total description length ==");
    println!("  cost first step : {first:.4}");
    println!("  cost last step  : {last:.4}");
    println!();

    println!("== learned weight noise (sigma = exp(rho)) ==");
    let mut lo = f64::INFINITY;
    let mut hi: f64 = 0.0;
    for h in 0..model.hidden {
        print!("  unit {h}:");
        for d in 0..model.inputs {
            let s = model.sigma(h, d);
            lo = lo.min(s);
            hi = hi.max(s);
            print!("  s[{h}][{d}] = {s:.4}");
        }
        println!();
    }
    println!("  range: {lo:.4} .. {hi:.4}  (strictly interior, not at a bound)");
    println!("  output weights v = {:?}", model.v.iter().map(|v| (v*1000.0).round()/1000.0).collect::<Vec<_>>());
    println!("  A unit with v = 0 contributes nothing, so its KL term is the only");
    println!("  cost left, and KL(Q||P) is minimised at Q = P, i.e. s = sigma_p = 1.");
    println!();

    // Weight code length against data cost at the learned point.
    let total = model.cost(&x, &t, &q, &cfg);
    let data = model.expected_error(&x, &t, &q) / (2.0 * cfg.sigma_d2);
    println!("== split of the total description length (nats) ==");
    println!("  weights (KL to prior) : {:.4}", total - data);
    println!("  data mis-fit          : {data:.4}");
    println!("  total                 : {total:.4}");
    println!();

    // Exact expectations versus Monte Carlo.
    println!("== exact expectation vs Monte Carlo ==");
    let exact = model.expected_error(&x, &t, &q);
    for samples in [10_000usize, 100_000, 1_000_000] {
        let (mc, se) = model.monte_carlo_error(&x, &t, samples, 11);
        println!(
            "  {samples:>9} samples: MC {mc:.6} +- {se:.6}, exact {exact:.6}, |diff|/se = {:.2}",
            (mc - exact).abs() / se
        );
    }
    println!();

    // The trade-off the paper describes, measured: scale every learned sigma.
    println!("== total cost when every learned sigma is scaled (k = 1 is the fit) ==");
    println!("  k        min sigma   max sigma   total cost");
    for k in [0.1f64, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0] {
        let mut m = model.clone();
        for h in 0..m.hidden {
            for d in 0..m.inputs {
                m.rho[h][d] += k.ln();
            }
        }
        let s_lo = (0..m.hidden)
            .flat_map(|h| (0..m.inputs).map(move |d| (h, d)))
            .map(|(h, d)| m.sigma(h, d))
            .fold(f64::INFINITY, f64::min);
        let s_hi = (0..m.hidden)
            .flat_map(|h| (0..m.inputs).map(move |d| (h, d)))
            .map(|(h, d)| m.sigma(h, d))
            .fold(0.0, f64::max);
        println!(
            "  {:>5.2}   {:>9.4}   {:>9.4}   {:>10.4}",
            k,
            s_lo,
            s_hi,
            m.cost(&x, &t, &q, &cfg)
        );
    }
}
