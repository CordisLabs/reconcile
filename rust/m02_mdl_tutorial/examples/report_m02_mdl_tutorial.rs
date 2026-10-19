//! Prints the NML regret numbers cited in this slice's README.
//! Run: cargo run -p m02_mdl_tutorial --example report_m02_mdl_tutorial

use m02_mdl_tutorial::{nml_code_length_bits, nml_regret_bits};

fn main() {
    println!("== NML regret for a Bernoulli source (bits) ==");
    println!("  n        R(n)     R(n) - 0.5 log2 n");
    for n in [1usize, 2, 5, 10, 100, 1_000, 10_000, 100_000] {
        let r = nml_regret_bits(n);
        let offset = r - 0.5 * (n as f64).log2();
        println!("  {n:>7}   {r:>7.4}   {offset:>+8.4}");
    }

    println!();
    println!("== NML code lengths for n = 10 (bits) ==");
    for k in [0usize, 2, 5, 8, 10] {
        println!("  k = {k:>2}: {:.4}", nml_code_length_bits(10, k));
    }

    println!();
    println!("== Kraft sum over all 2^12 sequences ==");
    let n = 12usize;
    let mut total = 0.0f64;
    for k in 0..=n {
        let count: f64 = (1..=k).map(|i| (n - k + i) as f64 / i as f64).product();
        total += count * 2f64.powf(-nml_code_length_bits(n, k));
    }
    println!("  sum of 2^-L = {total:.15}");
}
