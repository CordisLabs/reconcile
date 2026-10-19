//! Prints the numbers cited in this slice's README.
//! Run: cargo run -p m06_msi --example report_m06_msi

use m06_msi::{best_fixed_policy, example_environments, oracle_upsilon, upsilon, upsilon_normalised};

fn main() {
    let envs = example_environments();
    let random = upsilon_normalised(&envs, 0.5);
    let (p, best) = best_fixed_policy(&envs);
    let oracle = oracle_upsilon(&envs);

    println!("== Upsilon over {} environments ==", envs.len());
    println!("  random agent (p = 0.5)          {random:.4}");
    println!("  best fixed policy (p = {p:.3})     {best:.4}");
    println!("  oracle (per-environment best)   {oracle:.4}");
    println!();

    println!("== re-encoding: every complexity += c ==");
    println!("  c     random     best-fixed   oracle     unnormalised(random)");
    for extra in [0u32, 1, 3, 8, 16] {
        let shifted: Vec<_> = envs.iter().map(|e| e.reencoded(extra)).collect();
        let r = upsilon_normalised(&shifted, 0.5);
        let (_, b) = best_fixed_policy(&shifted);
        let o = oracle_upsilon(&shifted);
        let u = upsilon(&shifted, 0.5);
        println!("  {extra:>3}   {r:>8.4}   {b:>10.4}   {o:>8.4}   {u:>18.6e}");
    }
}
