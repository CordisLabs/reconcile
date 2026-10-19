//! Prints the numbers cited in this slice's README.
//! Run: cargo run -p m03_complexodynamics --example report_m03_complexodynamics

use m03_complexodynamics::{compressed_len, shortcut_bits, state_at, steps_description_bits};

fn main() {
    let n = 64usize;
    println!("n = {n}, coffee automaton from m04_coffee");
    println!("  t           shortcut bits   compressed state bits   ratio");
    for t in [1_000usize, 10_000, 100_000, 1_000_000, 10_000_000] {
        let grid = state_at(42, t, n);
        let state_bits = compressed_len(&grid.cells) as f64 * 8.0;
        let sc = shortcut_bits(t);
        println!("  {t:>9}   {sc:>12.0}   {state_bits:>21.0}   {:>6.0}x", state_bits / sc);
    }
    println!();
    println!("  log2 of the step count alone:");
    for t in [1_000usize, 1_000_000, 1_000_000_000] {
        println!("    t = {t:>10}: {:.0} bits", steps_description_bits(t));
    }
}
