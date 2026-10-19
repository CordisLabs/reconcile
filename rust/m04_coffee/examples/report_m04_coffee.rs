use m04_coffee::{coarse_grained, compressed_len, order2_bits, Grid, Rng};

fn main() {
    let n = 64usize;
    let g = 12usize;
    let steps = 10_000_000usize;
    let mut grid = Grid::layered(n);
    let mut rng = Rng::new(42);
    println!("n = {n}, grain = {g}, swaps = {steps}");
    println!("  step      entropy(fine)   complexity(LZ77)   complexity(order2)   dark");
    let mut t = 0usize;
    let mut curve = Vec::new();
    let mut curve2 = Vec::new();
    for k in 0..=30 {
        let target = steps * k / 30;
        while t < target {
            grid.step(&mut rng);
            t += 1;
        }
        let e = compressed_len(&grid.cells);
        let coarse = coarse_grained(&grid, g);
        let c = compressed_len(&coarse);
        let c2 = order2_bits(&coarse) / 8.0;      // bytes, to sit beside the LZ77 figure
        curve.push((t, e, c));
        curve2.push((t, c2));
        if k % 3 == 0 || k == 30 {
            println!("  {:>7}   {:>13}   {:>18}   {:>14.1}   {:.4}",
                t, e, c, curve2.last().unwrap().1, grid.dark_fraction());
        }
    }
    let peak = curve.iter().max_by_key(|r| r.2).copied().unwrap();
    println!();
    println!("  LZ77 complexity peaks at step {} ({} bytes); starts {}, ends {}",
        peak.0, peak.2, curve.first().unwrap().2, curve.last().unwrap().2);
    let peak2 = curve2.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
    println!("  order-2 complexity peaks at step {} ({:.1} bytes); starts {:.1}, ends {:.1}",
        peak2.0, peak2.1, curve2.first().unwrap().1, curve2.last().unwrap().1);
    let max_drop: f64 = curve
        .windows(2)
        .map(|w| (w[0].1 as f64 - w[1].1 as f64) / w[0].1 as f64)
        .fold(0.0, f64::max);
    println!(
        "  largest single entropy drop between samples: {:.1}% (compression noise)",
        max_drop * 100.0
    );
}
