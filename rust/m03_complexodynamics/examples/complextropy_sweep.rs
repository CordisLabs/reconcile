//! Complextropy across grain sizes and run lengths.
//!
//! One measurement is an anecdote. If the shape is a property of the model it should not
//! depend on the particular grain or on where the run is stopped.

use m03_complexodynamics::complextropy;
use m04_coffee::{Grid, Rng};

fn curve(n: usize, grain: usize, steps: usize, samples: usize, seed: u64) -> Vec<f64> {
    let mut grid = Grid::layered(n);
    let mut rng = Rng::new(seed);
    let mut t = 0;
    let mut out = Vec::new();
    for k in 0..=samples {
        let target = steps * k / samples;
        while t < target {
            grid.step(&mut rng);
            t += 1;
        }
        out.push(complextropy(&grid, grain));
    }
    out
}

fn main() {
    let n = 48;
    let samples = 20;
    println!("grain  steps   start   peak(sample)    end    fall from peak");
    for steps in [4_000_000usize, 12_000_000] {
        for grain in [6usize, 10, 14] {
            let c = curve(n, grain, steps, samples, 11);
            let (pi, &peak) = c
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();
            let end = c[samples];
            println!(
                "{grain:>5} {steps:>8} {:>7.0} {:>6} ({pi:>2}) {:>7.0} {:>12.0}%",
                c[0],
                peak,
                end,
                (peak - end) / peak * 100.0
            );
        }
    }
}
