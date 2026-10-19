use m03_complexodynamics::complextropy;
use m04_coffee::{Grid, Rng};
fn main() {
    let n = 48; let grain = 10;
    let mut grid = Grid::layered(n);
    let mut rng = Rng::new(11);
    let total = 4_000_000usize; let samples = 20usize;
    let mut t = 0;
    let mut curve = Vec::new();
    for k in 0..=samples {
        let target = total * k / samples;
        while t < target { grid.step(&mut rng); t += 1; }
        curve.push(complextropy(&grid, grain));
    }
    let peak = curve.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let pi = curve.iter().position(|&x| x == peak).unwrap();
    println!("start {:.0}, peak {:.0} at sample {}/{}, end {:.0}",
             curve[0], peak, pi, samples, curve[samples]);
    println!("curve: {:?}", curve.iter().map(|x| format!("{:.0}", x)).collect::<Vec<_>>());
}
