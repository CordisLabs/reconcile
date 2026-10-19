//! Noisy weights (paper §5) — the half of the paper the deterministic part above
//! does not touch.
//!
//! Mechanism, as the paper states it:
//!   * sender and receiver share a Gaussian prior P over each weight;
//!   * after learning the sender holds a Gaussian posterior Q;
//!   * by the "bits back" argument the cost of communicating that weight is the
//!     Kullback-Leibler divergence G(P;Q) (§5.1, equation 5);
//!   * high-variance weights are cheap to send but inject noise into the outputs,
//!     which makes the data mis-fits expensive — so the total has an interior
//!     optimum in the noise level.
//!
//! The paper's technical contribution is that with linear output units the
//! expectations under weight noise and their derivatives can be computed exactly,
//! without Monte Carlo. That is what this module does: a tanh hidden layer whose
//! pre-activation is Gaussian, integrated with Gauss-Hermite quadrature, with
//! analytic derivatives obtained from two Gaussian identities (Stein / Price):
//!
//! ```text
//! d/dm  E[f(a)] = E[f'(a)]
//! d/dv  E[f(a)] = 1/2 E[f''(a)]      (v = variance of a)
//! ```
//!
//! Both are implemented as expectations of the appropriate derivative, so no
//! finite differences enter the gradient.

use std::f64::consts::PI;

use crate::Rng;

/// Quadrature against the standard normal density.
///
/// Gauss-Hermite was the first choice and it is the classical one, but it converges
/// slowly here: `tanh` has poles at `+-i pi/2`, and after scaling by `sqrt(2 v)` those
/// poles sit close enough to the real axis that 48 nodes leave a 4e-6 error in
/// `E[tanh]` at variance 2.5 — enough to corrupt the gradient. The trapezoidal rule on
/// the Gaussian weight has no such problem: the integrand is analytic and decays like
/// `e^{-z^2/2}`, so the error falls off like `e^{-c/h}` and 256 nodes reach machine
/// precision. It also needs no root finding, which removes a whole class of silent
/// failure (a missed root is invisible once the weights are normalised).
pub struct Quadrature {
    z: Vec<f64>,
    w: Vec<f64>,
}

impl Quadrature {
    pub fn new(n: usize) -> Self {
        let n = n.max(64);
        let a = 10.0;
        let h = 2.0 * a / n as f64;
        let norm = (2.0 * PI).sqrt();
        let mut z = Vec::with_capacity(n + 1);
        let mut w = Vec::with_capacity(n + 1);
        for i in 0..=n {
            let zi = -a + i as f64 * h;
            let edge = if i == 0 || i == n { 0.5 } else { 1.0 };
            z.push(zi);
            w.push(edge * h * (-0.5 * zi * zi).exp() / norm);
        }
        Quadrature { z, w }
    }

    /// E[f(a)] for a ~ N(mean, variance).
    pub fn expect(&self, mean: f64, variance: f64, f: impl Fn(f64) -> f64) -> f64 {
        let s = variance.sqrt();
        self.z
            .iter()
            .zip(&self.w)
            .map(|(z, w)| w * f(mean + s * z))
            .sum()
    }
}

/// Moments of tanh(a) and of its square, plus the Gaussian-identity derivatives.
#[derive(Clone, Copy, Debug)]
pub struct TanhMoments {
    /// E[tanh a]
    pub a: f64,
    /// E[tanh^2 a]
    pub b: f64,
    /// d/dmean E[tanh a] = E[1 - tanh^2 a]
    pub da_dm: f64,
    /// d/dvar E[tanh a] = -E[tanh a (1 - tanh^2 a)]
    pub da_dv: f64,
    /// d/dmean E[tanh^2 a] = 2 E[tanh a (1 - tanh^2 a)]
    pub db_dm: f64,
    /// d/dvar E[tanh^2 a] = E[(1 - tanh^2)^2] - 2 E[tanh^2 (1 - tanh^2)]
    pub db_dv: f64,
}

pub fn tanh_moments(q: &Quadrature, mean: f64, variance: f64) -> TanhMoments {
    // One pass over the nodes accumulates all six integrands.
    let s = variance.sqrt();
    let mut a = 0.0;
    let mut b = 0.0;
    let mut e_s = 0.0;
    let mut e_t_s = 0.0;
    let mut e_s2 = 0.0;
    let mut e_t2_s = 0.0;
    for (z, w) in q.z.iter().zip(&q.w) {
        let x = mean + s * z;
        let t = x.tanh();
        let sech2 = 1.0 - t * t;
        a += w * t;
        b += w * t * t;
        e_s += w * sech2;
        e_t_s += w * t * sech2;
        e_s2 += w * sech2 * sech2;
        e_t2_s += w * t * t * sech2;
    }
    TanhMoments {
        a,
        b,
        da_dm: e_s,
        da_dv: -e_t_s,
        db_dm: 2.0 * e_t_s,
        db_dv: e_s2 - 2.0 * e_t2_s,
    }
}

pub struct Config {
    /// Prior variance sigma_p^2, fixed in advance as the paper requires.
    pub sigma_p2: f64,
    /// Variance of the target noise, fixing the exchange rate between nats and
    /// squared error.
    pub sigma_d2: f64,
}

#[derive(Clone)]
pub struct NoisyModel {
    pub inputs: usize,
    pub hidden: usize,
    /// Hidden weight means, `hidden x inputs`.
    pub mu: Vec<Vec<f64>>,
    /// Hidden weight log standard deviations, same shape.
    pub rho: Vec<Vec<f64>>,
    /// Linear output weights, one per hidden unit.
    pub v: Vec<f64>,
    pub c: f64,
}

#[derive(Clone, Debug)]
pub struct Gradient {
    pub dmu: Vec<Vec<f64>>,
    pub drho: Vec<Vec<f64>>,
    pub dv: Vec<f64>,
    pub dc: f64,
}

impl NoisyModel {
    pub fn new(inputs: usize, hidden: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        Self {
            inputs,
            hidden,
            mu: (0..hidden)
                .map(|_| (0..inputs).map(|_| 0.1 * rng.normal()).collect())
                .collect(),
            // Start with unit noise: s = 1.
            rho: vec![vec![0.0; inputs]; hidden],
            v: (0..hidden).map(|_| 0.1 * rng.normal()).collect(),
            c: 0.0,
        }
    }

    pub fn sigma(&self, h: usize, d: usize) -> f64 {
        self.rho[h][d].exp()
    }

    /// Mean and variance of the network output, exactly.
    pub fn predict(&self, x: &[f64], gh: &Quadrature) -> (f64, f64) {
        let mut mean = self.c;
        let mut var = 0.0;
        for h in 0..self.hidden {
            let mut m = 0.0;
            let mut v_act = 0.0;
            for d in 0..self.inputs {
                let s = self.sigma(h, d);
                m += self.mu[h][d] * x[d];
                v_act += s * s * x[d] * x[d];
            }
            let mom = tanh_moments(gh, m, v_act);
            mean += self.v[h] * mom.a;
            let unit_var = (mom.b - mom.a * mom.a).max(0.0);
            var += self.v[h] * self.v[h] * unit_var;
        }
        (mean, var)
    }

    /// Sum of (mean - target)^2 + variance over the data: the expected squared error
    /// under the weight noise, without the KL terms.
    pub fn expected_error(&self, x: &[Vec<f64>], t: &[f64], gh: &Quadrature) -> f64 {
        x.iter()
            .zip(t)
            .map(|(xn, tn)| {
                let (mean, var) = self.predict(xn, gh);
                (mean - tn) * (mean - tn) + var
            })
            .sum()
    }

    /// Total description length in nats: KL(Q||P) for every noisy weight, a plain
    /// Gaussian prior term for the deterministic output weights, and the expected
    /// data mis-fit under the noise.
    pub fn cost(&self, x: &[Vec<f64>], t: &[f64], gh: &Quadrature, cfg: &Config) -> f64 {
        let (total, _, _) = self.cost_and_gradient(x, t, gh, cfg, false);
        total
    }

    fn cost_and_gradient(
        &self,
        x: &[Vec<f64>],
        t: &[f64],
        gh: &Quadrature,
        cfg: &Config,
        want_grad: bool,
    ) -> (f64, Option<Gradient>, f64) {
        let mut kl = 0.0;
        for h in 0..self.hidden {
            for d in 0..self.inputs {
                let s = self.sigma(h, d);
                let mu = self.mu[h][d];
                // KL(N(mu, s^2) || N(0, sigma_p^2))
                kl += (cfg.sigma_p2.sqrt() / s).ln()
                    + (s * s + mu * mu) / (2.0 * cfg.sigma_p2)
                    - 0.5;
            }
        }
        // The output weights are not noisy; they still pay a Gaussian prior, which is
        // the s -> sigma_p limit of the same expression.
        for v in &self.v {
            kl += v * v / (2.0 * cfg.sigma_p2);
        }
        kl += self.c * self.c / (2.0 * cfg.sigma_p2);

        let mut grad = Gradient {
            dmu: vec![vec![0.0; self.inputs]; self.hidden],
            drho: vec![vec![0.0; self.inputs]; self.hidden],
            dv: vec![0.0; self.hidden],
            dc: 0.0,
        };
        let scale = 1.0 / (2.0 * cfg.sigma_d2);
        let mut expected_error = 0.0;

        for (xn, tn) in x.iter().zip(t) {
            let mut mean = self.c;
            let mut mom_c: Vec<TanhMoments> = Vec::with_capacity(self.hidden);
            for h in 0..self.hidden {
                let mut m = 0.0;
                let mut v_act = 0.0;
                for d in 0..self.inputs {
                    let s = self.sigma(h, d);
                    m += self.mu[h][d] * xn[d];
                    v_act += s * s * xn[d] * xn[d];
                }
                let mom = tanh_moments(gh, m, v_act);
                mean += self.v[h] * mom.a;
                mom_c.push(mom);
            }
            let mut var = 0.0;
            for h in 0..self.hidden {
                var += self.v[h] * self.v[h] * (mom_c[h].b - mom_c[h].a * mom_c[h].a).max(0.0);
            }
            let residual = mean - tn;
            expected_error += residual * residual + var;

            if want_grad {
                grad.dc += 2.0 * residual;
                for h in 0..self.hidden {
                    let mom = mom_c[h];
                    let unit_var = (mom.b - mom.a * mom.a).max(0.0);
                    grad.dv[h] += 2.0 * residual * mom.a + 2.0 * self.v[h] * unit_var;
                    for d in 0..self.inputs {
                        let s = self.sigma(h, d);
                        // d(mean)/d(mu) = v * E[sech^2] * x
                        grad.dmu[h][d] += 2.0 * residual * self.v[h] * mom.da_dm * xn[d];
                        // d(expected error)/d(mean of a) picks up the variance term too
                        grad.dmu[h][d] += self.v[h] * self.v[h]
                            * (mom.db_dm - 2.0 * mom.a * mom.da_dm)
                            * xn[d];
                        // variance of the pre-activation: d(v_act)/d(rho) = 2 s^2 x^2
                        let dvd = 2.0 * s * s * xn[d] * xn[d];
                        let mut g = 2.0 * residual * self.v[h] * mom.da_dv;
                        g += self.v[h] * self.v[h] * (mom.db_dv - 2.0 * mom.a * mom.da_dv);
                        grad.drho[h][d] += g * dvd;
                    }
                }
            }
        }

        let total = kl + expected_error * scale;
        if want_grad {
            grad.dc *= scale;
            grad.dc += self.c / cfg.sigma_p2;
            for h in 0..self.hidden {
                grad.dv[h] *= scale;
                grad.dv[h] += self.v[h] / cfg.sigma_p2;
                for d in 0..self.inputs {
                    grad.dmu[h][d] *= scale;
                    grad.dmu[h][d] += self.mu[h][d] / cfg.sigma_p2;
                    let s = self.sigma(h, d);
                    grad.drho[h][d] *= scale;
                    grad.drho[h][d] += -1.0 + s * s / cfg.sigma_p2;
                }
            }
            (total, Some(grad), expected_error)
        } else {
            (total, None, expected_error)
        }
    }

    pub fn gradient(
        &self,
        x: &[Vec<f64>],
        t: &[f64],
        gh: &Quadrature,
        cfg: &Config,
    ) -> (f64, Gradient) {
        let (c, g, _) = self.cost_and_gradient(x, t, gh, cfg, true);
        (c, g.unwrap())
    }

    /// Gradient descent on the total description length with a bold-driver step
    /// rule: accept the step and grow the learning rate when the cost falls, else
    /// restore and halve. Plain fixed-step descent diverges here because the
    /// curvature in the weight-noise parameters spans two orders of magnitude.
    pub fn fit(
        &mut self,
        x: &[Vec<f64>],
        t: &[f64],
        q: &Quadrature,
        cfg: &Config,
        steps: usize,
        lr: f64,
    ) -> Vec<f64> {
        let mut trace = Vec::with_capacity(steps);
        let mut lr = lr;
        for _ in 0..steps {
            let (cost, g) = self.gradient(x, t, q, cfg);
            trace.push(cost);
            let snapshot = self.clone();
            self.step(&g, lr);
            let next = self.cost(x, t, q, cfg);
            if next <= cost {
                lr = (lr * 1.05).min(0.5);
            } else {
                *self = snapshot;
                lr *= 0.5;
            }
            if lr < 1e-12 {
                break;
            }
        }
        trace
    }

    fn step(&mut self, g: &Gradient, lr: f64) {
        self.c -= lr * g.dc;
        for h in 0..self.hidden {
            self.v[h] -= lr * g.dv[h];
            for d in 0..self.inputs {
                self.mu[h][d] -= lr * g.dmu[h][d];
                self.rho[h][d] -= lr * g.drho[h][d];
            }
        }
    }

    /// Monte-Carlo estimate of the expected squared error and its standard error,
    /// used only to check the exact computation above.
    pub fn monte_carlo_error(
        &self,
        x: &[Vec<f64>],
        t: &[f64],
        samples: usize,
        seed: u64,
    ) -> (f64, f64) {
        let mut rng = Rng::new(seed);
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for _ in 0..samples {
            let mut err = 0.0;
            for (xn, tn) in x.iter().zip(t) {
                let mut y = self.c;
                for h in 0..self.hidden {
                    let mut a = 0.0;
                    for d in 0..self.inputs {
                        a += (self.mu[h][d] + self.sigma(h, d) * rng.normal()) * xn[d];
                    }
                    y += self.v[h] * a.tanh();
                }
                err += (y - tn) * (y - tn);
            }
            sum += err;
            sum_sq += err * err;
        }
        let n = samples as f64;
        let mean = sum / n;
        let var = (sum_sq / n - mean * mean).max(0.0);
        (mean, (var / n).sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIGMA_P2: f64 = 1.0;
    const SIGMA_D2: f64 = 0.01;

    fn gh() -> Quadrature {
        Quadrature::new(256)
    }

    fn cfg() -> Config {
        Config {
            sigma_p2: SIGMA_P2,
            sigma_d2: SIGMA_D2,
        }
    }

    fn toy() -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut x = Vec::new();
        let mut t = Vec::new();
        for i in 0..10 {
            let a = i as f64 * 0.2 - 1.0;
            let b = (i as f64 * 0.37).cos();
            x.push(vec![a, b]);
            t.push(0.8 * (1.2 * a - 0.8 * b).tanh() + 0.05 * (i as f64 * 1.7).sin());
        }
        (x, t)
    }

    fn toy_model() -> NoisyModel {
        let mut m = NoisyModel::new(2, 2, 3);
        for h in 0..2 {
            for d in 0..2 {
                m.rho[h][d] = -0.3 + 0.2 * (h * 2 + d) as f64;
            }
        }
        m
    }

    #[test]
    fn quadrature_is_exact_for_moments() {
        let g = gh();
        let (m, var) = (0.7, 0.4);
        assert!((g.expect(m, var, |_| 1.0) - 1.0).abs() < 1e-14);
        assert!((g.expect(m, var, |z| z) - m).abs() < 1e-12);
        assert!((g.expect(m, var, |z| z * z) - (m * m + var)).abs() < 1e-12);
        assert!((g.expect(m, var, |z| z * z * z) - (m * m * m + 3.0 * m * var)).abs() < 1e-11);
        assert!(
            (g.expect(m, var, |z| z.powi(4)) - (m.powi(4) + 6.0 * m * m * var + 3.0 * var * var))
                .abs()
                < 1e-10
        );
    }

    /// The two Gaussian identities the analytic gradient rests on:
    /// d/dm E[f] = E[f'] and d/dvar E[f] = 1/2 E[f''].
    #[test]
    fn gaussian_identities_hold() {
        let g = gh();
        let (m, var) = (0.3, 0.8);
        let mom = tanh_moments(&g, m, var);

        let h = 1e-6;
        let num_dm = (tanh_moments(&g, m + h, var).a - tanh_moments(&g, m - h, var).a) / (2.0 * h);
        assert!(
            (num_dm - mom.da_dm).abs() < 1e-8,
            "d/dm: analytic {} vs numeric {}",
            mom.da_dm,
            num_dm
        );

        let hv = var * 1e-5;
        let num_dv = (tanh_moments(&g, m, var + hv).a - tanh_moments(&g, m, var - hv).a)
            / (2.0 * hv);
        assert!(
            (num_dv - mom.da_dv).abs() < 1e-6,
            "d/dvar: analytic {} vs numeric {}",
            mom.da_dv,
            num_dv
        );
    }

    #[test]
    fn kl_divergence_is_non_negative() {
        let g = gh();
        let (x, t) = toy();
        let c = cfg();
        let mut m = toy_model();
        for (i, rho) in m.rho.iter_mut().enumerate() {
            for (j, r) in rho.iter_mut().enumerate() {
                *r = -2.0 + 0.5 * (i * 3 + j) as f64;
            }
        }
        // cost minus the data term is the code length of the weights alone.
        let total = m.cost(&x, &t, &g, &c);
        let data = m.expected_error(&x, &t, &g) / (2.0 * c.sigma_d2);
        let kl = total - data;
        assert!(kl >= -1e-12, "negative code length for the weights: {kl}");
    }

    #[test]
    fn gradient_matches_finite_differences() {
        let g = gh();
        let c = cfg();
        let (x, t) = toy();
        let mut s = toy_model();
        let (_, grad) = s.gradient(&x, &t, &g, &c);

        let eps = 1e-6;
        let mut worst_abs: f64 = 0.0;
        let mut worst_rel: f64 = 0.0;
        let mut cmp = |ana: f64, num: f64| {
            let d = (ana - num).abs();
            worst_abs = worst_abs.max(d);
            let scale = ana.abs().max(num.abs());
            if scale > 1e-5 {
                worst_rel = worst_rel.max(d / scale);
            }
        };

        for h in 0..s.hidden {
            let orig = s.v[h];
            s.v[h] = orig + eps;
            let cp = s.cost(&x, &t, &g, &c);
            s.v[h] = orig - eps;
            let cm = s.cost(&x, &t, &g, &c);
            s.v[h] = orig;
            cmp(grad.dv[h], (cp - cm) / (2.0 * eps));

            for d in 0..s.inputs {
                let o = s.mu[h][d];
                s.mu[h][d] = o + eps;
                let cp = s.cost(&x, &t, &g, &c);
                s.mu[h][d] = o - eps;
                let cm = s.cost(&x, &t, &g, &c);
                s.mu[h][d] = o;
                cmp(grad.dmu[h][d], (cp - cm) / (2.0 * eps));

                let o = s.rho[h][d];
                s.rho[h][d] = o + eps;
                let cp = s.cost(&x, &t, &g, &c);
                s.rho[h][d] = o - eps;
                let cm = s.cost(&x, &t, &g, &c);
                s.rho[h][d] = o;
                cmp(grad.drho[h][d], (cp - cm) / (2.0 * eps));
            }
        }

        let o = s.c;
        s.c = o + eps;
        let cp = s.cost(&x, &t, &g, &c);
        s.c = o - eps;
        let cm = s.cost(&x, &t, &g, &c);
        s.c = o;
        cmp(grad.dc, (cp - cm) / (2.0 * eps));

        assert!(
            worst_rel < 1e-6,
            "gradient mismatch: max |d| = {worst_abs:.2e}, max rel = {worst_rel:.2e}"
        );
    }

    /// The paper's claim: exact expectations replace Monte Carlo. Check the exact
    /// expected squared error against a sampled estimate and its standard error.
    #[test]
    fn exact_expected_error_agrees_with_monte_carlo() {
        let g = gh();
        let (x, t) = toy();
        let m = toy_model();
        let exact = m.expected_error(&x, &t, &g);
        let (mc, se) = m.monte_carlo_error(&x, &t, 400_000, 11);
        let z = (exact - mc).abs() / se.max(1e-12);
        assert!(
            z < 4.0,
            "exact {exact:.6} vs MC {mc:.6} (se {se:.6}, z = {z:.2})"
        );
    }

    /// The paper's qualitative claim: high-variance weights are cheap to send but
    /// they make the data mis-fit expensive, so the learned noise level sits strictly
    /// inside the admissible range.
    #[test]
    fn noise_level_converges_to_an_interior_value() {
        let g = gh();
        let c = cfg();
        let (x, t) = toy();
        let mut m = NoisyModel::new(2, 2, 5);
        let trace = m.fit(&x, &t, &g, &c, 4000, 2e-3);
        assert!(
            trace.last().unwrap() < trace.first().unwrap(),
            "training did not reduce the description length"
        );

        for h in 0..m.hidden {
            for d in 0..m.inputs {
                let s = m.sigma(h, d);
                assert!(
                    s > 1e-3 && s < 5.0,
                    "noise level {s} ran to the boundary at ({h},{d})"
                );
            }
        }

        // Local optimality of the noise level: perturbing every rho by +-1 must cost more.
        let base = m.cost(&x, &t, &g, &c);
        let mut up = m.clone();
        let mut down = m.clone();
        for h in 0..m.hidden {
            for d in 0..m.inputs {
                up.rho[h][d] += 1.0;
                down.rho[h][d] -= 1.0;
            }
        }
        assert!(
            up.cost(&x, &t, &g, &c) > base && down.cost(&x, &t, &g, &c) > base,
            "the learned noise level is not a local optimum"
        );
    }
}
