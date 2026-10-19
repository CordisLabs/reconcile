//! Machine Super Intelligence (Shane Legg, 2008) — the universal intelligence measure.
//!
//! Legg defines the intelligence of an agent as a complexity-weighted average of its
//! expected reward over all environments:
//!
//! ```text
//! Upsilon(pi) = sum_mu 2^{-K(mu)} V_mu^pi
//! ```
//!
//! where `K(mu)` is the Kolmogorov complexity of environment `mu` and `V_mu^pi` is the
//! expected reward of policy `pi` in it.
//!
//! `K` is not computable, so this implementation works over a **finite environment class
//! with a chosen encoding**, which is exactly the simplification the thesis says is
//! unavoidable in practice. What remains testable is the structure: the measure is
//! bounded, the optimal agent scores above a random one, and the ranking of agents does
//! not depend on how the environments happen to be encoded — changing every complexity
//! by the same additive constant rescales Upsilon by a constant factor and leaves the
//! ordering untouched.

#![forbid(unsafe_code)]

#[derive(Clone, Copy, Debug)]
pub struct Bandit {
    /// Reward for each of the two actions; deterministic so V is exact.
    pub rewards: [f64; 2],
    /// Bits needed to describe this environment under the chosen encoding.
    pub complexity: u32,
}

impl Bandit {
    /// Expected reward of a policy that picks action 0 with probability p.
    pub fn value(&self, p_action0: f64) -> f64 {
        p_action0 * self.rewards[0] + (1.0 - p_action0) * self.rewards[1]
    }

    pub fn optimal_value(&self) -> f64 {
        self.rewards[0].max(self.rewards[1])
    }

    /// Re-encoding: the same environment described with `extra` more bits.
    pub fn reencoded(&self, extra: u32) -> Self {
        Bandit {
            rewards: self.rewards,
            complexity: self.complexity + extra,
        }
    }
}

/// Upsilon without normalisation: sum_mu 2^{-K(mu)} V_mu^pi.
pub fn upsilon(envs: &[Bandit], p_action0: f64) -> f64 {
    envs.iter()
        .map(|e| 2f64.powi(-(e.complexity as i32)) * e.value(p_action0))
        .sum()
}

/// Upsilon as an average: sum_mu 2^{-K(mu)} V / sum_mu 2^{-K(mu)}.
///
/// This is the form that stays in the reward range and is invariant to a uniform shift
/// of every complexity.
pub fn upsilon_normalised(envs: &[Bandit], p_action0: f64) -> f64 {
    let z: f64 = envs.iter().map(|e| 2f64.powi(-(e.complexity as i32))).sum();
    upsilon(envs, p_action0) / z
}

/// Some environments where the two actions differ, so policy choice can matter.
pub fn example_environments() -> Vec<Bandit> {
    vec![
        Bandit { rewards: [1.0, 0.0], complexity: 1 },
        Bandit { rewards: [0.0, 1.0], complexity: 1 },
        Bandit { rewards: [0.8, 0.2], complexity: 3 },
        Bandit { rewards: [0.3, 0.7], complexity: 3 },
        Bandit { rewards: [0.6, 0.6], complexity: 5 },
        Bandit { rewards: [0.9, 0.1], complexity: 7 },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upsilon_is_bounded_by_the_reward_range() {
        let envs = example_environments();
        for p in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let u = upsilon_normalised(&envs, p);
            assert!(
                (0.0..=1.0).contains(&u),
                "Upsilon = {u} outside [0,1] for p = {p}"
            );
        }
    }

    /// The optimal policy of each environment cannot score worse than a fixed policy.
    #[test]
    fn a_uniform_optimal_policy_beats_random() {
        let envs = example_environments();
        // "random" picks each action with probability 1/2 in every environment
        let random_u = upsilon_normalised(&envs, 0.5);
        // "oracle" picks the better action in each environment; as a single fixed p it
        // cannot do that, so the oracle is computed per environment instead
        let z: f64 = envs.iter().map(|e| 2f64.powi(-(e.complexity as i32))).sum();
        let oracle_u: f64 = envs
            .iter()
            .map(|e| 2f64.powi(-(e.complexity as i32)) * e.optimal_value())
            .sum::<f64>()
            / z;
        assert!(
            oracle_u > random_u,
            "oracle {oracle_u:.4} should beat random {random_u:.4}"
        );
    }

    /// The paper's invariance claim: re-encoding all environments uniformly is a
    /// constant rescaling, not a change of ranking.
    #[test]
    fn ranking_is_invariant_under_reencoding() {
        let envs = example_environments();
        let base: Vec<f64> = [0.0, 0.3, 0.5, 0.7, 1.0]
            .iter()
            .map(|&p| upsilon_normalised(&envs, p))
            .collect();
        for extra in [1u32, 5, 12] {
            let shifted: Vec<Bandit> = envs.iter().map(|e| e.reencoded(extra)).collect();
            let u: Vec<f64> = [0.0, 0.3, 0.5, 0.7, 1.0]
                .iter()
                .map(|&p| upsilon_normalised(&shifted, p))
                .collect();
            for (a, b) in base.iter().zip(&u) {
                assert!(
                    (a - b).abs() < 1e-12,
                    "normalised Upsilon changed under re-encoding: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn unnormalised_upsilon_scales_by_the_encoding_shift() {
        let envs = example_environments();
        let base = upsilon(&envs, 0.5);
        for extra in [1u32, 4] {
            let shifted: Vec<Bandit> = envs.iter().map(|e| e.reencoded(extra)).collect();
            let u = upsilon(&shifted, 0.5);
            let expected = base * 2f64.powi(-(extra as i32));
            assert!(
                (u - expected).abs() < 1e-12,
                "extra {extra}: {u} vs expected {expected}"
            );
        }
    }
}

/// The best policy that cannot tell environments apart: a single mixing probability.
///
/// This is the middle rung between a random agent and an oracle. It is found by a grid
/// search, which is enough because `Upsilon` is a smooth function of one variable here.
pub fn best_fixed_policy(envs: &[Bandit]) -> (f64, f64) {
    let mut best = (0.5, upsilon_normalised(envs, 0.5));
    for i in 0..=200 {
        let p = i as f64 / 200.0;
        let u = upsilon_normalised(envs, p);
        if u > best.1 {
            best = (p, u);
        }
    }
    best
}

/// The oracle: the better action in every environment.
pub fn oracle_upsilon(envs: &[Bandit]) -> f64 {
    let z: f64 = envs.iter().map(|e| 2f64.powi(-(e.complexity as i32))).sum();
    envs.iter()
        .map(|e| 2f64.powi(-(e.complexity as i32)) * e.optimal_value())
        .sum::<f64>()
        / z
}

#[cfg(test)]
mod ranking {
    use super::*;

    #[test]
    fn the_three_agents_are_ordered() {
        let envs = example_environments();
        let random = upsilon_normalised(&envs, 0.5);
        let (best_p, best) = best_fixed_policy(&envs);
        let oracle = oracle_upsilon(&envs);
        assert!(best >= random, "best fixed {best:.4} < random {random:.4}");
        assert!(oracle >= best, "oracle {oracle:.4} < best fixed {best:.4}");
        assert!(oracle > random + 0.05, "the gap should be visible");
        let _ = best_p;
    }

    /// The invariance claim applies to the whole ranking, not just to one number.
    #[test]
    fn the_whole_ranking_survives_reencoding() {
        let envs = example_environments();
        let rank = |es: &[Bandit]| {
            let random = upsilon_normalised(es, 0.5);
            let (_, best) = best_fixed_policy(es);
            let oracle = oracle_upsilon(es);
            [random, best, oracle]
        };
        let base = rank(&envs);
        for extra in [2u32, 9] {
            let shifted: Vec<Bandit> = envs.iter().map(|e| e.reencoded(extra)).collect();
            let got = rank(&shifted);
            for (a, b) in base.iter().zip(&got) {
                assert!((a - b).abs() < 1e-12, "re-encoding changed the ranking: {a} vs {b}");
            }
        }
    }
}
