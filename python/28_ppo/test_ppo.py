"""Tests for the PPO slice. Run with: python3 test_ppo.py

The clipped surrogate is arithmetic and is tested exactly. The learning result is on
CartPole — which, importantly, is *not* one of the paper's benchmarks; the paper tests
locomotion and Atari. CartPole is used here because it runs in seconds on a CPU.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from implementation import (  # noqa: E402
    ActorCritic, clipped_surrogate, evaluate, generalised_advantage, train,
)

FAILURES = []


def check(name, condition, detail=""):
    status = "ok  " if condition else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(name)


def grad_of_ratio(ratio, advantage, clip=0.2):
    """dloss/dlogp at a given ratio, via the same function training uses."""
    logp = np.array([np.log(ratio)])
    logp_old = np.array([0.0])
    _, d = clipped_surrogate(logp, logp_old, np.array([advantage]), clip)
    return float(d[0])


def test_clip_is_exact():
    """Outside the clip range the gradient is exactly zero — when that branch is taken.

    Which branch wins depends on the sign of the advantage, so both signs are checked.
    """
    eps = 0.2
    # ratio above 1+eps, positive advantage: clipped branch wins, gradient is 0
    check("r > 1+eps, A > 0 -> gradient exactly 0",
          grad_of_ratio(1.5, 1.0) == 0.0, f"{grad_of_ratio(1.5, 1.0)}")
    # ratio above 1+eps, negative advantage: unclipped wins, gradient is -rA
    check("r > 1+eps, A < 0 -> gradient -rA",
          abs(grad_of_ratio(1.5, -1.0) - 1.5) < 1e-12, f"{grad_of_ratio(1.5, -1.0)}")
    # ratio below 1-eps, negative advantage: clipped branch wins, gradient is 0
    check("r < 1-eps, A < 0 -> gradient exactly 0",
          grad_of_ratio(0.5, -1.0) == 0.0, f"{grad_of_ratio(0.5, -1.0)}")
    # ratio below 1-eps, positive advantage: unclipped wins, gradient is -rA
    check("r < 1-eps, A > 0 -> gradient -rA",
          abs(grad_of_ratio(0.5, 1.0) + 0.5) < 1e-12, f"{grad_of_ratio(0.5, 1.0)}")
    # inside the range the gradient is always -rA
    for r in (0.85, 1.0, 1.15):
        check(f"r = {r} inside -> gradient -rA",
              abs(grad_of_ratio(r, 1.0) + r) < 1e-12, f"{grad_of_ratio(r, 1.0)}")


def test_ppo_solves_cartpole():
    model = ActorCritic(seed=0)
    history = train(model, iterations=60, episodes=8, epochs=3, batch=64, lr=3e-3, seed=0)
    final = evaluate(model, episodes=30, seed=99)          # greedy by default
    sampled = evaluate(model, episodes=30, seed=99, greedy=False)
    check("training improves the return", history[-1] > history[0],
          f"{history[0]:.1f} -> {history[-1]:.1f}")
    check("greedy evaluation reaches the CartPole ceiling", final > 475.0,
          f"mean return {final:.1f} (cap 500); sampled {sampled:.1f}")


def test_training_is_deterministic():
    a = train(ActorCritic(seed=3), iterations=5, episodes=4, epochs=2, batch=32, seed=11)
    b = train(ActorCritic(seed=3), iterations=5, episodes=4, epochs=2, batch=32, seed=11)
    check("same seeds give the same trace", np.allclose(a, b),
          f"first values {a[0]:.4f} vs {b[0]:.4f}")


def test_gae_boundaries():
    """GAE interpolates between the TD error and the Monte Carlo advantage."""
    rewards = np.array([1.0, 1.0, 1.0])
    values = np.array([0.5, 0.8, 0.2])
    dones = np.array([0.0, 0.0, 1.0])

    lam0 = generalised_advantage(rewards, values, dones, gamma=0.99, lam=0.0)
    td = rewards + 0.99 * np.append(values[1:], 0.0) * (1 - dones) - values
    check("lambda = 0 reduces GAE to the TD error", np.allclose(lam0, td, atol=1e-12),
          f"{np.round(lam0, 4)} vs {np.round(td, 4)}")

    lam1 = generalised_advantage(rewards, values, dones, gamma=0.99, lam=1.0)
    ret = np.array([1 + 0.99 + 0.99 ** 2, 1 + 0.99, 1.0])
    mc = ret - values
    check("lambda = 1 reduces GAE to the Monte Carlo advantage",
          np.allclose(lam1, mc, atol=1e-12), f"{np.round(lam1, 4)} vs {np.round(mc, 4)}")


if __name__ == "__main__":
    test_clip_is_exact()
    test_gae_boundaries()
    test_ppo_solves_cartpole()
    test_training_is_deterministic()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} test(s) failed: {', '.join(FAILURES)}")
        sys.exit(1)
    print("all tests passed")
