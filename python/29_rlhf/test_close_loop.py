"""Tests for the closed RLHF loop. Run with: python3 test_close_loop.py

The judge is the hidden reward: preferences are generated from it, never shown, and the
policy is optimised only against a reward model fitted from those comparisons. If the
policy's true return approaches the optimum, the loop worked.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from close_loop import (  # noqa: E402
    OPTIMAL,
    RANDOM,
    RewardModel,
    counts_of,
    metrics,
    ppo,
    preference_pairs,
    run,
    sample_sequences,
    train_reward,
)

FAILURES = []


def check(name, condition, detail=""):
    status = "ok  " if condition else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(name)


def test_reward_model_fits_preferences():
    rng = np.random.default_rng(0)
    a, b, label, _ = preference_pairs(rng, 4000)
    model = RewardModel()
    loss = train_reward(model, a, b, label)
    a2, b2, l2, _ = preference_pairs(rng, 2000)
    ra, _ = model.forward(a2)
    rb, _ = model.forward(b2)
    acc = float(np.mean((ra > rb).astype(int) == l2))
    check("Bradley-Terry loss falls", loss < 0.5, f"{loss:.3f}")
    check("held-out preference accuracy > 0.85", acc > 0.85, f"{acc:.4f}")


def test_policy_recovers_the_hidden_objective():
    out, _ = run(seed=0)
    check(
        "policy's true return approaches the optimum",
        out["true_mean"] > 0.9 * OPTIMAL,
        f"{out['true_mean']:.3f} of {OPTIMAL}",
    )
    check(
        "policy beats random by a wide margin",
        out["true_mean"] > 2 * RANDOM,
        f"{out['true_mean']:.3f} vs random {RANDOM:.3f}",
    )
    check(
        "policy emits the best token almost always",
        out["best_token_freq"] > 0.9,
        f"{out['best_token_freq']:.4f}",
    )


def test_learned_reward_reproduces_the_ordering_not_the_scale():
    """Bradley-Terry fixes differences up to a scale, so the absolute level can drift.

    This is the mechanism behind reward hacking in the real pipeline: a policy maximising
    the learned reward can push past the true one, even though the ordering transfers.
    """
    out, _ = run(seed=0)
    check(
        "learned reward scale differs from the true one",
        abs(out["learned_mean"] - out["true_mean"]) > 0.5,
        f"learned {out['learned_mean']:.2f} vs true {out['true_mean']:.2f}",
    )
    check(
        "but the ordering still transferred",
        out["learned_mean"] > 0.9 * out["true_mean"],
        "policy achieves near-optimal true return despite the drift",
    )


if __name__ == "__main__":
    test_reward_model_fits_preferences()
    test_policy_recovers_the_hidden_objective()
    test_learned_reward_reproduces_the_ordering_not_the_scale()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} test(s) failed: {', '.join(FAILURES)}")
        sys.exit(1)
    print("all tests passed")
